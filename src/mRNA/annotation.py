"""Versioned annotation and lossless mapping diagnostics."""

import re
from importlib.resources import files

import numpy as np
import pandas as pd

from ._core import record, working


def strip_versions(ids):
    """Strip only terminal numeric versions from Ensembl gene/transcript IDs."""
    return pd.Index([re.sub(r"^(ENS[A-Z]*[GT]\d+)\.\d+$", r"\1", str(x)) for x in ids])


def reference(species, *, release="legacy"):
    """Load the archived human v43 or mouse v32 table, explicitly marked historical.

    Length definitions were not documented by the original package. These
    lengths must not be assumed to be quantifier-provided effective lengths.
    Supply a current versioned table for production annotation when appropriate.
    """
    if release != "legacy" or species not in {"human", "mouse"}:
        raise ValueError("Bundled references are legacy human v43 or mouse v32 only.")
    filename = {"human": "h38_gene_info_v43.csv.gz", "mouse": "m39_gene_info_v32.csv.gz"}[species]
    with files("mRNA._data").joinpath(filename).open("rb") as handle:
        table = pd.read_csv(handle, compression="gzip")
    table = table.rename(
        columns={
            "Ensembl": "ensembl",
            "Symbol": "symbol",
            "GeneType": "gene_type",
            "Length": "length",
        }
    )
    table.attrs["provenance"] = {
        "species": species,
        "release": release,
        "filename": filename,
        "length_definition": "undocumented",
    }
    return table


def map_ids(ids, reference, *, from_id="ensembl", to_id="symbol", strip_ensembl_versions=True):
    """Return one diagnostic row per input ID; ambiguous matches stay unresolved."""
    if from_id not in reference or to_id not in reference:
        raise KeyError("The reference must contain the requested identifier columns.")
    raw = pd.Index(ids)
    if raw.isna().any():
        raise ValueError("Input IDs must be nonmissing.")
    ids = raw.astype(str)
    ref = reference[[from_id, to_id]].dropna().astype(str).drop_duplicates()
    lookup_ids = strip_versions(ids) if from_id == "ensembl" and strip_ensembl_versions else ids
    if from_id == "ensembl" and strip_ensembl_versions:
        ref[from_id] = strip_versions(ref[from_id])
    lookup = ref.groupby(from_id, sort=False)[to_id].agg(lambda x: sorted(set(x)))
    rows = []
    for original, lookup_id in zip(ids, lookup_ids, strict=True):
        candidates = lookup.get(lookup_id, [])
        rows.append(
            {
                "input_id": original,
                "lookup_id": lookup_id,
                "mapped_id": candidates[0] if len(candidates) == 1 else "",
                "status": "mapped"
                if len(candidates) == 1
                else "ambiguous"
                if candidates
                else "unmapped",
                "n_candidates": len(candidates),
                "candidates": "|".join(candidates),
            }
        )
    out = pd.DataFrame(
        rows, columns=["input_id", "lookup_id", "mapped_id", "status", "n_candidates", "candidates"]
    )
    out.attrs["provenance"] = reference.attrs.get("provenance", {})
    return out


def annotate(
    adata, reference, *, from_id="ensembl", to_id="symbol", key_added="symbol", copy=False
):
    """Attach mapped symbols in var without renaming, merging, or dropping genes."""
    adata = working(adata, copy)
    mapping = map_ids(adata.var_names, reference, from_id=from_id, to_id=to_id)
    adata.var[key_added] = mapping["mapped_id"].to_numpy()
    adata.var[f"{key_added}_mapping_status"] = mapping["status"].to_numpy()
    record(
        adata,
        f"annotation_{key_added}",
        {
            "from_id": from_id,
            "to_id": to_id,
            "reference": reference.attrs.get("provenance", "user_supplied"),
        },
        table=mapping,
    )
    return adata


def aggregate_transcripts(
    adata,
    mapping,
    *,
    transcript_col="transcript",
    gene_col="gene",
    layer="counts",
    input_type="counts",
):
    """Sum raw transcript counts by an explicit, unambiguous mapping; reject omissions."""
    from anndata import AnnData
    from scipy import sparse

    from ._core import matrix

    if input_type not in {"counts", "estimated_counts"}:
        raise ValueError("Aggregate raw or estimated counts, not TPM/CPM.")
    m = matrix(adata, layer, input_type=input_type)
    refs = mapping[[transcript_col, gene_col]].drop_duplicates().copy()
    if refs.isna().any().any() or refs[transcript_col].duplicated().any():
        raise ValueError("Every transcript must map to exactly one nonmissing gene.")
    refs = refs.set_index(transcript_col).reindex(adata.var_names)
    if refs[gene_col].isna().any():
        raise ValueError("The mapping must cover every input transcript.")
    genes = pd.Index(pd.unique(refs[gene_col].astype(str)))
    codes = genes.get_indexer(refs[gene_col].astype(str))
    assignment = sparse.csr_matrix(
        (np.ones(adata.n_vars), (np.arange(adata.n_vars), codes)), shape=(adata.n_vars, len(genes))
    )
    counts = m @ assignment
    out = AnnData(X=counts, obs=adata.obs.copy(), var=pd.DataFrame(index=genes))
    out.layers[input_type] = counts.copy()
    out.uns["mRNA"] = {"representations": {"X": input_type, input_type: input_type}}
    record(out, "aggregate_transcripts", {"source_layer": layer}, table=refs.reset_index())
    return out
