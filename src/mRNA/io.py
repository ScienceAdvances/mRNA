"""Input and persistence of sample-by-gene bulk expression data."""

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata import read_h5ad as read_h5ad

from ._core import INPUT_TYPES, matrix, record, require_adata


def _align(frame, labels, name):
    frame = pd.DataFrame(index=labels) if frame is None else frame.copy()
    if not frame.index.is_unique or frame.index.isna().any():
        raise ValueError(f"{name} IDs must be unique and nonmissing.")
    frame.index = frame.index.astype(str)
    if set(frame.index) != set(labels):
        raise ValueError(f"{name} IDs must exactly match the expression matrix.")
    return frame.loc[labels].copy()


def from_frames(expression, metadata=None, annotation=None, *, input_type):
    """Build AnnData from labeled samples x genes, preserving raw counts once."""
    if input_type not in INPUT_TYPES:
        raise ValueError("Specify a supported input_type explicitly.")
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a labeled samples-by-genes DataFrame.")
    for labels in (expression.index, expression.columns):
        if not labels.is_unique or labels.isna().any():
            raise ValueError("Expression IDs must be unique and nonmissing.")
    sample_ids = expression.index.astype(str)
    gene_ids = expression.columns.astype(str)
    obs = _align(metadata, sample_ids, "Sample metadata")
    var = _align(annotation, gene_ids, "Gene annotation")
    adata = AnnData(X=expression.to_numpy(copy=True), obs=obs, var=var)
    matrix(adata, input_type=input_type)
    reps = {"X": input_type}
    if input_type == "counts":
        if (np.asarray(adata.X.sum(axis=1)) == 0).any():
            raise ValueError("Count matrices must not contain empty libraries.")
        adata.layers["counts"] = adata.X.copy()
        reps["counts"] = "counts"
    adata.uns["mRNA"] = {"representations": reps}
    record(adata, "input", {"input_type": input_type, "orientation": "samples_by_genes"})
    return adata


def read_table(path, metadata=None, *, input_type, orientation="samples_by_genes", sep=None):
    path = Path(path)
    if sep is None:
        sep = "," if ".csv" in path.suffixes else "\t"
    expr = pd.read_csv(path, sep=sep, index_col=0)
    if orientation == "genes_by_samples":
        expr = expr.T
    elif orientation != "samples_by_genes":
        raise ValueError("orientation must be samples_by_genes or genes_by_samples.")
    return from_frames(expr, metadata, input_type=input_type)


def read_featurecounts(path, metadata=None):
    """Read featureCounts gene-level tables, preserving supplied length annotation."""
    table = pd.read_csv(path, sep="\t", comment="#", index_col=0)
    annotation_columns = ["Chr", "Start", "End", "Strand", "Length"]
    if not set(annotation_columns).issubset(table.columns):
        raise ValueError("Expected featureCounts annotation columns.")
    annot = table[annotation_columns].rename(columns={"Length": "length"})
    adata = from_frames(
        table.drop(columns=annotation_columns).T, metadata, annot, input_type="counts"
    )
    record(adata, "featurecounts", {"path": str(path), "length_definition": "featureCounts"})
    return adata


def read_quantifications(paths, metadata=None, *, source="salmon"):
    """Read per-sample quantifier files without rounding fractional estimated counts.

    Pass a mapping of sample ID to Salmon quant.sf, kallisto abundance.tsv, or
    RSEM isoforms.results/genes.results. Feature IDs must match across samples.
    Effective lengths are sample-specific and preserved in a separate layer.
    """
    schemas = {
        "salmon": ("Name", "NumReads", "TPM", "EffectiveLength"),
        "kallisto": ("target_id", "est_counts", "tpm", "eff_length"),
        "rsem": (None, "expected_count", "TPM", "effective_length"),
    }
    if source not in schemas or not paths:
        raise ValueError("Supply sample-to-file paths and source salmon, kallisto, or rsem.")
    matrices = {"estimated_counts": [], "tpm": [], "effective_length": []}
    genes = None
    for sample, path in paths.items():
        table = pd.read_csv(path, sep="\t")
        identifier, counts_col, tpm_col, length_col = schemas[source]
        identifier = identifier or ("transcript_id" if "transcript_id" in table else "gene_id")
        if not {identifier, counts_col, tpm_col, length_col}.issubset(table):
            raise ValueError(f"Missing quantifier columns for sample {sample!r}.")
        table = table.set_index(identifier)
        table.index = table.index.astype(str)
        if not table.index.is_unique or table.index.isna().any():
            raise ValueError("Quantifier feature IDs must be unique and nonmissing.")
        if genes is None:
            genes = table.index.copy()
        elif set(genes) != set(table.index):
            raise ValueError("Quantifier feature IDs must match across all samples.")
        table = table.loc[genes]
        for key, column in zip(matrices, (counts_col, tpm_col, length_col), strict=True):
            values = table[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Quantifier values must be finite and nonnegative.")
            matrices[key].append(values)
    samples = pd.Index(list(paths)).astype(str)
    adata = from_frames(
        pd.DataFrame(matrices["estimated_counts"], index=samples, columns=genes),
        metadata,
        input_type="estimated_counts",
    )
    for key, values in matrices.items():
        adata.layers[key] = np.asarray(values)
    adata.uns["mRNA"]["representations"].update(estimated_counts="estimated_counts", tpm="tpm")
    record(
        adata,
        "quantifier",
        {
            "source": source,
            "feature_level": "quantifier_provided",
            "effective_length_unit": "bp",
            "paths": {str(k): str(v) for k, v in paths.items()},
            "de_support": "estimated counts require a quantification-aware inference workflow",
        },
    )
    return adata


def export_results(adata, outdir, *, keys=None):
    """Export result tables and provenance as portable CSV and JSON files."""
    import json

    require_adata(adata)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries = adata.uns.get("mRNA", {})
    chosen = (
        list(keys)
        if keys is not None
        else [k for k, v in entries.items() if isinstance(v, dict) and "table" in v]
    )
    exported = {}
    provenance = {}
    for i, key in enumerate(chosen):
        entry = entries[key]
        path = outdir / f"result_{i:02d}.csv"
        entry["table"].to_csv(path, index=False)
        exported[key] = path.resolve()
        provenance[key] = {k: v for k, v in entry.items() if k not in {"table", "profiles"}}
    from ._core import jsonable

    (outdir / "provenance.json").write_text(json.dumps(jsonable(provenance), indent=2))
    return exported


def report(adata, path, *, keys=None, title="Bulk transcriptomics report"):
    """Write a static, escaped HTML report with native plots and result downloads."""
    import base64
    import html
    from io import BytesIO

    from . import pl

    require_adata(adata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = export_results(adata, path.parent / f"{path.stem}_tables", keys=keys)
    sections = [
        f"<h1>{html.escape(title)}</h1>",
        f"<p>{adata.n_obs} biological sample profiles; {adata.n_vars} genes.</p>",
    ]
    for key, csv_path in exported.items():
        table = adata.uns["mRNA"][key]["table"]
        sections.append(
            f"<h2>{html.escape(key)}</h2><p><a href='{html.escape(csv_path.relative_to(path.parent.resolve()).as_posix())}'>Download full table</a></p>"
        )
        sections.append(table.head(30).to_html(index=False, escape=True))
        if {"log2_fold_change", "adj_p_value", "status"}.issubset(table):
            ax = pl.volcano(adata, key=key)
            buffer = BytesIO()
            ax.figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            sections.append(
                f"<img alt='Volcano plot' src='data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'>"
            )
            from ._core import dependency

            dependency("matplotlib.pyplot", "plot").close(ax.figure)
    sections.append(
        f"<p><a href='{html.escape((Path(f'{path.stem}_tables') / 'provenance.json').as_posix())}'>Analysis parameters and backend versions</a></p>"
    )
    path.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><body>"
        + "\n".join(sections)
        + "</body></html>"
    )
    return path.resolve()


def write(adata, path):
    """Write the native AnnData format; models are intentionally stored separately."""
    require_adata(adata)
    path = Path(path)
    if path.suffix != ".h5ad":
        raise ValueError("Use the .h5ad extension for AnnData persistence.")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path.resolve()
