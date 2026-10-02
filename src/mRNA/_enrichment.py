"""GSEApy and decoupler adapters with explicit backgrounds and gene-set resources."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._core import dense, dependency, matrix, record, require_adata, working


def read_gene_sets(gene_sets):
    """Load a local GMT or explicit dict; online resource lookup is separate."""
    if isinstance(gene_sets, dict):
        out = {}
        for term, genes in gene_sets.items():
            if isinstance(genes, str):
                raise TypeError("Each gene set must contain an iterable of gene IDs, not a string.")
            out[str(term)] = sorted({str(x) for x in genes})
    else:
        path = Path(gene_sets)
        if not path.is_file():
            raise ValueError(
                "Supply a local GMT file or gene-set dictionary; fetch online resources explicitly."
            )
        out = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3 or parts[0] in out:
                raise ValueError("GMT terms must be unique and have a description and genes.")
            out[parts[0]] = sorted(set(parts[2:]))
    if not out or any(not genes for genes in out.values()):
        raise ValueError("Nonempty gene sets are required.")
    return out


def _resource(sets, resource_version):
    if not isinstance(resource_version, str) or not resource_version.strip():
        raise ValueError("Declare a nonempty gene-set resource version.")
    digest = hashlib.sha256(json.dumps(sets, sort_keys=True).encode()).hexdigest()
    return {"resource_version": resource_version, "resource_sha256": digest, "n_sets": len(sets)}


def _gene_ids(adata, genes, gene_id_col):
    if gene_id_col is None:
        return pd.Series(list(genes), index=genes, dtype=str)
    if gene_id_col not in adata.var:
        raise KeyError(f"Gene annotation {gene_id_col!r} is missing.")
    mapped = adata.var[gene_id_col].reindex(genes)
    if mapped.isna().all():
        raise ValueError("No gene IDs map to the requested identifier column.")
    return mapped.fillna("").astype(str)


def ora(
    adata,
    *,
    gene_sets,
    resource_version,
    de_key="de",
    direction="both",
    alpha=0.05,
    lfc_threshold=0,
    gene_id_col=None,
    key_added="ora",
):
    """Local ORA with the tested-gene universe, optionally separate up/down lists."""
    from .get import result
    from .tl import label_significance

    require_adata(adata)
    if direction not in {"both", "up", "down"}:
        raise ValueError("direction must be both, up, or down.")
    sets = read_gene_sets(gene_sets)
    resource = _resource(sets, resource_version)
    gp = dependency("gseapy", "enrichment")
    de = label_significance(result(adata, de_key), alpha=alpha, lfc_threshold=lfc_threshold)
    eligible = de[de.status.eq("tested")].copy()
    mapped = _gene_ids(adata, eligible.gene, gene_id_col).to_numpy()
    eligible["mapped_gene"] = mapped
    eligible = eligible[eligible.mapped_gene.ne("")]
    background = sorted(set(eligible.mapped_gene))
    if not background:
        raise ValueError("No eligible tested genes map to the enrichment universe.")
    wanted = {"both": ["Up", "Down"], "up": ["Up"], "down": ["Down"]}[direction]
    genes = sorted(set(eligible.loc[eligible.change.isin(wanted), "mapped_gene"]))
    if not genes or not any(set(genes).intersection(values) for values in sets.values()):
        table = pd.DataFrame(
            columns=[
                "Gene_set",
                "Term",
                "Overlap",
                "P-value",
                "Adjusted P-value",
                "Odds Ratio",
                "Combined Score",
                "Genes",
            ]
        )
    else:
        native = gp.enrich(
            gene_list=genes, gene_sets=sets, background=background, outdir=None, no_plot=True
        )
        table = native.results.copy()
    record(
        adata,
        key_added,
        {
            **resource,
            "de_key": de_key,
            "direction": direction,
            "alpha": alpha,
            "lfc_threshold": lfc_threshold,
            "gene_id_col": gene_id_col,
            "background": background,
            "selected_genes": genes,
            "unmapped_tested_genes": len(de[de.status.eq("tested")]) - len(eligible),
        },
        backend="gseapy",
        table=table,
    )
    return table


def gsea(
    adata,
    *,
    gene_sets,
    resource_version,
    de_key="de",
    score="statistic",
    gene_id_col=None,
    duplicate_policy="error",
    seed=0,
    key_added="gsea",
    backend_options=None,
):
    """Preranked GSEA using all finite fitted-gene statistics; retain leading edges."""
    from .get import rank_genes

    require_adata(adata)
    gp = dependency("gseapy", "enrichment")
    sets = read_gene_sets(gene_sets)
    resource = _resource(sets, resource_version)
    rank = rank_genes(adata, key=de_key, score=score)
    input_rank_size = len(rank)
    mapped = _gene_ids(adata, rank.index, gene_id_col)
    rank.index = mapped.to_numpy()
    rank = rank[rank.index != ""]
    if rank.index.duplicated().any():
        if duplicate_policy != "mean":
            raise ValueError(
                "Mapped ranks contain duplicate gene IDs; set duplicate_policy='mean' explicitly."
            )
        rank = rank.groupby(level=0).mean().sort_values(ascending=False, kind="stable")
    if not len(rank) or not np.isfinite(rank).all():
        raise ValueError("A finite, nonempty ranked gene list is required.")
    opts = {"threads": 1, **(backend_options or {})}
    native = gp.prerank(rnk=rank, gene_sets=sets, outdir=None, no_plot=True, seed=seed, **opts)
    table = native.res2d.copy()
    # GSEApy may expose numeric output as object dtype; make HDF5 serialization explicit.
    for col in ("ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"):
        if col in table:
            table[col] = pd.to_numeric(table[col])
    record(
        adata,
        key_added,
        {
            **resource,
            "de_key": de_key,
            "score": score,
            "gene_id_col": gene_id_col,
            "seed": seed,
            "duplicate_policy": duplicate_policy,
            "unmapped_rank_genes": input_rank_size - int(mapped.ne("").sum()),
            "collapsed_rank_genes": int(mapped.ne("").sum()) - len(rank),
            "tied_ranks": int(rank.duplicated().sum()),
            "backend_options": opts,
            "ranked_gene_ids": list(rank.index),
        },
        backend="gseapy",
        table=table,
    )
    profiles = {}
    for i, (term, values) in enumerate(native.results.items()):
        profiles[f"set_{i}"] = {
            "term": str(term),
            "running_score": np.asarray(values["RES"], dtype=float),
            "hits": np.asarray(values["hits"], dtype=int),
        }
    adata.uns["mRNA"][key_added]["profiles"] = profiles
    return table


def score_gene_sets(
    adata,
    *,
    gene_sets,
    resource_version,
    method="ssgsea",
    layer,
    input_type,
    gene_id_col=None,
    seed=0,
    key_added=None,
    copy=False,
    backend_options=None,
):
    """Store sample-by-gene-set scores in obsm, distinct from contrast-level GSEA."""
    adata = working(adata, copy)
    if method not in {"ssgsea", "gsva"}:
        raise ValueError("method must be ssgsea or gsva.")
    if input_type == "counts":
        raise ValueError("Normalize expression before sample-level gene-set scoring.")
    gp = dependency("gseapy", "enrichment")
    sets = read_gene_sets(gene_sets)
    resource = _resource(sets, resource_version)
    expression = pd.DataFrame(
        dense(matrix(adata, layer, input_type=input_type)).T,
        index=adata.var_names,
        columns=adata.obs_names,
    )
    expression.index = _gene_ids(adata, expression.index, gene_id_col).to_numpy()
    expression = expression[expression.index != ""]
    if not expression.index.is_unique:
        raise ValueError("Sample scoring requires unique mapped gene IDs.")
    opts = {"threads": 1, **(backend_options or {})}
    if method == "ssgsea":
        native = gp.ssgsea(expression, sets, outdir=None, no_plot=True, seed=seed, **opts)
        value_col = "NES"
    else:
        native = gp.gsva(expression, sets, outdir=None, seed=seed, **opts)
        value_col = "ES"
    scores = (
        native.res2d.pivot(index="Name", columns="Term", values=value_col)
        .reindex(adata.obs_names)
        .astype(float)
    )
    key_added = key_added or f"score_{method}"
    adata.obsm[key_added] = scores
    record(
        adata,
        key_added,
        {
            **resource,
            "layer": layer,
            "input_type": input_type,
            "gene_id_col": gene_id_col,
            "method": method,
            "seed": seed,
            "score_type": value_col,
            "backend_options": opts,
        },
        backend="gseapy",
    )
    return adata


def activity(
    adata,
    network,
    *,
    method="ulm",
    layer,
    input_type,
    resource_version,
    key_added=None,
    copy=False,
    backend_options=None,
):
    """Infer sample-level TF/pathway activity using an explicit signed network."""
    from anndata import AnnData

    adata = working(adata, copy)
    if method not in {"ulm", "mlm", "viper", "aucell"}:
        raise ValueError("Supported activity methods: ulm, mlm, viper, aucell.")
    if input_type == "counts":
        raise ValueError("Normalize expression before activity scoring.")
    dc = dependency("decoupler", "activity")
    if not {"source", "target"}.issubset(network.columns):
        raise ValueError("Network columns source and target are required.")
    # A working object avoids backend filtering/mutation of the source sample/gene axes.
    temporary = AnnData(
        X=matrix(adata, layer, input_type=input_type).copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    getattr(dc.mt, method)(temporary, network.copy(), raw=False, **(backend_options or {}))
    key_added = key_added or f"activity_{method}"
    adata.obsm[key_added] = temporary.obsm[f"score_{method}"].copy()
    pvalue_key = f"padj_{method}"
    if pvalue_key in temporary.obsm:
        adata.obsm[f"{key_added}_padj"] = temporary.obsm[pvalue_key].copy()
    digest = hashlib.sha256(network.to_csv(index=False).encode()).hexdigest()
    record(
        adata,
        key_added,
        {
            "layer": layer,
            "input_type": input_type,
            "method": method,
            "resource_version": resource_version,
            "network_sha256": digest,
            "backend_options": backend_options or {},
        },
        backend="decoupler",
    )
    return adata
