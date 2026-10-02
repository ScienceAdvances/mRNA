"""Bulk matrix QC, unit conversion, and exploratory preprocessing."""

import numpy as np
import pandas as pd
from scipy import sparse

from ._core import dense, matrix, record, store_layer, totals, working


def qc(adata, *, layer="counts", input_type="counts", copy=False):
    adata = working(adata, copy)
    m = matrix(adata, layer, input_type=input_type)
    if input_type in {"counts", "estimated_counts"}:
        adata.obs["total_counts"] = totals(m, 1)
        adata.obs["n_genes_detected"] = totals(m > 0, 1)
        adata.var["total_counts"] = totals(m, 0)
        adata.var["n_samples_detected"] = totals(m > 0, 0)
    else:
        values = dense(m)
        adata.obs["expression_median"] = np.median(values, axis=1)
        adata.obs["expression_std"] = values.std(axis=1)
        adata.var["mean_expression"] = values.mean(axis=0)
    record(adata, "qc", {"layer": layer, "input_type": input_type})
    return adata


def mad_outliers(
    adata,
    *,
    metrics=("total_counts", "n_genes_detected"),
    nmads=3,
    key_added="qc_outlier",
    copy=False,
):
    """Flag exploratory sample QC outliers without automatically dropping samples."""
    adata = working(adata, copy)
    if not metrics or not np.isfinite(nmads) or nmads <= 0:
        raise ValueError("Use at least one metric and a finite positive MAD threshold.")
    flags = []
    for column in metrics:
        values = adata.obs[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Outlier metrics must be finite.")
        center = np.median(values)
        mad = np.median(abs(values - center))
        # Degenerate metrics provide no robust scale estimate; do not invent one.
        flag = abs(values - center) > nmads * mad if mad > 0 else np.zeros(len(values), dtype=bool)
        adata.obs[f"{key_added}_{column}"] = flag
        flags.append(flag)
    adata.obs[key_added] = np.any(flags, axis=0)
    record(adata, key_added, {"metrics": metrics, "nmads": nmads, "action": "flag_only"})
    return adata


def _libraries(m, library_sizes, index):
    if library_sizes is None:
        sizes = totals(m, 1)
    elif isinstance(library_sizes, pd.Series):
        if not library_sizes.index.is_unique or set(library_sizes.index) != set(index):
            raise ValueError("Library size IDs must match samples exactly.")
        sizes = library_sizes.loc[index].to_numpy(dtype=float)
    else:
        sizes = np.asarray(library_sizes, dtype=float)
    if sizes.shape != (len(index),) or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("Finite, positive library sizes are required for every sample.")
    return sizes


def cpm(adata, *, layer="counts", library_sizes=None, key_added="cpm", set_X=False, copy=False):
    """Compute CPM using all input genes or explicitly supplied original library sizes."""
    adata = working(adata, copy)
    m = matrix(adata, layer, input_type="counts")
    sizes = _libraries(m, library_sizes, adata.obs_names)
    values = sparse.diags(1e6 / sizes) @ m if sparse.issparse(m) else m * (1e6 / sizes)[:, None]
    store_layer(
        adata,
        key_added,
        values,
        "cpm",
        {"source_layer": layer, "library_sizes": sizes, "unit": "CPM"},
        set_X=set_X,
    )
    return adata


def _lengths(adata, lengths):
    if isinstance(lengths, str):
        if lengths not in adata.var:
            raise KeyError(f"Length column {lengths!r} is missing from var.")
        values = adata.var[lengths].to_numpy(dtype=float)
    elif isinstance(lengths, pd.Series):
        if not lengths.index.is_unique or set(lengths.index) != set(adata.var_names):
            raise ValueError("Length IDs must match all genes exactly.")
        values = lengths.loc[adata.var_names].to_numpy(dtype=float)
    else:
        values = np.asarray(lengths, dtype=float)
    if values.shape != (adata.n_vars,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Finite, positive lengths in base pairs are required for every gene.")
    return values


def tpm(
    adata,
    *,
    lengths="length",
    length_definition,
    layer="counts",
    key_added="tpm",
    set_X=False,
    copy=False,
):
    """Compute gene-level TPM; the length definition must be declared explicitly."""
    adata = working(adata, copy)
    if not length_definition:
        raise ValueError("Declare the source and definition of gene lengths.")
    m = matrix(adata, layer, input_type="counts")
    lens = _lengths(adata, lengths)
    rate = m @ sparse.diags(1 / lens) if sparse.issparse(m) else m / lens
    sizes = _libraries(rate, None, adata.obs_names)
    values = (
        sparse.diags(1e6 / sizes) @ rate if sparse.issparse(rate) else rate * (1e6 / sizes)[:, None]
    )
    store_layer(
        adata,
        key_added,
        values,
        "tpm",
        {
            "source_layer": layer,
            "length_definition": length_definition,
            "lengths_bp": lens,
            "unit": "TPM",
        },
        set_X=set_X,
    )
    return adata


def fpkm(
    adata,
    *,
    lengths="length",
    length_definition,
    layer="counts",
    library_sizes=None,
    count_unit="fragments",
    key_added=None,
    set_X=False,
    copy=False,
):
    """Compute FPKM for fragment counts or RPKM for read counts."""
    if count_unit not in {"reads", "fragments"}:
        raise ValueError("count_unit must be reads or fragments.")
    if not length_definition:
        raise ValueError("Declare the source and definition of gene lengths.")
    adata = working(adata, copy)
    m = matrix(adata, layer, input_type="counts")
    lens = _lengths(adata, lengths)
    sizes = _libraries(m, library_sizes, adata.obs_names)
    if sparse.issparse(m):
        values = sparse.diags(1e9 / sizes) @ m @ sparse.diags(1 / lens)
    else:
        values = m * (1e9 / sizes)[:, None] / lens
    representation = "fpkm" if count_unit == "fragments" else "rpkm"
    key_added = representation if key_added is None else key_added
    store_layer(
        adata,
        key_added,
        values,
        representation,
        {
            "source_layer": layer,
            "length_definition": length_definition,
            "library_sizes": sizes,
            "lengths_bp": lens,
            "count_unit": count_unit,
        },
        set_X=set_X,
    )
    return adata


def rpkm(
    adata,
    *,
    length_definition,
    lengths="length",
    layer="counts",
    library_sizes=None,
    key_added="rpkm",
    set_X=False,
    copy=False,
):
    """Read-count equivalent of FPKM, explicitly stored as RPKM."""
    return fpkm(
        adata,
        length_definition=length_definition,
        lengths=lengths,
        layer=layer,
        library_sizes=library_sizes,
        count_unit="reads",
        key_added=key_added,
        set_X=set_X,
        copy=copy,
    )


def log_transform(
    adata, *, layer, input_type, key_added, base=2, pseudocount=1, set_X=False, copy=False
):
    """Store a declared log transform without altering original counts."""
    adata = working(adata, copy)
    if input_type not in {
        "counts",
        "estimated_counts",
        "normalized_counts",
        "cpm",
        "tpm",
        "fpkm",
        "rpkm",
    }:
        raise ValueError("Log transforms require a linear-scale input representation.")
    m = matrix(adata, layer, input_type=input_type)
    if (
        not np.isfinite(base)
        or base <= 0
        or base == 1
        or not np.isfinite(pseudocount)
        or pseudocount <= 0
    ):
        raise ValueError("Use a positive base other than 1 and a finite positive pseudocount.")
    if (dense(m) < 0).any():
        raise ValueError("Log transforms require nonnegative input.")
    values = np.log(dense(m) + pseudocount) / np.log(base)
    representation = {"cpm": "logcpm", "tpm": "logtpm"}.get(input_type, "log_expression")
    store_layer(
        adata,
        key_added,
        values,
        representation,
        {"source_layer": layer, "base": base, "pseudocount": pseudocount},
        set_X=set_X,
    )
    return adata


def filter_genes(adata, *, layer="counts", min_count=10, min_samples=2, min_total_count=10):
    """Return a filtered copy; record the full original feature universe."""
    m = matrix(adata, layer, input_type="counts")
    if min_count < 0 or min_total_count < 0 or not 1 <= min_samples <= adata.n_obs:
        raise ValueError("Invalid expression filter thresholds.")
    detected = np.full(adata.n_vars, adata.n_obs) if min_count == 0 else totals(m >= min_count, 0)
    keep = (detected >= min_samples) & (totals(m, 0) >= min_total_count)
    if not keep.any():
        raise ValueError("No genes pass expression filtering.")
    out = adata[:, keep].copy()
    record(
        out,
        "filter_genes",
        {
            "layer": layer,
            "min_count": min_count,
            "min_samples": min_samples,
            "min_total_count": min_total_count,
            "original_gene_ids": list(adata.var_names),
            "retained_gene_ids": list(out.var_names),
        },
    )
    return out


def normalize(
    adata,
    *,
    method="tmm",
    layer="counts",
    key_added="normalized_counts",
    backend_options=None,
    copy=False,
):
    """Store relative normalized counts; DE still receives the original counts."""
    from ._core import dependency

    adata = working(adata, copy)
    counts = dense(matrix(adata, layer, input_type="counts"))
    if (counts.sum(axis=1) <= 0).any():
        raise ValueError("Normalization requires nonempty libraries.")
    if method == "tmm":
        ep = dependency("edgepython", "edge")
        dge = ep.calc_norm_factors(ep.make_dgelist(counts.T), **(backend_options or {}))
        effective = (dge["samples"]["lib.size"] * dge["samples"]["norm.factors"]).to_numpy()
        factors = effective / np.exp(np.log(effective).mean())
        backend = "edgepython"
    elif method == "deseq2":
        dds_mod = dependency("pydeseq2.dds", "deseq2")
        frame = pd.DataFrame(counts, index=adata.obs_names, columns=adata.var_names)
        model = dds_mod.DeseqDataSet(
            counts=frame,
            metadata=adata.obs.copy(),
            design="~ 1",
            **{"quiet": True, "n_cpus": 1, **(backend_options or {})},
        )
        model.fit_size_factors()
        factors = model.obs["size_factors"].to_numpy()
        backend = "pydeseq2"
    else:
        raise ValueError("method must be tmm or deseq2.")
    store_layer(
        adata,
        key_added,
        counts / factors[:, None],
        "normalized_counts",
        {"source_layer": layer, "method": method, "size_factors": factors},
    )
    record(
        adata,
        key_added,
        {
            "source_layer": layer,
            "method": method,
            "size_factors": factors,
            "backend_options": backend_options or {},
        },
        backend=backend,
    )
    return adata


def remove_batch_effect(
    adata,
    *,
    layer,
    input_type,
    batch_col,
    protect_design="~ 1",
    key_added="batch_corrected",
    copy=False,
):
    """Remove fitted batch terms for exploration, retaining the protected design."""
    from ._design import build_design

    adata = working(adata, copy)
    if input_type not in {"log_expression", "logcpm", "logtpm", "vst"}:
        raise ValueError("Exploratory batch adjustment requires transformed expression.")
    m = dense(matrix(adata, layer, input_type=input_type))
    protected, _ = build_design(adata.obs, protect_design)
    if batch_col not in adata.obs or adata.obs[batch_col].isna().any():
        raise ValueError("Nonmissing batch annotations are required.")
    batches = pd.get_dummies(adata.obs[batch_col], drop_first=True, dtype=float)
    if not batches.shape[1]:
        raise ValueError("At least two batches are required.")
    design = np.column_stack([protected, batches])
    if np.linalg.matrix_rank(design) < design.shape[1] or adata.n_obs <= design.shape[1]:
        raise ValueError(
            "Batch and protected design are confounded or lack residual degrees of freedom."
        )
    coef = np.linalg.lstsq(design, m, rcond=None)[0]
    values = m - batches.to_numpy() @ coef[protected.shape[1] :]
    store_layer(
        adata,
        key_added,
        values,
        "log_expression",
        {
            "source_layer": layer,
            "batch_col": batch_col,
            "protect_design": protect_design,
            "purpose": "exploration_only",
        },
    )
    return adata
