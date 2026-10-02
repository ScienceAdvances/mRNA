"""Sample-level bulk inference, exploration, and functional interpretation."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._core import dense, dependency, matrix, record, require_adata, working
from ._design import build_design, contrast_vector
from ._enrichment import activity as activity
from ._enrichment import gsea as gsea
from ._enrichment import ora as ora
from ._enrichment import score_gene_sets as score_gene_sets


@dataclass
class FittedModel:
    """Native fitted model outside AnnData persistence, reusable across contrasts."""

    method: str
    source: object
    metadata: pd.DataFrame
    design: pd.DataFrame
    spec: object
    gene_names: pd.Index
    payload: object
    parameters: dict
    native_results: dict = field(default_factory=dict)


def capabilities():
    """Implemented capabilities, rather than assumed equivalence to all R features."""
    return pd.DataFrame(
        [
            {
                "method": "deseq2",
                "input": "counts",
                "tests": "Wald",
                "normalization": "DESeq2 size factors",
                "shrinkage": "explicit coefficient apeGLM",
            },
            {
                "method": "edgepython",
                "input": "counts",
                "tests": "QL, LRT, TREAT",
                "normalization": "TMM",
                "shrinkage": "backend prior_count",
            },
            {
                "method": "limma",
                "input": "counts or processed log_expression",
                "tests": "moderated t",
                "normalization": "TMM + voom for counts",
                "shrinkage": "empirical Bayes variance moderation",
            },
        ]
    )


def _tmm(counts, design, *, filter_expression, min_total_count, options):
    ep = dependency("edgepython", "edge")
    y = ep.make_dgelist(counts.T)
    keep = (
        np.asarray(ep.filter_by_expr(y, design=design, min_total_count=min_total_count))
        if filter_expression
        else counts.sum(axis=0) >= min_total_count
    )
    keep &= counts.sum(axis=0) > 0
    if not keep.any():
        raise ValueError("No genes pass count expression filtering.")
    # Preserve the explicit original library denominator after gene filtering.
    y = ep.make_dgelist(counts[:, keep].T, lib_size=counts.sum(axis=1))
    y = ep.calc_norm_factors(y, **options)
    return ep, y, keep


def fit(
    adata,
    *,
    design,
    method="deseq2",
    input_type="counts",
    layer="counts",
    min_total_count=10,
    filter_expression=True,
    n_cpus=1,
    normalization_options=None,
    backend_options=None,
    voom_options=None,
):
    """Fit sample-level models with an explicit representation and full-rank design.

    Use layer=None and input_type='log_expression' for processed microarrays.
    Counts-based limma uses edgePython TMM followed by pylimma voom. Biological
    replicates, pairing, and confounders remain properties of the study design.
    """
    require_adata(adata)
    if method not in {"deseq2", "edgepython", "limma"}:
        raise ValueError("method must be deseq2, edgepython, or limma.")
    if input_type not in ({"counts", "log_expression"} if method == "limma" else {"counts"}):
        raise ValueError(
            f"{method} does not accept {input_type}; use original counts or processed microarray log-expression."
        )
    if min_total_count < 0 or n_cpus < 1:
        raise ValueError("Use nonnegative filtering thresholds and positive n_cpus.")
    m = dense(matrix(adata, layer, input_type=input_type)).copy()
    if input_type == "counts" and (m.sum(axis=1) == 0).any():
        raise ValueError("Count inference requires nonempty libraries.")
    if (
        adata.uns.get("mRNA", {}).get(layer or "X", {}).get("parameters", {}).get("purpose")
        == "exploration_only"
    ):
        raise ValueError("Batch-adjusted exploratory expression must not be used for DE.")
    design_matrix, spec = build_design(adata.obs, design)
    opts, norm_opts = dict(backend_options or {}), dict(normalization_options or {})
    params = {
        "input_type": input_type,
        "source_layer": layer,
        "design": design if isinstance(design, str) else "explicit_matrix",
        "design_columns": list(design_matrix.columns),
        "sample_ids": list(adata.obs_names),
        "original_gene_ids": list(adata.var_names),
        "min_total_count": min_total_count,
        "filter_expression": filter_expression,
        "backend_options": opts,
        "normalization_options": norm_opts,
        "n_cpus": n_cpus,
    }
    if method == "deseq2":
        if norm_opts or voom_options:
            raise ValueError("Use PyDESeq2 backend_options for its normalization settings.")
        dds_module = dependency("pydeseq2.dds", "deseq2")
        keep = (m.sum(axis=0) >= min_total_count) & (m.sum(axis=0) > 0)
        if not keep.any():
            raise ValueError("No genes pass count expression filtering.")
        frame = pd.DataFrame(m[:, keep], index=adata.obs_names, columns=adata.var_names[keep])
        dds = dds_module.DeseqDataSet(
            counts=frame,
            metadata=adata.obs.copy(),
            design=design_matrix,
            n_cpus=n_cpus,
            **{"quiet": True, **opts},
        )
        dds.deseq2()
        payload = dds
        params["normalization"] = "DESeq2_size_factors"
    elif method == "edgepython":
        if voom_options:
            raise ValueError("voom_options are only used by counts-based limma.")
        ep, y, keep = _tmm(
            m,
            design_matrix.to_numpy(),
            filter_expression=filter_expression,
            min_total_count=min_total_count,
            options=norm_opts,
        )
        test = opts.pop("test", "ql")
        if test not in {"ql", "lrt"}:
            raise ValueError(
                "edgePython fit test must be ql or lrt; TREAT is requested at contrast time."
            )
        dispersion_options = opts.pop("dispersion_options", {})
        y = ep.estimate_disp(y, design_matrix.to_numpy(), **dispersion_options)
        native_fit = (ep.glm_ql_fit if test == "ql" else ep.glm_fit)(
            y, design_matrix.to_numpy(), **opts
        )
        payload = {"fit": native_fit, "dge": y, "test": test}
        params.update(
            normalization=norm_opts.get("method", "TMM"),
            library_sizes=y["samples"]["lib.size"].to_numpy(),
            norm_factors=y["samples"]["norm.factors"].to_numpy(),
            test=test,
        )
    else:
        # Avoid accepting another project that uses the same import namespace.
        from importlib.metadata import PackageNotFoundError, version

        try:
            version("python-limma")
        except PackageNotFoundError:
            pass
        else:
            raise ImportError("Remove python-limma; mRNA uses the distinct pylimma distribution.")
        limma = dependency("pylimma", "limma")
        if input_type == "counts":
            _, y, keep = _tmm(
                m,
                design_matrix.to_numpy(),
                filter_expression=filter_expression,
                min_total_count=min_total_count,
                options=norm_opts,
            )
            effective = (y["samples"]["lib.size"] * y["samples"]["norm.factors"]).to_numpy()
            voom_opts = dict(voom_options or {})
            expr = limma.voom(
                y["counts"], design=design_matrix.to_numpy(), lib_size=effective, **voom_opts
            )
            payload = limma.lm_fit(
                expr["E"], design=design_matrix.to_numpy(), weights=expr["weights"], **opts
            )
            params.update(
                normalization=f"{norm_opts.get('method', 'TMM')}_voom",
                effective_library_sizes=effective,
                norm_factors=y["samples"]["norm.factors"].to_numpy(),
                voom_options=voom_opts,
            )
        else:
            if norm_opts or voom_options:
                raise ValueError(
                    "Processed microarray expression has no count-normalization or voom options."
                )
            keep = np.ones(adata.n_vars, dtype=bool)
            payload = limma.lm_fit(m.T, design=design_matrix.to_numpy(), **opts)
            params["normalization"] = "processed_log_expression"
    gene_names = adata.var_names[keep].copy()
    params["fitted_genes"] = list(gene_names)
    model = FittedModel(
        method, adata, adata.obs.copy(), design_matrix, spec, gene_names, payload, params
    )
    record(
        adata,
        f"fit_{method}",
        params,
        backend={"deseq2": "pydeseq2", "edgepython": "edgepython", "limma": "pylimma"}[method],
    )
    return model


def _bh(values):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if ((values[valid] < 0) | (values[valid] > 1)).any():
        raise ValueError("Backend returned invalid p-values.")
    if valid.any():
        out[valid] = stats.false_discovery_control(values[valid], method="bh")
    return out


def contrast(
    model,
    *,
    contrast,
    key_added="de",
    alpha=0.05,
    lfc_threshold=0,
    min_replicates=2,
    backend_options=None,
    shrink_coefficient=None,
):
    """Test a named comparison or numeric contrast, preserving backend diagnostics.

    Named comparisons in interaction models average the counterfactual design
    over the observed covariates. Supply a numeric vector for a specific simple
    effect. Edge QL/LRT ranking statistics are signed sqrt(F/LR), with the native
    F/LR columns retained. An lfc_threshold labels results only; edgePython's
    TREAT hypothesis is explicitly requested with backend_options={'lfc': ...}.
    """
    if not isinstance(model, FittedModel):
        raise TypeError("model must be returned by mRNA.tl.fit.")
    if (
        list(model.source.obs_names) != model.parameters["sample_ids"]
        or list(model.source.var_names) != model.parameters["original_gene_ids"]
    ):
        raise ValueError("Source AnnData axes changed after fitting; fit a new model.")
    if not 0 < alpha < 1 or lfc_threshold < 0 or min_replicates < 2:
        raise ValueError("Use 0 < alpha < 1, lfc_threshold >= 0, and min_replicates >= 2.")
    vector, label = contrast_vector(model, contrast, min_replicates=min_replicates)
    opts = dict(backend_options or {})
    if model.method == "deseq2":
        ds = dependency("pydeseq2.ds", "deseq2")
        native = ds.DeseqStats(
            model.payload,
            contrast=vector,
            alpha=alpha,
            **{"quiet": True, "n_cpus": model.parameters["n_cpus"], **opts},
        )
        native.summary()
        table = native.results_df.copy()
        table["gene"] = table.index.astype(str)
        for canonical, original in {
            "log2_fold_change": "log2FoldChange",
            "statistic": "stat",
            "p_value": "pvalue",
            "adj_p_value": "padj",
            "mean_expression": "baseMean",
            "lfc_se": "lfcSE",
        }.items():
            table[canonical] = table[original]
        if shrink_coefficient is not None:
            if shrink_coefficient not in model.design.columns:
                raise ValueError("shrink_coefficient must name a fitted design column.")
            expected = np.zeros(len(vector))
            expected[model.design.columns.get_loc(shrink_coefficient)] = 1
            if not np.allclose(vector, expected):
                raise ValueError(
                    "LFC shrinkage currently requires a positive single-coefficient contrast."
                )
            native.lfc_shrink(coeff=shrink_coefficient)
            table["log2_fold_change_shrunken"] = native.results_df["log2FoldChange"]
            table["lfc_se_shrunken"] = native.results_df["lfcSE"]
        statistic_type, mean_type = "Wald_z", "DESeq2_normalized_count_mean"
    elif model.method == "edgepython":
        if shrink_coefficient is not None:
            raise ValueError("shrink_coefficient is only available for PyDESeq2.")
        ep = dependency("edgepython", "edge")
        if "lfc" in opts:
            native = ep.glm_treat(model.payload["fit"], contrast=vector, **opts)
            statistic_type = "not_reported_TREAT"
        else:
            test = model.payload["test"]
            native = (ep.glm_ql_ftest if test == "ql" else ep.glm_lrt)(
                model.payload["fit"], contrast=vector, **opts
            )
            statistic_type = "signed_sqrt_QL_F" if test == "ql" else "signed_sqrt_LR"
        table = native["table"].copy().reset_index(drop=True)
        table["gene"] = model.gene_names.to_numpy()
        table["log2_fold_change"] = table["logFC"]
        if "lfc" in opts:
            # TREAT reports a p-value, not a native F/LR statistic.
            table["statistic"] = np.nan
        else:
            stat_col = "F" if "F" in table else "LR"
            table["statistic"] = np.sign(table["logFC"]) * np.sqrt(table[stat_col].clip(lower=0))
        table["p_value"] = table["PValue"]
        table["adj_p_value"] = _bh(table["PValue"])
        table["mean_expression"] = table["logCPM"]
        mean_type = "edgeR_logCPM"
    else:
        if shrink_coefficient is not None:
            raise ValueError("shrink_coefficient is only available for PyDESeq2.")
        limma = dependency("pylimma", "limma")
        fitted = limma.contrasts_fit(model.payload, contrasts=vector[:, None])
        native = limma.e_bayes(fitted, **opts)
        table = limma.top_table(
            native,
            coef=0,
            number=len(model.gene_names),
            sort_by="none",
            genelist=pd.DataFrame({"gene": model.gene_names}),
        ).copy()
        table["log2_fold_change"] = table["log_fc"]
        table["statistic"] = table["t"]
        table["mean_expression"] = table["ave_expr"]
        statistic_type, mean_type = "moderated_t", "limma_average_log_expression"
    table = (
        table.set_index("gene").reindex(model.source.var_names).rename_axis("gene").reset_index()
    )
    table["method"], table["contrast"] = model.method, label
    table["statistic_type"], table["mean_expression_type"] = statistic_type, mean_type
    table["status"] = np.where(table.gene.isin(model.gene_names), "tested", "prefiltered")
    unavailable = table.status.eq("tested") & ~np.isfinite(table.p_value)
    table.loc[unavailable, "status"] = "untested"
    table["reason"] = np.where(
        table.status.eq("prefiltered"),
        "expression_filter",
        np.where(
            unavailable,
            "backend_missing_pvalue",
            np.where(~np.isfinite(table.adj_p_value), "backend_missing_padj", ""),
        ),
    )
    table = label_significance(table, alpha=alpha, lfc_threshold=lfc_threshold)
    params = {
        **model.parameters,
        "contrast": contrast,
        "contrast_vector": vector,
        "alpha": alpha,
        "lfc_threshold": lfc_threshold,
        "backend_options_test": opts,
        "shrink_coefficient": shrink_coefficient,
        "statistic_type": statistic_type,
        "mean_expression_type": mean_type,
    }
    backend = {"deseq2": "pydeseq2", "edgepython": "edgepython", "limma": "pylimma"}[model.method]
    model.native_results[key_added] = native
    record(model.source, key_added, params, backend=backend, table=table)
    return table


def differential_expression(
    adata,
    *,
    design,
    contrast,
    method="deseq2",
    input_type="counts",
    layer="counts",
    key_added="de",
    fit_options=None,
    **test_options,
):
    model = fit(
        adata,
        design=design,
        method=method,
        input_type=input_type,
        layer=layer,
        **(fit_options or {}),
    )
    contrast_result = globals()["contrast"](
        model, contrast=contrast, key_added=key_added, **test_options
    )
    return contrast_result, model


def label_significance(table, *, alpha=0.05, lfc_threshold=0):
    """Recompute labels from adjusted p-values, never preserving stale labels."""
    if not 0 < alpha < 1 or lfc_threshold < 0:
        raise ValueError("Invalid significance thresholds.")
    out = table.copy()
    out["change"] = "NS"
    sig = out.adj_p_value.lt(alpha)
    out.loc[sig & out.log2_fold_change.gt(lfc_threshold), "change"] = "Up"
    out.loc[sig & out.log2_fold_change.lt(-lfc_threshold), "change"] = "Down"
    return out


def pca(adata, *, layer, input_type, n_components=10, key_added="X_pca", copy=False):
    adata = working(adata, copy)
    m = dense(matrix(adata, layer, input_type=input_type)).astype(float)
    if input_type not in {"log_expression", "logcpm", "logtpm", "vst"}:
        raise ValueError("PCA expects explicitly transformed expression.")
    if not 1 <= n_components <= min(adata.n_obs - 1, adata.n_vars):
        raise ValueError("n_components exceeds the matrix dimensions.")
    m = m - m.mean(axis=0)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    if not np.any(s > 0):
        raise ValueError("PCA requires expression variance.")
    adata.obsm[key_added] = u[:, :n_components] * s[:n_components]
    adata.varm[f"{key_added}_loadings"] = vt[:n_components].T
    variance = s**2 / (adata.n_obs - 1)
    record(
        adata,
        key_added,
        {
            "layer": layer,
            "input_type": input_type,
            "n_components": n_components,
            "variance_ratio": (variance / variance.sum())[:n_components],
        },
    )
    return adata


def sample_correlation(
    adata, *, layer, input_type, method="pearson", key_added="sample_correlation"
):
    from .get import expression

    matrix(adata, layer, input_type=input_type)
    if method not in {"pearson", "spearman"}:
        raise ValueError("Use pearson or spearman correlation.")
    table = expression(adata, layer=layer).T.corr(method=method)
    adata.obsp[key_added] = table.to_numpy()
    record(adata, key_added, {"layer": layer, "input_type": input_type, "method": method})
    return table


def mds(adata, *, layer, input_type, n_components=2, key_added="X_mds", copy=False):
    """Classical metric MDS of Euclidean distances on transformed expression."""
    adata = working(adata, copy)
    if input_type not in {"log_expression", "logcpm", "logtpm", "vst"}:
        raise ValueError("MDS expects transformed expression.")
    m = dense(matrix(adata, layer, input_type=input_type))
    if not 1 <= n_components < adata.n_obs:
        raise ValueError("n_components must be between 1 and n_samples-1.")
    from scipy.spatial.distance import pdist, squareform

    d2 = squareform(pdist(m)) ** 2
    h = np.eye(adata.n_obs) - np.ones((adata.n_obs, adata.n_obs)) / adata.n_obs
    eigenvalues, vectors = np.linalg.eigh(-0.5 * h @ d2 @ h)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    adata.obsm[key_added] = vectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0))
    record(
        adata, key_added, {"layer": layer, "input_type": input_type, "n_components": n_components}
    )
    return adata


def correlate_phenotype(
    adata, *, phenotype, layer, input_type, method="spearman", key_added="phenotype_correlation"
):
    """Exploratory marginal gene/phenotype correlations with BH across genes."""
    if phenotype not in adata.obs or adata.obs[phenotype].isna().any():
        raise ValueError("A complete numeric phenotype is required.")
    y = adata.obs[phenotype].to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.ptp(y) == 0 or adata.n_obs < 3:
        raise ValueError("Phenotype needs finite variation and at least three samples.")
    m = dense(matrix(adata, layer, input_type=input_type))
    if method not in {"pearson", "spearman"}:
        raise ValueError("Use pearson or spearman correlation.")
    function = stats.pearsonr if method == "pearson" else stats.spearmanr
    rows = [
        function(m[:, i], y) if np.ptp(m[:, i]) else (np.nan, np.nan) for i in range(adata.n_vars)
    ]
    table = pd.DataFrame(rows, columns=["correlation", "p_value"])
    table["gene"] = adata.var_names.to_numpy()
    table["adj_p_value"] = _bh(table.p_value)
    record(
        adata,
        key_added,
        {
            "layer": layer,
            "input_type": input_type,
            "phenotype": phenotype,
            "method": method,
            "design": "marginal_unadjusted",
        },
        table=table,
    )
    return table


def compare_results(adata, *, keys, alpha=0.05, lfc_threshold=0, key_added="comparison"):
    """Compare effects and labels across contrasts/methods without pooling p-values."""
    from .get import result

    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Provide distinct result keys.")
    tables = []
    for key in keys:
        table = label_significance(result(adata, key), alpha=alpha, lfc_threshold=lfc_threshold)
        table["result_key"] = key
        tables.append(
            table[["gene", "result_key", "log2_fold_change", "adj_p_value", "change", "status"]]
        )
    out = pd.concat(tables, ignore_index=True)
    record(
        adata,
        key_added,
        {
            "keys": keys,
            "alpha": alpha,
            "lfc_threshold": lfc_threshold,
            "purpose": "descriptive_comparison",
        },
        table=out,
    )
    return out
