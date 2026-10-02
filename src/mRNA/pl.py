"""Matplotlib plots that return native axes/figures without saving implicitly."""

import numpy as np
import pandas as pd

from ._core import dependency, require_adata
from .get import expression as expression_frame
from .get import result


def _axes(ax=None, figsize=(5, 4)):
    plt = dependency("matplotlib.pyplot", "plot")
    return ax if ax is not None else plt.subplots(figsize=figsize)[1]


def volcano(adata, *, key="de", alpha=0.05, lfc_threshold=0, ax=None):
    from .tl import label_significance

    table = label_significance(result(adata, key), alpha=alpha, lfc_threshold=lfc_threshold)
    ax = _axes(ax)
    for label, color in {"NS": "#999999", "Up": "#D55E00", "Down": "#0072B2"}.items():
        rows = table[table.change.eq(label)]
        ax.scatter(
            rows.log2_fold_change,
            -np.log10(rows.adj_p_value.clip(lower=np.finfo(float).tiny)),
            s=12,
            color=color,
            alpha=0.65,
            label=label,
            rasterized=True,
        )
    ax.axhline(-np.log10(alpha), ls="--", color="#555555", lw=0.8)
    if lfc_threshold:
        for x in (-lfc_threshold, lfc_threshold):
            ax.axvline(x, ls="--", color="#555555", lw=0.8)
    ax.set(xlabel="Log2 fold change", ylabel="−log10 adjusted p-value")
    ax.legend(frameon=False)
    return ax


def ma(adata, *, key="de", alpha=0.05, ax=None):
    table = result(adata, key)
    ax = _axes(ax)
    colors = np.where(table.adj_p_value.lt(alpha), "#D55E00", "#999999")
    ax.scatter(
        table.mean_expression, table.log2_fold_change, c=colors, s=12, alpha=0.6, rasterized=True
    )
    ax.axhline(0, color="#555555", lw=0.8)
    kind = table.mean_expression_type.iloc[0]
    ax.set(xlabel=kind.replace("_", " "), ylabel="Log2 fold change")
    return ax


def embedding(adata, *, basis="X_pca", color=None, components=(0, 1), ax=None):
    require_adata(adata)
    coords = np.asarray(adata.obsm[basis])
    if min(components) < 0 or max(components) >= coords.shape[1]:
        raise ValueError("Requested embedding components are unavailable.")
    ax = _axes(ax)
    if color is None:
        ax.scatter(coords[:, components[0]], coords[:, components[1]], s=35)
    elif pd.api.types.is_numeric_dtype(adata.obs[color]):
        points = ax.scatter(
            coords[:, components[0]], coords[:, components[1]], c=adata.obs[color], s=35
        )
        ax.figure.colorbar(points, ax=ax, label=color)
    else:
        for level in pd.unique(adata.obs[color]):
            mask = adata.obs[color].eq(level).to_numpy()
            ax.scatter(
                coords[mask, components[0]], coords[mask, components[1]], label=str(level), s=35
            )
        ax.legend(title=color, frameon=False)
    ax.set(
        xlabel=f"{basis.removeprefix('X_').upper()} {components[0] + 1}",
        ylabel=f"{basis.removeprefix('X_').upper()} {components[1] + 1}",
    )
    return ax


def heatmap(adata, *, genes, layer, zscore=False, cluster_samples=False, ax=None):
    table = expression_frame(adata, layer=layer).loc[:, list(genes)]
    if cluster_samples:
        from scipy.cluster.hierarchy import leaves_list, linkage

        table = table.iloc[leaves_list(linkage(table.to_numpy(), method="average"))]
    values = table.to_numpy().T
    if zscore:
        std = values.std(axis=1, keepdims=True)
        values = (values - values.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)
    ax = _axes(ax, figsize=(7, max(3, len(genes) * 0.25)))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r" if zscore else "viridis")
    ax.set_xticks(np.arange(len(table)), table.index, rotation=90)
    ax.set_yticks(np.arange(len(table.columns)), table.columns)
    ax.figure.colorbar(image, ax=ax, label="Gene z-score" if zscore else layer)
    return ax


def correlation(adata, *, key="sample_correlation", ax=None):
    values = np.asarray(adata.obsp[key])
    ax = _axes(ax, figsize=(6, 5))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(adata.n_obs), adata.obs_names, rotation=90)
    ax.set_yticks(np.arange(adata.n_obs), adata.obs_names)
    ax.figure.colorbar(image, ax=ax, label="Sample correlation")
    return ax


def qc(adata):
    require_adata(adata)
    plt = dependency("matplotlib.pyplot", "plot")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    input_type = adata.uns["mRNA"]["qc"]["parameters"]["input_type"]
    if input_type in {"counts", "estimated_counts"}:
        columns, labels = ("total_counts", "n_genes_detected"), ("Library size", "Detected genes")
    else:
        columns, labels = (
            ("expression_median", "expression_std"),
            ("Median expression", "Expression SD"),
        )
    for ax, column, label in zip(axes, columns, labels, strict=True):
        ax.bar(np.arange(adata.n_obs), adata.obs[column], color="#0072B2")
        ax.set_xticks(np.arange(adata.n_obs), adata.obs_names, rotation=90)
        ax.set_ylabel(label)
    return fig


def gene_expression(adata, *, gene, groupby, layer, ax=None):
    values = expression_frame(adata, layer=layer)[gene]
    levels = pd.unique(adata.obs[groupby])
    groups = [values[adata.obs[groupby].eq(level)].to_numpy() for level in levels]
    ax = _axes(ax)
    ax.boxplot(groups, tick_labels=[str(x) for x in levels], showfliers=False)
    for i, group in enumerate(groups):
        # Deterministic spacing shows individual biological samples.
        ax.scatter(i + 1 + np.linspace(-0.08, 0.08, len(group)), group, s=18, color="#0072B2")
    ax.set(xlabel=groupby, ylabel=layer, title=str(gene))
    return ax


def enrichment(adata, *, key="ora", top_n=15, kind="dot", ax=None):
    if top_n < 1 or kind not in {"dot", "bar"}:
        raise ValueError("Use top_n >= 1 and kind='dot' or 'bar'.")
    table = result(adata, key)
    ax = _axes(ax, figsize=(6, max(3, min(top_n, len(table)) * 0.3)))
    if table.empty:
        ax.text(0.5, 0.5, "No enrichment results", transform=ax.transAxes, ha="center")
        return ax
    pcol = "Adjusted P-value" if "Adjusted P-value" in table else "FDR q-val"
    table = table.sort_values(pcol).head(top_n).iloc[::-1]
    x = -np.log10(pd.to_numeric(table[pcol]).clip(lower=np.finfo(float).tiny))
    y = np.arange(len(table))
    if kind == "bar":
        ax.barh(y, x, color="#0072B2")
    else:
        ax.scatter(x, y, s=60, c=x, cmap="viridis")
    ax.set_yticks(y, table.Term)
    ax.set_xlabel("−log10 adjusted p-value")
    return ax


def gsea(adata, *, term, key="gsea", ax=None):
    profiles = adata.uns["mRNA"][key]["profiles"]
    profile = next((x for x in profiles.values() if x["term"] == term), None)
    if profile is None:
        raise KeyError(f"No enrichment profile for {term!r}.")
    ax = _axes(ax, figsize=(6, 3.5))
    values = profile["running_score"]
    ax.plot(np.arange(len(values)), values, color="#009E73")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.plot(profile["hits"], np.full(len(profile["hits"]), min(values)), "|", color="#333333")
    ax.set(xlabel="Gene rank", ylabel="Running enrichment score", title=term)
    return ax
