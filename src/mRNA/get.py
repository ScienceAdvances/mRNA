"""Extract copies of expression, result tables, and ranked gene lists."""

import numpy as np
import pandas as pd

from ._core import dense, matrix, require_adata


def expression(adata, *, layer=None):
    return pd.DataFrame(
        dense(matrix(adata, layer)).copy(),
        index=adata.obs_names.copy(),
        columns=adata.var_names.copy(),
    )


def result(adata, key="de"):
    require_adata(adata)
    return adata.uns["mRNA"][key]["table"].copy()


def rank_genes(adata, *, key="de", score="statistic", duplicate_policy="error"):
    """Use the full finite signed DE ranking, never a significance-filtered subset."""
    table = result(adata, key)
    if score not in table:
        raise KeyError(f"Ranking column {score!r} is missing.")
    keep = table["status"].ne("prefiltered") & np.isfinite(table[score])
    ranked = pd.Series(
        table.loc[keep, score].to_numpy(), index=table.loc[keep, "gene"], name="score"
    )
    if ranked.index.duplicated().any():
        if duplicate_policy == "error":
            raise ValueError("Duplicate ranking gene IDs require an explicit aggregation policy.")
        if duplicate_policy != "mean":
            raise ValueError("duplicate_policy must be error or mean.")
        ranked = ranked.groupby(level=0).mean()
    return ranked.sort_values(ascending=False, kind="stable")


def tested_genes(adata, *, key="de"):
    table = result(adata, key)
    return pd.Index(table.loc[table["status"].eq("tested"), "gene"])
