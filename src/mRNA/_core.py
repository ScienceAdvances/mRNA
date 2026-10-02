"""Shared validation, matrix access, and serializable provenance."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

NONNEGATIVE = {"counts", "estimated_counts", "normalized_counts", "cpm", "tpm", "fpkm", "rpkm"}
INPUT_TYPES = NONNEGATIVE | {"log_expression", "logcpm", "logtpm", "vst"}


def dependency(module, extra):
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module.split(".")[0]:
            raise
        raise ImportError(f"Install mRNA[{extra}] to use {module}.") from exc


def require_adata(adata):
    if not isinstance(adata, AnnData):
        raise TypeError("Expected an AnnData object with samples in obs and genes in var.")
    if adata.isbacked:
        raise ValueError("Materialize backed data with to_memory() before analysis.")
    if adata.is_view:
        raise ValueError("Materialize AnnData views with copy() before analysis.")
    for labels, axis in ((adata.obs_names, "sample"), (adata.var_names, "gene")):
        if not labels.is_unique or labels.isna().any() or any(str(x) == "" for x in labels):
            raise ValueError(f"Unique, nonmissing, nonempty {axis} IDs are required.")
    if not adata.n_obs or not adata.n_vars:
        raise ValueError("Expression data must contain samples and genes.")
    return adata


def matrix(adata, layer=None, *, input_type=None):
    require_adata(adata)
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Expression layer {layer!r} is missing.")
    m = adata.X if layer is None else adata.layers[layer]
    if m is None or len(m.shape) != 2:
        raise ValueError("A two-dimensional expression matrix is required.")
    values = m.data if sparse.issparse(m) else np.asarray(m)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Expression values must be finite numeric values.")
    if input_type is not None:
        if input_type not in INPUT_TYPES:
            raise ValueError(f"Unsupported input_type: {input_type!r}.")
        known = adata.uns.get("mRNA", {}).get("representations", {}).get(layer or "X")
        if known is not None and known != input_type:
            raise ValueError(f"Stored representation {known!r} conflicts with {input_type!r}.")
        if input_type in NONNEGATIVE and (values < 0).any():
            raise ValueError(f"{input_type} requires nonnegative expression values.")
        if input_type == "counts" and not np.equal(values, np.floor(values)).all():
            raise ValueError("Raw counts must be nonnegative integers; do not round TPM/CPM.")
    return m


def dense(m):
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


def totals(m, axis):
    return np.asarray(m.sum(axis=axis)).ravel()


def working(adata, copy):
    require_adata(adata)
    return adata.copy() if copy else adata


def jsonable(value):
    if value is None:
        return "none"
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, pd.Index)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def record(adata, key, params, *, backend="mRNA", table=None):
    from . import __version__

    if not isinstance(key, str) or not key or "/" in key or key == "representations":
        raise ValueError(
            "Use a nonempty result key without slashes or the reserved name representations."
        )
    try:
        backend_version = __version__ if backend == "mRNA" else version(backend)
    except PackageNotFoundError:
        backend_version = "unknown"
    entry = {
        "parameters": jsonable(params),
        "backend": backend,
        "backend_version": backend_version,
        "schema_version": "1",
    }
    if table is not None:
        entry["table"] = table.copy()
    adata.uns.setdefault("mRNA", {})[key] = entry


def store_layer(adata, key, values, representation, params, *, set_X=False):
    if not isinstance(key, str) or not key or "/" in key or key == "representations":
        raise ValueError(
            "Use a nonempty layer key without slashes or the reserved name representations."
        )
    if key == "counts":
        raise ValueError("Derived expression must not overwrite the original counts layer.")
    adata.layers[key] = values
    reps = adata.uns.setdefault("mRNA", {}).setdefault("representations", {})
    reps[key] = representation
    if set_X:
        adata.X = values.copy()
        reps["X"] = representation
    record(adata, key, params)
