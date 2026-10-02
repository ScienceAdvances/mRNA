"""Formulaic design construction and explicit contrast vectors."""

import numpy as np
import pandas as pd
from formulaic import model_matrix


def build_design(obs, design):
    if isinstance(design, str):
        model = model_matrix(design, obs.copy(), na_action="raise")
        if not hasattr(model, "model_spec"):
            raise ValueError("Use a one-sided design formula, e.g. '~ batch + condition'.")
        spec = model.model_spec
        frame = pd.DataFrame(model, index=obs.index).astype(float)
    elif isinstance(design, pd.DataFrame):
        if not design.index.is_unique or set(design.index) != set(obs.index):
            raise ValueError("Design matrix sample IDs must match obs exactly.")
        frame, spec = design.loc[obs.index].astype(float).copy(), None
    else:
        raise TypeError("design must be a formula string or a labeled DataFrame.")
    if not frame.columns.is_unique or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Design columns must be unique and values finite.")
    if frame.shape[1] == 0 or np.linalg.matrix_rank(frame) < frame.shape[1]:
        raise ValueError("The design is rank deficient; check confounded factors.")
    if len(frame) <= frame.shape[1]:
        raise ValueError("The design has no residual degrees of freedom.")
    return frame, spec


def contrast_vector(model, contrast, *, min_replicates=2):
    if isinstance(contrast, (tuple, list)) and len(contrast) == 3 and isinstance(contrast[0], str):
        factor, comparison, reference = contrast
        obs = model.metadata
        if model.spec is None:
            raise ValueError("Named comparisons require a formula design; supply a numeric vector.")
        if factor not in obs or pd.api.types.is_numeric_dtype(obs[factor]):
            raise ValueError("Named comparisons require a categorical factor in metadata.")
        if comparison == reference:
            raise ValueError("Comparison and reference must differ.")
        for level in (comparison, reference):
            if obs[factor].eq(level).sum() < min_replicates:
                raise ValueError(
                    f"At least {min_replicates} biological samples are required in {level!r}."
                )
        designs = []
        for level in (comparison, reference):
            changed = obs.copy()
            changed[factor] = level
            designs.append(np.asarray(model.spec.get_model_matrix(changed)).mean(axis=0))
        vector = designs[0] - designs[1]
        label = f"{factor}:{comparison}_vs_{reference}"
    else:
        vector = np.asarray(contrast, dtype=float)
        label = "numeric_contrast"
    if (
        vector.shape != (model.design.shape[1],)
        or not np.isfinite(vector).all()
        or not vector.any()
    ):
        raise ValueError("Contrast must be a finite nonzero vector matching the design columns.")
    return vector, label
