"""Deterministic bulk examples with independent samples and known effect direction."""

import numpy as np
import pandas as pd

from .io import from_frames


def toy_counts(*, n_samples=12, n_genes=200, seed=0):
    if n_samples < 6 or n_genes < 30 or n_samples % 2:
        raise ValueError("Use an even n_samples >= 6 and n_genes >= 30.")
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "condition": ["control"] * (n_samples // 2) + ["treated"] * (n_samples // 2),
            "batch": ["b1", "b2"] * (n_samples // 2),
        },
        index=[f"sample_{i:02d}" for i in range(n_samples)],
    )
    means = rng.lognormal(4.2, 0.7, n_genes)
    mu = np.broadcast_to(means, (n_samples, n_genes)).copy()
    n_de = min(15, n_genes // 4)
    mu[n_samples // 2 :, :n_de] *= 4
    mu[n_samples // 2 :, n_de : 2 * n_de] *= 0.25
    counts = rng.negative_binomial(20, 20 / (20 + mu))
    var = pd.DataFrame(
        {
            "length": rng.integers(500, 5000, n_genes),
            "truth": ["up"] * n_de + ["down"] * n_de + ["unchanged"] * (n_genes - 2 * n_de),
        },
        index=[f"gene_{i:04d}" for i in range(n_genes)],
    )
    return from_frames(
        pd.DataFrame(counts, index=obs.index, columns=var.index), obs, var, input_type="counts"
    )


def toy_microarray(*, n_samples=12, n_genes=200, seed=0):
    template = toy_counts(n_samples=n_samples, n_genes=n_genes, seed=seed)
    rng = np.random.default_rng(seed)
    expr = rng.normal(7, 0.35, (n_samples, n_genes))
    expr[n_samples // 2 :, template.var.truth.eq("up")] += 2
    expr[n_samples // 2 :, template.var.truth.eq("down")] -= 2
    return from_frames(
        pd.DataFrame(expr, index=template.obs_names, columns=template.var_names),
        template.obs,
        template.var,
        input_type="log_expression",
    )
