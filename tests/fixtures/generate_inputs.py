"""Generate deterministic input files for independent R reference calculations."""

from pathlib import Path

import pandas as pd

import mRNA as mr
from mRNA._design import build_design

outdir = Path(__file__).parent
counts = mr.datasets.toy_counts(seed=19)
arrays = mr.datasets.toy_microarray(seed=19)
mr.get.expression(counts, layer="counts").to_csv(outdir / "counts.csv")
mr.get.expression(arrays).to_csv(outdir / "microarray.csv")
counts.obs.to_csv(outdir / "metadata.csv")
design, _ = build_design(counts.obs, "~batch+condition")
design.to_csv(outdir / "design.csv")
pd.Series([0, 0, 1], index=design.columns, name="contrast").to_csv(outdir / "contrast.csv")
