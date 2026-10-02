"""Independent R reference fixtures; R is not required to run these tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mRNA as mr

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind,input_type,layer",
    [
        ("microarray", "log_expression", None),
        ("voom", "counts", "counts"),
    ],
)
def test_limma_against_r_reference(kind, input_type, layer):
    pytest.importorskip("pylimma")
    pytest.importorskip("edgepython")
    expr = pd.read_csv(
        FIXTURES / ("microarray.csv" if kind == "microarray" else "counts.csv"), index_col=0
    )
    obs = pd.read_csv(FIXTURES / "metadata.csv", index_col=0)
    a = mr.io.from_frames(expr, obs, input_type=input_type)
    model = mr.tl.fit(
        a, design="~batch+condition", method="limma", input_type=input_type, layer=layer
    )
    table = mr.tl.contrast(model, contrast=("condition", "treated", "control")).set_index("gene")
    ref = pd.read_csv(FIXTURES / f"limma_{kind}_reference.csv").set_index("gene").loc[table.index]
    for col, reference in [
        ("log2_fold_change", "logFC"),
        ("p_value", "P.Value"),
        ("adj_p_value", "adj.P.Val"),
    ]:
        np.testing.assert_allclose(table[col], ref[reference], rtol=1e-8, atol=1e-10)


@pytest.mark.integration
def test_edge_ql_against_r_reference():
    pytest.importorskip("edgepython")
    expr = pd.read_csv(FIXTURES / "counts.csv", index_col=0)
    obs = pd.read_csv(FIXTURES / "metadata.csv", index_col=0)
    a = mr.io.from_frames(expr, obs, input_type="counts")
    model = mr.tl.fit(a, design="~batch+condition", method="edgepython")
    table = mr.tl.contrast(model, contrast=("condition", "treated", "control")).set_index("gene")
    ref = pd.read_csv(FIXTURES / "edge_ql_reference.csv").set_index("gene").loc[table.index]
    # Documented observed QL differences; do not claim bitwise equivalence.
    np.testing.assert_allclose(table.log2_fold_change, ref.logFC, rtol=0, atol=1e-4)
    np.testing.assert_allclose(table.p_value, ref.PValue, rtol=0, atol=2e-4)
    np.testing.assert_allclose(table.adj_p_value, ref.FDR, rtol=0, atol=4e-4)
    np.testing.assert_array_equal(table.adj_p_value < 0.05, ref.FDR < 0.05)
    normalization = pd.read_csv(FIXTURES / "tmm_reference.csv")
    np.testing.assert_allclose(
        model.parameters["norm_factors"], normalization.norm_factor, rtol=1e-12
    )
