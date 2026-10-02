import numpy as np
import pandas as pd
import pytest

import mRNA as mr


@pytest.mark.integration
@pytest.mark.parametrize(
    "method,package", [("deseq2", "pydeseq2"), ("edgepython", "edgepython"), ("limma", "pylimma")]
)
def test_real_count_backends_direction_reuse_and_h5ad(method, package, tmp_path):
    pytest.importorskip(package)
    a = mr.datasets.toy_counts()
    a.X[:, -1] = 0
    a.layers["counts"][:, -1] = 0
    original = a.layers["counts"].copy()
    model = mr.tl.fit(a, design="~batch+condition", method=method)
    first = mr.tl.contrast(model, contrast=("condition", "treated", "control"), key_added="forward")
    second = mr.tl.contrast(
        model, contrast=("condition", "control", "treated"), key_added="reverse"
    )
    np.testing.assert_allclose(
        first.log2_fold_change, -second.log2_fold_change, rtol=1e-7, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(first.p_value, second.p_value, rtol=1e-7, atol=1e-9, equal_nan=True)
    assert first.loc[first.gene.isin(a.var_names[:15]), "log2_fold_change"].median() > 1.5
    assert first.loc[first.gene.isin(a.var_names[15:30]), "log2_fold_change"].median() < -1.5
    assert first.iloc[-1].status == "prefiltered"
    assert first.iloc[-1].reason == "expression_filter"
    assert pd.isna(first.iloc[-1].adj_p_value)
    np.testing.assert_array_equal(a.layers["counts"], original)
    mr.tl.compare_results(a, keys=["forward", "reverse"])
    loaded = mr.io.read_h5ad(mr.io.write(a, tmp_path / f"{method}.h5ad"))
    assert len(mr.get.result(loaded, "forward")) == a.n_vars


@pytest.mark.integration
def test_limma_matches_independent_model_and_paired_design():
    pytest.importorskip("pylimma")
    a = mr.datasets.toy_microarray()
    a.obs["donor"] = [f"d{i}" for i in range(6)] * 2
    model = mr.tl.fit(
        a, design="~donor+condition", method="limma", input_type="log_expression", layer=None
    )
    t = mr.tl.contrast(model, contrast=("condition", "treated", "control"))
    paired_mean = a.X[6:].mean(axis=0) - a.X[:6].mean(axis=0)
    np.testing.assert_allclose(t.log2_fold_change, paired_mean, atol=1e-12)
    assert np.isfinite(t.adj_p_value).all()


@pytest.mark.integration
def test_explicit_vector_interaction_and_replicate_validation():
    pytest.importorskip("pylimma")
    a = mr.datasets.toy_microarray()
    model = mr.tl.fit(
        a, design="~batch*condition", method="limma", input_type="log_expression", layer=None
    )
    named = mr.tl.contrast(model, contrast=("condition", "treated", "control"))
    vector = np.zeros(model.design.shape[1])
    vector[model.design.columns.get_loc("condition[T.treated]")] = 1
    vector[-1] = 0.5
    numeric = mr.tl.contrast(model, contrast=vector)
    np.testing.assert_allclose(named.log2_fold_change, numeric.log2_fold_change, atol=1e-12)
    with pytest.raises(ValueError, match="nonzero"):
        mr.tl.contrast(model, contrast=np.zeros(len(vector)))
    with pytest.raises(ValueError, match="biological samples"):
        mr.tl.contrast(model, contrast=("condition", "missing", "control"))


@pytest.mark.integration
@pytest.mark.parametrize("test", ["ql", "lrt"])
def test_edge_treat(test):
    pytest.importorskip("edgepython")
    a = mr.datasets.toy_counts()
    model = mr.tl.fit(a, design="~condition", method="edgepython", backend_options={"test": test})
    ordinary = mr.tl.contrast(model, contrast=("condition", "treated", "control"))
    treat = mr.tl.contrast(
        model, contrast=("condition", "treated", "control"), backend_options={"lfc": 1.0}
    )
    assert (treat.p_value >= ordinary.p_value - 1e-12).all()
    assert treat.statistic_type.str.contains("TREAT").all()


@pytest.mark.integration
def test_deseq_shrink_keeps_original_statistics():
    pytest.importorskip("pydeseq2")
    a = mr.datasets.toy_counts()
    model = mr.tl.fit(a, design="~condition", method="deseq2")
    original = mr.tl.contrast(model, contrast=("condition", "treated", "control"))
    shrunk = mr.tl.contrast(
        model,
        contrast=("condition", "treated", "control"),
        shrink_coefficient="condition[T.treated]",
    )
    np.testing.assert_allclose(shrunk.log2_fold_change, original.log2_fold_change)
    np.testing.assert_allclose(shrunk.p_value, original.p_value)
    assert "log2_fold_change_shrunken" in shrunk
    assert np.isfinite(shrunk.log2_fold_change_shrunken).all()


@pytest.mark.integration
@pytest.mark.parametrize("method", ["tmm", "deseq2"])
def test_statistical_normalization(method):
    pytest.importorskip("edgepython" if method == "tmm" else "pydeseq2")
    a = mr.datasets.toy_counts()
    original = a.layers["counts"].copy()
    mr.pp.normalize(a, method=method)
    assert np.isfinite(a.layers["normalized_counts"]).all()
    np.testing.assert_array_equal(a.layers["counts"], original)
    with pytest.raises(ValueError, match="conflicts"):
        mr.tl.fit(a, design="~condition", method="deseq2", layer="normalized_counts")
