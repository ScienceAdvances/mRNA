import mRNA as mr


def test_sample_outlier_flags_do_not_drop_samples():
    a = mr.datasets.toy_counts()
    a.layers["counts"][-1] *= 100
    mr.pp.qc(a)
    mr.pp.mad_outliers(a)
    assert a.obs.qc_outlier.iloc[-1]
    assert a.n_obs == 12


def test_microarray_qc_has_expression_metrics():
    a = mr.datasets.toy_microarray()
    mr.pp.qc(a, layer=None, input_type="log_expression")
    assert "expression_median" in a.obs
    assert "expression_std" in a.obs
    assert "total_counts" not in a.obs
