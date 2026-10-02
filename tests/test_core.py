import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import mRNA as mr


@pytest.fixture
def small():
    counts = pd.DataFrame(
        [[10, 20, 0], [30, 10, 10]], index=["s1", "s2"], columns=["g1", "g2", "g3"]
    )
    var = pd.DataFrame({"length": [1000, 2000, 500]}, index=counts.columns)
    obs = pd.DataFrame({"condition": ["A", "B"]}, index=counts.index)
    return mr.io.from_frames(counts, obs, var, input_type="counts")


@pytest.mark.parametrize("sparse_input", [False, True])
def test_units_preserve_counts_and_gene_universe(small, sparse_input):
    if sparse_input:
        small.X = sparse.csr_matrix(small.X)
        small.layers["counts"] = sparse.csr_matrix(small.layers["counts"])
    original = mr.get.expression(small, layer="counts")
    mr.pp.cpm(small)
    np.testing.assert_allclose(mr.get.expression(small, layer="cpm").sum(axis=1), 1e6)
    np.testing.assert_allclose(mr.get.expression(small, layer="cpm").loc["s1", "g1"], 1e6 / 3)
    mr.pp.tpm(small, length_definition="fixture exonic lengths")
    np.testing.assert_allclose(mr.get.expression(small, layer="tpm").loc["s1"], [500000, 500000, 0])
    mr.pp.fpkm(small, length_definition="fixture exonic lengths")
    np.testing.assert_allclose(
        mr.get.expression(small, layer="fpkm").loc["s1", "g1"], 10 * 1e9 / (30 * 1000)
    )
    mr.pp.rpkm(small, length_definition="fixture exonic lengths")
    np.testing.assert_allclose(
        mr.get.expression(small, layer="rpkm"), mr.get.expression(small, layer="fpkm")
    )
    pd.testing.assert_frame_equal(mr.get.expression(small, layer="counts"), original)
    assert list(small.var_names) == ["g1", "g2", "g3"]


def test_original_library_denominator_after_filter(small):
    sizes = pd.Series([30, 50], index=small.obs_names)
    selected = small[:, :2].copy()
    mr.pp.cpm(selected, library_sizes=sizes)
    assert selected.layers["cpm"][1, 1] == 200000
    assert selected.layers["cpm"].sum(axis=1)[1] == 800000


def test_sparse_filter_with_zero_detection_threshold(small):
    small.layers["counts"] = sparse.csr_matrix(small.layers["counts"])
    out = mr.pp.filter_genes(small, min_count=0, min_samples=2, min_total_count=10)
    assert out.var_names.tolist() == ["g1", "g2", "g3"]
    assert small.n_vars == 3


@pytest.mark.parametrize("lengths", [[1000, 0, 500], [1000, np.nan, 500], [1000, 2000]])
def test_length_validation(small, lengths):
    with pytest.raises(ValueError, match="length"):
        mr.pp.tpm(small, lengths=lengths, length_definition="fixture")


def test_metadata_alignment_and_no_mutation():
    counts = pd.DataFrame([[5, 8], [10, 20]], index=["s1", "s2"], columns=["g1", "g2"])
    metadata = pd.DataFrame({"condition": ["B", "A"]}, index=["s2", "s1"])
    backup = metadata.copy()
    a = mr.io.from_frames(counts, metadata, input_type="counts")
    assert a.obs.condition.tolist() == ["A", "B"]
    pd.testing.assert_frame_equal(metadata, backup)
    with pytest.raises(ValueError, match="exactly"):
        mr.io.from_frames(counts, metadata.iloc[:1], input_type="counts")


@pytest.mark.parametrize("value", [-1, 0.2, np.nan, np.inf])
def test_invalid_counts_rejected(value):
    frame = pd.DataFrame([[5, value]], index=["s1"], columns=["g1", "g2"])
    with pytest.raises(ValueError):
        mr.io.from_frames(frame, input_type="counts")


def test_copy_qc_log_and_persistence(small, tmp_path):
    transformed = mr.pp.qc(small, copy=True)
    assert "total_counts" not in small.obs
    assert transformed.obs.total_counts.tolist() == [30, 50]
    mr.pp.cpm(transformed)
    mr.pp.log_transform(transformed, layer="cpm", input_type="cpm", key_added="logcpm", set_X=True)
    np.testing.assert_allclose(transformed.X, np.log2(transformed.layers["cpm"] + 1))
    target = mr.io.write(transformed, tmp_path / "test.h5ad")
    loaded = mr.io.read_h5ad(target)
    np.testing.assert_array_equal(loaded.layers["counts"], small.layers["counts"])
    assert loaded.uns["mRNA"]["representations"]["X"] == "logcpm"
    with pytest.raises(ValueError, match="conflicts"):
        mr.pp.qc(loaded, layer=None, input_type="counts")
    with pytest.raises(ValueError, match="overwrite"):
        mr.pp.cpm(small, key_added="counts")
    with pytest.raises(ValueError, match="linear-scale"):
        mr.pp.log_transform(loaded, layer="logcpm", input_type="logcpm", key_added="double_log")


def test_views_and_empty_libraries(small):
    with pytest.raises(ValueError, match="views"):
        mr.pp.qc(small[:, :2])
    with pytest.raises(ValueError, match="empty libraries"):
        mr.io.from_frames(pd.DataFrame([[0, 0]]), input_type="counts")


def test_annotation_is_lossless(small):
    ref = pd.DataFrame({"ensembl": ["g1", "g2", "g2"], "symbol": ["A", "B", "C"]})
    mr.annotation.annotate(small, ref)
    assert small.var.symbol.tolist() == ["A", "", ""]
    assert small.var.symbol_mapping_status.tolist() == ["mapped", "ambiguous", "unmapped"]
    assert list(small.var_names) == ["g1", "g2", "g3"]
    assert small.shape == (2, 3)


def test_version_stripping_both_sides_and_legacy_reference():
    ref = pd.DataFrame({"ensembl": ["ENSG000001.12"], "symbol": ["A"]})
    result = mr.annotation.map_ids(["ENSG000001.9", "ENSG999999.1"], ref)
    assert result.mapped_id.tolist() == ["A", ""]
    assert result.input_id.tolist() == ["ENSG000001.9", "ENSG999999.1"]
    ref = mr.annotation.reference("human", release="legacy")
    assert len(ref) > 10000
    assert ref.attrs["provenance"]["length_definition"] == "undocumented"


def test_transcript_aggregation(small):
    mapping = pd.DataFrame({"transcript": ["g1", "g2", "g3"], "gene": ["A", "A", "B"]})
    summed = mr.annotation.aggregate_transcripts(small, mapping)
    np.testing.assert_array_equal(summed.X, [[30, 0], [40, 10]])
    with pytest.raises(ValueError, match="cover every"):
        mr.annotation.aggregate_transcripts(small, mapping.iloc[:2])


def test_featurecounts_and_fractional_quantifier(tmp_path):
    counts_file = tmp_path / "counts.tsv"
    counts_file.write_text(
        "# generated fixture\nGeneid\tChr\tStart\tEnd\tStrand\tLength\ts1\ts2\ng1\t1\t1\t100\t+\t100\t10\t20\n"
    )
    a = mr.io.read_featurecounts(counts_file)
    assert a.var.length.iloc[0] == 100
    assert a.obs_names.tolist() == ["s1", "s2"]
    quant = tmp_path / "quant.sf"
    quant.write_text(
        "Name\tLength\tEffectiveLength\tTPM\tNumReads\nt1\t1000\t850\t600000\t20.5\nt2\t1200\t1000\t400000\t10.25\n"
    )
    a = mr.io.read_quantifications({"s1": quant})
    np.testing.assert_allclose(a.X, [[20.5, 10.25]])
    assert "counts" not in a.layers
    assert a.layers["effective_length"][0, 0] == 850
    with pytest.raises(KeyError):
        mr.pp.cpm(a)


def test_significance_relabeling_and_missing_padj():
    table = pd.DataFrame(
        {
            "log2_fold_change": [2, -2, 3],
            "adj_p_value": [0.001, 0.5, np.nan],
            "change": ["Down", "Up", "Up"],
        }
    )
    assert mr.tl.label_significance(table).change.tolist() == ["Up", "NS", "NS"]
    assert table.change.tolist() == ["Down", "Up", "Up"]


def test_pca_mds_and_batch_contract():
    a = mr.datasets.toy_microarray()
    original = a.X.copy()
    mr.tl.pca(a, layer=None, input_type="log_expression", n_components=3)
    mr.tl.mds(a, layer=None, input_type="log_expression")
    assert a.obsm["X_pca"].shape == (12, 3)
    assert a.obsm["X_mds"].shape == (12, 2)
    corr = mr.tl.sample_correlation(a, layer=None, input_type="log_expression")
    np.testing.assert_allclose(np.diag(corr), 1)
    mr.pp.remove_batch_effect(
        a, layer=None, input_type="log_expression", batch_col="batch", protect_design="~condition"
    )
    np.testing.assert_array_equal(a.X, original)
    with pytest.raises(ValueError, match="exploratory"):
        mr.tl.fit(
            a,
            design="~condition",
            method="limma",
            input_type="log_expression",
            layer="batch_corrected",
        )


def test_design_confounding_and_saturation():
    a = mr.datasets.toy_microarray()
    a.obs["confounded"] = a.obs.condition
    with pytest.raises(ValueError, match="rank deficient"):
        mr.tl.fit(
            a,
            method="limma",
            design="~condition+confounded",
            input_type="log_expression",
            layer=None,
        )
    a.obs["sample_factor"] = a.obs_names
    with pytest.raises(ValueError, match="degrees of freedom"):
        mr.tl.fit(
            a, method="limma", design="~sample_factor", input_type="log_expression", layer=None
        )
