import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

import mRNA as mr


@pytest.fixture
def functional():
    pytest.importorskip("pylimma")
    pytest.importorskip("gseapy")
    a = mr.datasets.toy_microarray()
    m = mr.tl.fit(a, design="~condition", method="limma", input_type="log_expression", layer=None)
    mr.tl.contrast(m, contrast=("condition", "treated", "control"))
    return a


@pytest.mark.integration
def test_ora_uses_tested_background(functional, tmp_path):
    a = functional
    table = mr.get.result(a)
    table.loc[100:, "status"] = "prefiltered"
    a.uns["mRNA"]["de"]["table"] = table
    sets = {"up": list(a.var_names[:15]), "outside": list(a.var_names[180:])}
    result = mr.tl.ora(a, gene_sets=sets, resource_version="fixture-v1", direction="up")
    params = a.uns["mRNA"]["ora"]["parameters"]
    assert len(params["background"]) == 100
    selected = set(params["selected_genes"])
    expected = hypergeom.sf(len(selected & set(sets["up"])) - 1, 100, 15, len(selected))
    np.testing.assert_allclose(result.loc[result.Term.eq("up"), "P-value"], expected, rtol=1e-12)
    mr.io.write(a, tmp_path / "ora.h5ad")


@pytest.mark.integration
def test_gsea_direction_full_ranks_profiles_and_reproducibility(functional, tmp_path):
    a = functional
    sets = {"up": list(a.var_names[:15]), "down": list(a.var_names[15:30])}
    options = {"min_size": 5, "permutation_num": 50}
    first = mr.tl.gsea(
        a, gene_sets=sets, resource_version="fixture", seed=42, backend_options=options
    )
    second = mr.tl.gsea(
        a,
        gene_sets=sets,
        resource_version="fixture",
        seed=42,
        key_added="gsea2",
        backend_options=options,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.set_index("Term").loc["up", "NES"] > 0
    assert first.set_index("Term").loc["down", "NES"] < 0
    assert len(a.uns["mRNA"]["gsea"]["parameters"]["ranked_gene_ids"]) == 200
    assert (first.Lead_genes.str.len() > 0).all()
    loaded = mr.io.read_h5ad(mr.io.write(a, tmp_path / "gsea.h5ad"))
    assert len(loaded.uns["mRNA"]["gsea"]["profiles"]) == 2


@pytest.mark.integration
@pytest.mark.parametrize("method", ["ssgsea", "gsva"])
def test_sample_scores_and_h5ad(functional, method, tmp_path):
    a = functional
    sets = {"up": list(a.var_names[:15]), "down": list(a.var_names[15:30])}
    original = a.X.copy()
    mr.tl.score_gene_sets(
        a,
        gene_sets=sets,
        resource_version="fixture",
        method=method,
        layer=None,
        input_type="log_expression",
        backend_options={"min_size": 5},
    )
    scores = a.obsm[f"score_{method}"]
    assert scores.shape == (12, 2)
    assert (
        scores.loc[a.obs.condition.eq("treated"), "up"].mean()
        > scores.loc[a.obs.condition.eq("control"), "up"].mean()
    )
    np.testing.assert_array_equal(a.X, original)
    mr.io.write(a, tmp_path / f"{method}.h5ad")


@pytest.mark.integration
def test_activity_signed_targets_and_source_preservation(functional, tmp_path):
    pytest.importorskip("decoupler")
    a = functional
    net = pd.DataFrame(
        {
            "source": ["positive"] * 15 + ["negative"] * 15,
            "target": list(a.var_names[:15]) * 2,
            "weight": [1.0] * 15 + [-1.0] * 15,
        }
    )
    shape = a.shape
    mr.tl.activity(a, net, resource_version="fixture", layer=None, input_type="log_expression")
    scores = a.obsm["activity_ulm"]
    np.testing.assert_allclose(scores.positive, -scores.negative, atol=1e-6)
    assert a.shape == shape
    mr.io.write(a, tmp_path / "activity.h5ad")


@pytest.mark.integration
def test_plots_report_and_mapping_validation(functional, tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    a = functional
    sets = {"up": list(a.var_names[:15])}
    mr.tl.ora(a, gene_sets=sets, resource_version="fixture", direction="up")
    mr.tl.gsea(
        a,
        gene_sets=sets,
        resource_version="fixture",
        backend_options={"min_size": 5, "permutation_num": 30},
    )
    mr.pp.qc(a, layer=None, input_type="log_expression")
    mr.tl.pca(a, layer=None, input_type="log_expression", n_components=2)
    mr.tl.sample_correlation(a, layer=None, input_type="log_expression")
    axes = [
        mr.pl.volcano(a),
        mr.pl.ma(a),
        mr.pl.embedding(a, color="condition"),
        mr.pl.heatmap(
            a, genes=list(a.var_names[:5]), layer=None, zscore=True, cluster_samples=True
        ),
        mr.pl.correlation(a),
        mr.pl.gene_expression(a, gene=a.var_names[0], groupby="condition", layer=None),
        mr.pl.enrichment(a),
        mr.pl.gsea(a, term="up"),
    ]
    for i, ax in enumerate(axes):
        ax.figure.savefig(tmp_path / f"plot_{i}.png")
        assert (tmp_path / f"plot_{i}.png").stat().st_size > 1000
        plt.close(ax.figure)
    fig = mr.pl.qc(a)
    fig.savefig(tmp_path / "qc.png")
    plt.close(fig)
    path = mr.io.report(a, tmp_path / "report.html", title="<escaped title>")
    assert "&lt;escaped title&gt;" in path.read_text()
    assert (tmp_path / "report_tables" / "provenance.json").is_file()
    a.var["symbol"] = ["duplicate"] * a.n_vars
    with pytest.raises(ValueError, match="duplicate"):
        mr.tl.gsea(a, gene_sets=sets, resource_version="fixture", gene_id_col="symbol")


def test_gene_set_input_validation():
    from mRNA._enrichment import read_gene_sets

    with pytest.raises(TypeError, match="iterable"):
        read_gene_sets({"bad": "one-string"})
    with pytest.raises(ValueError, match="Nonempty"):
        read_gene_sets({"bad": []})


@pytest.mark.integration
def test_empty_ora_is_serializable(functional, tmp_path):
    table = mr.tl.ora(
        functional, gene_sets={"outside": ["not_in_expression"]}, resource_version="fixture"
    )
    assert table.empty
    mr.io.write(functional, tmp_path / "empty.h5ad")
