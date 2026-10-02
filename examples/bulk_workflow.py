"""Run all three bulk DE engines, functional analysis, and publication export."""

from pathlib import Path

import mRNA as mr

outdir = Path(__file__).parent / "output"
outdir.mkdir(parents=True, exist_ok=True)
adata = mr.datasets.toy_counts()
mr.pp.qc(adata)
# Real workflows can import supplied TPM; this toy dataset uses synthetic lengths.
mr.pp.tpm(adata, length_definition="synthetic gene lengths")
mr.pp.log_transform(adata, layer="tpm", input_type="tpm", key_added="logtpm", base=2, pseudocount=1)
mr.tl.pca(adata, layer="logtpm", input_type="logtpm", n_components=5)
mr.tl.sample_correlation(adata, layer="logtpm", input_type="logtpm")
for method in ("deseq2", "edgepython", "limma"):
    # Each DE backend fits raw counts and applies its own normalization.
    model = mr.tl.fit(
        adata, design="~batch+condition", method=method, layer="counts", input_type="counts"
    )
    table = mr.tl.contrast(
        model, contrast=("condition", "treated", "control"), key_added=f"de_{method}"
    )
    table.to_csv(outdir / f"{method}.csv", index=False)

gene_sets = {
    "Synthetic upregulated": list(adata.var_names[:15]),
    "Synthetic downregulated": list(adata.var_names[15:30]),
}
mr.tl.ora(
    adata, gene_sets=gene_sets, resource_version="synthetic-v1", de_key="de_deseq2", direction="up"
)
mr.tl.gsea(
    adata,
    gene_sets=gene_sets,
    resource_version="synthetic-v1",
    de_key="de_deseq2",
    backend_options={"min_size": 5, "permutation_num": 100},
)
mr.tl.score_gene_sets(
    adata,
    gene_sets=gene_sets,
    resource_version="synthetic-v1",
    layer="logtpm",
    input_type="logtpm",
    backend_options={"min_size": 5},
)
mr.io.write(adata, outdir / "bulk.h5ad")
mr.io.report(
    adata,
    outdir / "bulk_report.html",
    keys=["de_deseq2", "de_edgepython", "de_limma", "ora", "gsea"],
)

# Native figures can be passed directly to atopos.imagesave().
ax = mr.pl.volcano(adata, key="de_deseq2")
ax.figure.savefig(outdir / "volcano.png", dpi=180, bbox_inches="tight")
print(f"Outputs: {outdir.resolve()}")
