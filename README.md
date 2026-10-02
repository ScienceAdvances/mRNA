# [mRNA](https://github.com/ScienceAdvances/mRNA)

AnnData-native analysis of **bulk RNA-seq and processed expression microarrays**.
mRNA is independent of cellscope and immune. Figures can be styled and exported
with atopos; all plots return native Matplotlib objects without implicit saving.
The distribution name is `mRNA`; the standard Python import is `import mRNA as mr`.

## Install

Python **3.12 or newer**. Python 3.12, 3.13, and 3.14 are covered by CI. This is
an initial local release; install from this checkout rather than assuming a PyPI
release exists:

```sh
uv venv --python 3.14
uv pip install -e '.[all,dev]'
```

Core dependencies include AnnData, NumPy, pandas, SciPy, and Formulaic. Optional
extras: `deseq2`, `edge`, `limma`, `enrichment`, `activity`, `plot`, and `dev`.
The tested contemporary baseline is PyDESeq2 0.5.4, pylimma 0.2.0, edgePython
0.2.6, GSEApy 1.3.1, and decoupler 2.2.0. See `requirements-tested.txt` for exact
versions used during local verification.

For manual PyPI releases with uv and Hatchling, see [PUBLISHING.md](PUBLISHING.md)
and `scripts/release.py`.

## Data model

- `AnnData.obs`: biological samples and their condition, batch, donor, or other covariates.
- `AnnData.var`: unique stable gene IDs, annotation, and explicitly sourced lengths.
- `AnnData.layers['counts']`: original nonnegative integer counts.
- `AnnData.layers['cpm'/'tpm'/...]`: derived representations, stored separately.
- `AnnData.X`: initial input or an explicitly selected active representation.
- `AnnData.obsm`: PCA/MDS coordinates and sample-level gene-set/activity scores.
- `AnnData.obsp`: sample correlations.
- `AnnData.uns['mRNA']`: results and serializable parameters/backend provenance.

Samples are rows and genes are columns. Derived layers preserve original counts.
Inputs require an explicit representation and metadata IDs must match exactly.
Analysis modifies AnnData unless `copy=True` is exposed and supplied. Filtered
data are returned as copies; views/backed data must be materialized explicitly.
Fitted native models are returned separately, not embedded into `.h5ad`.

## Bulk workflow: log2(TPM + 1) exploration and count-based DE

The bulk example uses `log2(TPM + 1)` for PCA, sample correlations, expression
heatmaps, and sample-level gene-set scores. Differential expression uses the
original counts and each backend's normalization. ORA and preranked GSEA use
the differential-expression results rather than the TPM matrix.

```python
import mRNA as mr

adata = mr.datasets.toy_counts()
mr.pp.qc(adata)
# Synthetic lengths are for this toy example; real data can supply quantified TPM.
mr.pp.tpm(adata, lengths="length", length_definition="synthetic example lengths")
mr.pp.log_transform(adata, layer="tpm", input_type="tpm", key_added="logtpm", base=2, pseudocount=1)
mr.tl.pca(adata, layer="logtpm", input_type="logtpm", n_components=5)
mr.tl.sample_correlation(adata, layer="logtpm", input_type="logtpm")

model = mr.tl.fit(
    adata, design="~ batch + condition", method="deseq2", layer="counts", input_type="counts"
)
de = mr.tl.contrast(model, contrast=("condition", "treated", "control"))

gene_sets = {"example_up": list(adata.var_names[:15])}
mr.tl.ora(adata, gene_sets=gene_sets, resource_version="example-v1", direction="up")
mr.tl.gsea(
    adata,
    gene_sets=gene_sets,
    resource_version="example-v1",
    seed=0,
    backend_options={"min_size": 5, "permutation_num": 100},
)

ax = mr.pl.volcano(adata)
mr.io.write(adata, "results/bulk.h5ad")
```

Swap `method='edgepython'` for TMM/QL GLM inference, or `method='limma'` for
TMM/voom followed by moderated t-tests. Counts-based limma requires the `edge`
extra as well as `limma`; `all` includes both. DESeq2 uses its own size factors.
Generic conversion of TPM/CPM into raw counts is not supported.

If TPM is already available as a DataFrame with samples in rows and genes in
columns, import it directly without recalculating TPM or logging it twice:

```python
adata_tpm = mr.io.from_frames(tpm, metadata, input_type="tpm")
mr.pp.log_transform(
    adata_tpm, layer=None, input_type="tpm", key_added="logtpm", base=2, pseudocount=1
)
mr.tl.pca(adata_tpm, layer="logtpm", input_type="logtpm", n_components=5)
```

Here `tpm` and `metadata` are user-supplied DataFrames. A TPM-only dataset
supports expression exploration; the count-based DE engines require separately
available original counts. See `examples/bulk_workflow.py` for the complete
workflow with both representations.

## Processed microarrays

```python
adata = mr.datasets.toy_microarray()
model = mr.tl.fit(
    adata, design="~ condition", method="limma", layer=None, input_type="log_expression"
)
de = mr.tl.contrast(model, contrast=("condition", "treated", "control"))
```

Raw CEL/RMA processing and FASTQ alignment/quantification are upstream stages.
Quantifier import preserves estimated counts and effective lengths; estimated
counts are not silently rounded into raw integer counts.

## Designs, contrasts, and results

Use independent biological replicates; cells and technical libraries are not
independent bulk samples. Named comparisons require at least two samples in
each condition. Use `~ donor + condition` for paired studies or
`~ batch + condition` for batch adjustment. Confounded/rank-deficient and
saturated designs are rejected. Random-effect models are not implemented.

Formulaic builds the same labeled design for all three adapters. Named contrasts
in interaction designs are marginal contrasts averaged over observed covariates.
Pass a numeric coefficient vector to request a specific simple effect. Reuse
`tl.fit` results for additional contrasts.

Canonical result fields: `gene`, `log2_fold_change`, `statistic`, `p_value`,
`adj_p_value`, `method`, `contrast`, `status`, `reason`, and `change`. Native
backend columns are retained. `statistic_type` and `mean_expression_type`
identify their different meanings/units. Filtered and untestable genes stay in
the table with missing statistics and an explanation. Backend independent
filtering may yield a missing adjusted p-value, which is preserved.

- PyDESeq2: Wald tests; optional `shrink_coefficient` for a positive single-coefficient
  contrast, with shrunken LFC and SE saved separately from unshrunken statistics.
- edgePython: default QL, or `backend_options={'test': 'lrt'}` at fitting time.
  TREAT is requested at contrast time with `backend_options={'lfc': threshold}`.
  Signed sqrt(F/LR) is used for rankings; native F/LR is retained.
  TREAT does not report an F/LR statistic; its canonical statistic stays missing.
  Choose an explicit alternative ranking column for TREAT results.
- pylimma: contrasts followed by empirical Bayes moderation. mRNA uses
  **john-mulvey/pylimma**, not the different `python-limma` distribution that
  exports the same import namespace. Do not install both in one environment.

`lfc_threshold` controls result labeling, not the statistical null hypothesis.
No automated majority-vote combination of backend p-values is provided.

## Annotation and functional scores

`annotation.map_ids` returns mapped/unmapped/ambiguous diagnostics without
duplicating the expression matrix. `annotation.annotate` adds `var` columns
and leaves stable gene IDs untouched. A custom table supports Ensembl, Entrez,
symbol, alias, or other identifiers through explicit column selection.
`annotation.reference('human', release='legacy')` and its mouse equivalent
expose archived atopos tables with their historical release labels. They are
not advertised as latest annotations; their gene-length derivation is unknown.
TPM/FPKM always require an explicit length definition. CPM requires no mapping.

Enrichment uses local GMT files or dictionaries with a required resource version
and content hash. ORA uses the tested-gene background and supports separate up/
down sets. GSEA uses the full finite fitted-gene ranking. `gene_id_col='symbol'`
can select mapped annotation; mapping loss is recorded and duplicate ranks
require an explicit policy. Do not provide a significance-filtered ranking.

`tl.score_gene_sets` stores ssGSEA/GSVA scores in `obsm`. `tl.activity` uses
decoupler 2.x and an explicit TF/pathway network. These are sample-level scores,
not integer count matrices. `pp.remove_batch_effect` produces an explicitly
exploratory layer; differential expression should model batch covariates.

## Validation and scope

```sh
ruff check .
ruff format --check .
MPLBACKEND=Agg python -m pytest
python -m build
```

The tests cover data integrity, denominator/length handling, mapping ambiguity,
real optional backend calculations, effect direction, reusable contrasts,
enrichment backgrounds, gene-set scores, plots, and `.h5ad` round trips.
See `DESIGN.md`, `VALIDATION.md`, and the examples for details.

This first implementation targets the P0/P1 matrix-analysis workflow. Coexpression
networks, deconvolution, repeated-measures random effects, differential splicing,
cross-cohort meta-analysis, and survival/prediction modules remain explicit
future extensions, rather than unvalidated wrappers.
