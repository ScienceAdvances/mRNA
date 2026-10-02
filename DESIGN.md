# mRNA: architecture and statistical contract

mRNA owns bulk transcriptomics matrix analysis. It is independent of cellscope
(single-cell RNA) and immune (receptor repertoires). atopos is an optional
consumer of native plot objects, not an analysis dependency.

## Storage and API

The underlying object is AnnData, with biological samples in `obs` and unique
stable gene IDs in `var`. A count input is preserved in `layers['counts']`.
Unit conversions and transforms write distinct layers. All representations are
declared in `uns['mRNA']['representations']`; declarations are checked before
numeric validation. Existing externally constructed AnnData needs an explicit
input type at analysis boundaries. Gene annotation adds columns without changing
the feature axis. Fitted backend models live in a separate FittedModel object.

| Namespace | Responsibility |
| --- | --- |
| io | Labeled expression/featureCounts/quantifier imports, native h5ad persistence, exports and static reports |
| annotation | Version stripping, mapping diagnostics, var annotation, explicit transcript count aggregation |
| pp | QC, gene filtering, CPM/TPM/FPKM, transforms, TMM/DESeq2 normalization, exploratory batch adjustment |
| tl | Designs, fitted models, contrasts, PCA/MDS/correlation, ORA/GSEA, scores, regulator activity, descriptive comparisons |
| pl | Matplotlib QC, embedding, expression/correlation heatmaps, volcano/MA and enrichment plots |
| get | Copied expression/results, complete fitted-gene rankings and tested-gene universes |
| datasets | Deterministic count and processed-microarray examples |

Use a formula or labeled design matrix. Formulaic constructs a full-rank matrix
shared by all adapters; this avoids backend-specific factor coding differences.
Named contrasts reconstruct and average counterfactual design rows. This is
explicit marginal averaging when interactions exist. Numeric contrasts allow
specific coefficient combinations. Group comparisons require biological
replicates and saturated/confounded designs are rejected.

Models copy their metadata and fitted gene IDs. Source axis changes invalidate
contrast attachment. Native models remain outside `uns`; serializable results
include provenance, representation, design, filtering, normalization, contrast
direction, backend version, statistic/mean units, and random seeds where used.
Views and backed objects must be explicitly materialized.

## Numerical semantics

CPM uses all input features, or explicitly supplied original library sizes.
No annotation join is required and unmapped genes cannot change the denominator.
TPM/FPKM require aligned positive lengths and an explicit length definition.
Archived lengths are not claimed to be current effective-length annotations.
Fractional quantifier estimates remain `estimated_counts`; native TPM and
sample-specific effective length are preserved. Quantification-aware offsets/
uncertainty inference is not simulated by rounding fractional counts.

The documented bulk workflow uses TPM followed by `log2(TPM + 1)` for PCA,
sample correlation, expression heatmaps, and sample-level gene-set scoring.
Already quantified TPM can be imported directly; the toy count example derives
TPM using explicitly declared synthetic gene lengths. These exploratory steps
select the `logtpm` layer explicitly and preserve the raw counts for DE.
Contrast-level ORA/GSEA instead use the fitted DE results and gene rankings.

TMM relative normalized counts divide by effective library sizes centered on
their geometric mean. PyDESeq2 relative normalized counts use its size factors.
DE always uses raw counts with backend normalization, rather than a derived
CPM/TPM/normalized matrix. Count limma uses edgePython filtering/TMM with
original library denominators and pylimma voom weights. Processed microarrays
use log-expression directly. Exploratory batch correction protects an explicit
biological design and is tagged to reject its direct use for DE.

PyDESeq2 Wald tests retain native statistics and independent-filtering behavior.
LFC shrinkage is explicit and stored separately from the original LFC/SE/test.
Edge QL/LRT use signed sqrt(F/LR) for complete rankings and preserve F/LR.
TREAT does not report a native F/LR statistic, so its `statistic` is missing;
users can explicitly rank by another available column. An LFC labeling
threshold alone does not change a test's null hypothesis.

Results retain all original genes. `prefiltered` means expression filtering;
`untested` means a missing backend p-value; `tested` can still have a missing
adjusted p-value from independent filtering. Missing values remain missing.
Tables retain native columns and canonical columns together, with explicit
`statistic_type` and `mean_expression_type` fields. Result comparisons are
descriptive and do not pool p-values or use majority voting.

## Annotation and enrichment

Mapping returns one row per input, with original ID, normalized lookup ID,
mapped ID, candidate list, and mapped/ambiguous/unmapped status. Ambiguous
matches stay unresolved. Original stable IDs remain the modeling feature keys.
Transcript aggregation is a separately requested operation with an exhaustive
unambiguous mapping. Human/mouse atopos snapshots remain explicitly historical;
current references and other organisms can be supplied as tables.

Functional analysis takes local GMT/dictionaries plus an explicit resource
version and records a content hash. ORA uses the tested-gene background and
separate directional gene selections. Preranked GSEA uses all finite fitted-gene
statistics, including nonsignificant genes. It records mapping losses, duplicate
rank handling, tied ranks, seed, backend options, leading edges and plot profiles.
HDF5 profile keys use generated identifiers instead of pathway names containing
slashes. Sample ssGSEA/GSVA matrices and decoupler 2.x network scores go into
`obsm`; they are distinct from contrast-level enrichment and are not count data.

## Extension scope

Implemented: the P0 matrix workflow plus quantifier import, sample-level scores,
TF/pathway activity, exploratory batch adjustment, phenotype correlation,
multi-result summaries and static report export. Input representations remain
explicit; raw FASTQ alignment/quantification and raw CEL/RMA are upstream.

Deferred P2 modules: coexpression networks, deconvolution, repeated-measures
random effects/time-course specialization, exon/transcript usage inference,
cross-cohort meta-analysis, survival and predictive signatures. Each needs a
validated backend, appropriate input data, and study-specific assumptions.
Native backend features are not assumed to be exposed automatically by mRNA.

## Software compatibility

Python >=3.12 with current AnnData/NumPy/pandas/SciPy APIs. Optional engines use
public interfaces and lazy imports. The limma adapter selects john-mulvey/pylimma
and explicitly rejects the colliding python-limma distribution. Backend versions
are recorded; exact locally tested versions are exported separately. CI tests
both core-only and complete installations across Python 3.12-3.14 and builds
the source/wheel distributions. R fixtures require no R installation at runtime.

Upstream references:

- https://pydeseq2.readthedocs.io/en/stable/
- https://pylimma.readthedocs.io/en/latest/
- https://github.com/pachterlab/edgepython
- https://gseapy.readthedocs.io/en/latest/run.html
- https://decoupler.readthedocs.io/en/stable/notebooks/bulk/rna.html
- https://anndata.readthedocs.io/en/stable/
