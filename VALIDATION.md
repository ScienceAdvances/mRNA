# Validation

Validation date: 2026-10-02. Statistical backends run on real example matrices;
the integration tests do not mock backend computations.

## Python and dependency coverage

Local complete environments: Python 3.12.13, 3.13.15, and 3.14.7. Exact dependency
versions are listed in `requirements-tested.txt`. Core-only wheel installation
is verified separately. The CI configuration repeats core/complete installations
on Python 3.12, 3.13, and 3.14.

Each complete local environment passes the same test suite, including the
independent R reference checks. A separate core-only installation of the built
wheel verifies that optional engines are not required for importing mRNA via `import mRNA as mr`,
preprocessing, annotation, and persistence.

The suite tests independent expected unit-conversion values, original library
denominators after gene selection, immutable counts, mapping ambiguity/loss,
stable feature axes, source metadata alignment, invalid designs, paired effects,
reversed contrast signs/p-values, shrinkage retention, directional scores,
ORA against an independent hypergeometric calculation, leading-edge data,
plot rendering and AnnData serialization.

## R fixtures

Inputs were generated with seed 19 on 12 biological sample profiles and 200
genes. The design includes batch and condition. Fixture generation uses an
isolated temporary R library and the upstream versions targeted by the Python
ports: **R 4.6.1, limma 3.66.0, edgeR 4.8.2**. These are reference versions,
not a claim that the Python tools implement every feature of current R releases.

| Workflow | Max absolute LFC difference | Max absolute raw p difference | Max absolute adjusted p difference |
| --- | ---: | ---: | ---: |
| Processed microarray / limma | 5.6e-15 | 1.1e-14 | 1.2e-14 |
| TMM + voom / limma | 1.2e-11 | 4.4e-11 | 4.3e-11 |
| TMM + QL / edgePython | 7.8e-5 | 1.2e-4 | 3.1e-4 |

TMM normalization factors agree to an absolute difference below 4.1e-15.
The edgePython QL reference calls the same 33 significant genes at FDR <0.05
in this example. Its maximum relative p/FDR difference is about 0.8%; numerical
agreement is therefore not bitwise equivalence. Borderline results on other
datasets may differ. Tests use documented absolute tolerances rather than
silently assuming exact parity.

PyDESeq2 is checked using real model fitting, paired/multi-factor design
construction, positive/negative known effects, reversed contrasts, missing
statistics, count preservation, and supported coefficient shrinkage. This
initial validation does **not** include a separately generated R DESeq2 fixture;
do not infer universal R parity from these checks.

To regenerate the independent R references after installing the listed R
packages into a chosen library:

```sh
python tests/fixtures/generate_inputs.py
R_LIBS_USER=/path/to/reference/library Rscript tests/fixtures/generate_reference.R tests/fixtures
```

Ordinary pytest runs read the committed CSV fixtures and do not require R.

## Known practical limits

- The toy count data may trigger PyDESeq2's documented switch from a parametric
  to a mean dispersion trend; the warning is retained and not suppressed.
- The bundled atopos human/mouse annotation tables are historical snapshots
  with undocumented length derivation. Supply current references/lengths for
  the appropriate assembly and quantification definition.
- Fractional quantifier estimates are imported faithfully but are not rounded
  or advertised as a complete tximport-equivalent DE workflow.
- No-replicate inference, random effects, raw CEL/FASTQ processing, and the P2
  specialized modules are not implemented in this release.
