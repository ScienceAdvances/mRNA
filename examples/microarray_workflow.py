"""Processed microarray expression through native pylimma and GSEA."""

import mRNA as mr

adata = mr.datasets.toy_microarray()
model = mr.tl.fit(
    adata, method="limma", design="~condition", input_type="log_expression", layer=None
)
de = mr.tl.contrast(model, contrast=("condition", "treated", "control"))
print(de[["gene", "log2_fold_change", "adj_p_value", "change"]].head())
