# Independent references using the upstream versions ported by the Python tools.
# R_LIBS_USER=/tmp/mRNA-r-library Rscript tests/fixtures/generate_reference.R tests/fixtures
library(limma)
library(edgeR)
args <- commandArgs(trailingOnly=TRUE)
directory <- if (length(args)) args[[1]] else "tests/fixtures"
counts <- t(as.matrix(read.csv(file.path(directory,"counts.csv"),row.names=1,check.names=FALSE)))
expr <- t(as.matrix(read.csv(file.path(directory,"microarray.csv"),row.names=1,check.names=FALSE)))
design <- as.matrix(read.csv(file.path(directory,"design.csv"),row.names=1,check.names=FALSE))
contrast <- as.matrix(read.csv(file.path(directory,"contrast.csv"),row.names=1,check.names=FALSE))
save_table <- function(table, filename) {
    table$gene <- rownames(table)
    write.csv(table,file.path(directory,filename),row.names=FALSE)
}
micro <- eBayes(contrasts.fit(lmFit(expr,design),contrast))
save_table(topTable(micro,coef=1,number=Inf,sort.by="none"),"limma_microarray_reference.csv")
y <- DGEList(counts)
keep <- filterByExpr(y,design=design)
y <- y[keep,,keep.lib.sizes=TRUE]
y <- calcNormFactors(y)
v <- voom(y,design)
vf <- eBayes(contrasts.fit(lmFit(v,design),contrast))
save_table(topTable(vf,coef=1,number=Inf,sort.by="none"),"limma_voom_reference.csv")
y <- estimateDisp(y,design)
fit <- glmQLFit(y,design)
test <- glmQLFTest(fit,contrast=contrast[,1])
save_table(topTags(test,n=Inf,sort.by="none")$table,"edge_ql_reference.csv")
write.csv(data.frame(sample=rownames(y$samples),lib_size=y$samples$lib.size,norm_factor=y$samples$norm.factors),
          file.path(directory,"tmm_reference.csv"),row.names=FALSE)
writeLines(c(paste("R",getRversion()),paste("limma",packageVersion("limma")),paste("edgeR",packageVersion("edgeR"))),
           file.path(directory,"reference_versions.txt"))
