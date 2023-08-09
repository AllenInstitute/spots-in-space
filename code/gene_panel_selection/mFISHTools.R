
suppressPackageStartupMessages({
  library(mfishtools)    # This library!
  library(matrixStats)   # For rowMedians function, which is fast
  library(feather)       # NEED THIS LIBRARY TOO for reading in BG taxonomy
  library(edgeR)         # NEED THIS LIBRARY TOO for calculating cpm/rpkm
  library(anndata)
  library(optparse)
})
options(stringsAsFactors = FALSE)  # IMPORTANT
print("Libraries loaded.")

inputs = list(make_option("--dat_path", type="character", default="", 
              help="path to data", metavar="character"))

opt_parser = OptionParser(option_list=inputs)
opts = parse_args(opt_parser)

out_path <- opts$dat_path
print(out_path)

## Read in the data

print('reading in data...')
ad <- read_h5ad(paste0(out_path, "/mfishtools_ad_temp.h5ad"))

dataIn <- ad$X
annotations <- ad$obs

# Move cluster info up here
cl          <- annotations$cluster_label
names(cl)   <- rownames(annotations)

# Subsample the data to speed up code
subSamp     <- subsampleCells(cl,100)
annotations <- annotations[subSamp,]
dataIn      <- dataIn[subSamp,] # This is VERY slow and might crash if you don't have enough memory, but hopefully not
cl          <- cl[subSamp]

# Convert to data matrix
normDat <- t(as.matrix(dataIn[,colnames(dataIn)!="sample_id"]))
colnames(normDat) <- rownames(annotations)

exprThresh <- 1
medianExpr <- do.call("cbind", tapply(names(cl), cl, function(x) rowMedians(normDat[,x]))) 
propExpr   <- do.call("cbind", tapply(names(cl), cl, function(x) rowMeans(normDat[,x]>exprThresh)))
rownames(medianExpr) <- rownames(propExpr) <- genes <- rownames(normDat)

if (ad$uns$filter_panel$numBinaryGenes > 0) {
    print('filtering genes...')
    runGenes <- filterPanelGenes(
      summaryExpr = 2^medianExpr-1,  # medians (could also try means); We enter linear values to match the linear limits below
      propExpr    = propExpr,    # proportions
      startingGenes  = ad$uns$filter_panel$startingGenes,  # Starting genes (from above)
      numBinaryGenes = ad$uns$filter_panel$numBinaryGenes,      # Number of binary genes (explained below)
      minOn     = 10,   # Minimum required expression in highest expressing cell type
      maxOn     = 500,  # Maximum allowed expression
    ) 
} else {
    runGenes <- rownames(ad$var)
}

corDist         <- function(x) return(as.dist(1-cor(x)))
clusterDistance <- as.matrix(corDist(medianExpr[runGenes,]))
print(dim(clusterDistance))

print('building panel...')
fishPanel <- buildMappingBasedMarkerPanel(
  mapDat        = normDat[runGenes,],     # Data for optimization
  medianDat     = medianExpr[runGenes,], # Median expression levels of relevant genes in relevant clusters
  clustersF     = cl,                   # Vector of cluster assignments
  panelSize     = ad$uns$build_panel$panelSize,                           # Final panel size
  currentPanel  = ad$uns$filter_panel$startingGenes,            # Starting gene panel
  subSamp       = 50,                           # Maximum number of cells per cluster to include in analysis (20-50 is usually best)
  optimize      = "CorrelationDistance",        # CorrelationDistance maximizes the cluster distance as described
  clusterDistance = clusterDistance,            # Cluster distance matrix
  )
print(fishPanel)

frac_mapped <- fractionCorrectWithGenes(fishPanel, normDat, medianExpr[runGenes,], cl, plot=FALSE, return=TRUE)

gene_df <- data.frame(
        gene = fishPanel,
        frac_mapped = frac_mapped
    )

write.csv(gene_df, paste0(out_path, "/gene_list.csv"))
print(paste0(out_path, "/gene_list.csv"))