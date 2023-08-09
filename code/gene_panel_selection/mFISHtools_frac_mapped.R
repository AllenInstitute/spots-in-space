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
gene_list <- rownames(ad$var)

# annotations <- feather(paste0(input_folder,"anno.feather"))
# dataIn      <- feather(paste0(input_folder,"data.feather"))

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

frac_mapped <- fractionCorrectWithGenes(gene_list, normDat, medianExpr, cl, plot=FALSE, return=TRUE)

gene_df <- data.frame(
        gene = gene_list,
        frac_mapped = frac_mapped
    )

write.csv(gene_df, paste0(out_path, "/frac_mapped.csv"))
print(paste0(out_path, "/frac_mapped.csv"))