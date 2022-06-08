library(feather)
counts <- read_feather("/home/imaging_mfish/surveyNAS05/scratch/human/tax_feathers_6122/MTG/count_n.feather")
sums   <- read_feather("/home/imaging_mfish/surveyNAS05/scratch/human/tax_feathers_6122/MTG/sums.feather")
dend   <- readRDS("/home/imaging_mfish/surveyNAS05/scratch/human/tax_feathers_6122/MTG/dend.RData") 
mean_counts  <- as.matrix(as.data.frame(sums[,2:(length(labels(dend))+1)]))
rownames(mean_counts) <- sums$gene
colnames(mean_counts) <- labels(dend)
mean_counts <- t(t(mean_counts)/counts$n_cells)
library(scrattch.hicat)
means_logCPM <- logCPM(mean_counts)
saveRDS(means_logCPM, '/home/imaging_mfish/surveyNAS05/scratch/human/tax_feathers_6122/MTG/MTG_tax_6122.RDS')
#THIS IS ALREADY THE CLUSTER MEANS!!