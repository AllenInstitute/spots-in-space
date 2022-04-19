library(anndata)
library(ggplot2)
library(umap)

save_folder <- "~/surveyNAS05/scratch/vizgen_download/analyzed_data/"

files <- list.files(save_folder,
                    recursive = TRUE,
                    pattern = ".h5ad",
                    full.names = TRUE)

anndatas <- lapply(files,read_h5ad)

combined_ad <- concat(anndatas)

umap_mfish <- umap(as.data.frame(combined_ad$X),
                   n_neighbors = 25,
                   n_components = 2,
                   metric = "euclidean",
                   min_dist = 0.4,
                   pca = 50)

combined_ad$obsm[['X_umap']] <- umap_mfish

write_h5ad(combined_ad,paste0(cirro_folder,"human_merscope_data_vizgen_download.h5ad"))
