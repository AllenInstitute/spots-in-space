####################### load all data and process #####################

# need to rewrite to save output in separate folder
# need to pull x offset from plate information

library(tidyverse)
library(readxl)
library(ggplot2)
library(gridExtra)
library(Matrix)
library(matrixStats)
library(scrattch.hicat)
library(tibble)
library(patchwork)
library(dplyr)
library(MASS)
library(viridis)
library(dplyr)
library(cowplot)
library(ggthemes)
library(data.table)
library(doMC)
library(anndata)
library(uwot)
library(spdep)


BiocManager::install("BiocNeighbors")

options(stringsAsFactors = FALSE)
options(scipen=999)
options(mc.cores = 20)

gene_name_conversion <- read_xlsx("/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/vizgen_data/InitialGeneList_AIBS.xlsx", sheet=2)

# needs to be update by pulling information directly from sharepoint site
run_tracker <- read_xlsx("/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/MERSCOPE Atlas Run tracker_220214.xlsx", sheet=1)

assembled_experiments <- run_tracker %>% 
  dplyr::filter(`Experiment assembled` == "Yes")

source("/allen/programs/celltypes/workgroups/rnaseqanalysis/yzizhen/joint_analysis/forebrain_new/Vizgen/map_knn.R")

# load rnaseq data (wholebrain)
load("/allen/programs/celltypes/workgroups/rnaseqanalysis/yzizhen/10X_analysis/wholebrain_v3/cl.clean.rda")
load("/allen/programs/celltypes/workgroups/rnaseqanalysis/yzizhen/10X_analysis/wholebrain_v3/cl.means.rda")
load("/allen/programs/celltypes/workgroups/rnaseqanalysis/yzizhen/10X_analysis/wholebrain_v3/anno.df.clean.rda")

anat.df <- anno.df %>% 
  dplyr::select(region_label,region_id,region_color) %>% 
  distinct(region_label,.keep_all = TRUE)

anno_anat <- anno.df %>% 
  group_by(cl, region_label) %>% 
  summarize(size=n()) %>% 
  group_by(cl) %>% 
  summarize(region_label=region_label[which.max(size)])

anno_anat <- merge(anno_anat,
                   anat.df,
                   by = "region_label",
                   all.x = TRUE,
                   ally.y = FALSE)

train.cl.df = cl.df.clean

cl.df_subset <- train.cl.df %>% 
  dplyr::select(
    cluster_id,
    cluster_label,
    cluster_color,
    class_label,
    cl,
    subclass_label,
    subclass_label_full,
    subclass_id,
    subclass_color,
    class_id,
    class_color)

# set filter criteria
min_genes <- 5
min_total_reads <- 30
min_vol <- 100
upper_bound_reads <- 2000
upper_genes_read <- 450

########################## code to process cell by gene table #######################
inputFolder <- "/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/atlas/merfish_output/"
outputFolder <- "/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/atlas/merfish_processed/"


folder <- list.dirs(inputFolder, full.names = FALSE, recursive = FALSE)

folders_to_process <- intersect(folder,assembled_experiments$`Run Name`)

i <- folders_to_process[5]

for (i in folders_to_process) {
  
  file_name <- i
  
  dir.create(file.path(paste0(outputFolder,i)), showWarnings = FALSE)
  
  file_name_parts <- str_split(file_name, pattern = "_")
  experiment <- file_name_parts[[1]][2]
  idx <- which(run_tracker$`Experiment name` == experiment)
  
  section <- substr(experiment,7,8)
  animal <- substr(experiment, 1, 6)
  species <- run_tracker[[idx,2]]
  merscope <- file_name_parts[[1]][3]
  target_atlas_plate <- run_tracker[[idx,3]]
  coverslip_batch <- run_tracker[[idx,4]]
  section_date <- substr(run_tracker[[idx,6]],1,10)
  panel <- run_tracker[[idx,15]]
  codebook_full <- run_tracker[[idx,42]]
  image_rotation <- run_tracker[[idx,62]]
  codebook_parts <- str_split(codebook_full, pattern = "_")
  codebook <- codebook_parts[[1]][3]
  
  
  cbg <- read.csv(paste0(inputFolder,i,"/cell_by_gene.csv"), 
                  header=TRUE, 
                  row.names = 1,
                  check.names = FALSE)
  
  
  metadata <- read.csv(paste0(inputFolder,i,"/cell_metadata.csv"), 
                       header=TRUE,
                       row.names = 1,
                       check.names = FALSE)
  
  metadata <- metadata[match(rownames(cbg),rownames(metadata)),]
  
  blanks <- dplyr::select(cbg,contains("Blank"))
  cbg <- dplyr::select(cbg,-contains("Blank"))
  
  
  coord <- metadata %>% 
    dplyr::select(center_x,center_y) 
  
  colnames(coord) <- c("center_x","center_y")
  
  coord_adj <- Rotation(coord, image_rotation*pi/180)
  coord_adj <- as.data.frame(coord_adj)
  colnames(coord_adj) <- c("corrected_x","corrected_y")
  
  metadata <- merge(metadata,
                    coord_adj,
                    by=0)
  
  metadata$rotation <- image_rotation
  
  metadata <- tibble::column_to_rownames(metadata,var = "Row.names")
  
  ggplot(metadata, aes(x=corrected_x,y=corrected_y))+
    geom_point(size=0.5,stroke=0,shape=19,color="grey") +
    coord_fixed() +
    scale_y_reverse() +
    ggtitle("All cells") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/original_cellxgene.pdf"))
  
  # convert vizgen gene names to Allen gene names
  GeneNames                            = gene_name_conversion$Vizgen.Gene
  includeClas                          = colnames(cbg)
  excludeClas                          = sort(setdiff(GeneNames,includeClas))
  kpSamp                               = !is.element(GeneNames,excludeClas)
  gene_name_conversion_filtered        = gene_name_conversion[kpSamp,]
  
  
  cbg <- cbg[,order(colnames(cbg))]
  gene_name_conversion_ordered <- gene_name_conversion_filtered[order(gene_name_conversion_filtered$Vizgen.Gene), ]
  
  allen_genes <- gene_name_conversion_ordered$Gene
  
  colnames(cbg) <- allen_genes
  
  metadata$genes_detected <- rowSums(cbg!=0)
  metadata$total_reads <- rowSums(cbg)
  
  # calculate sum of neuron specific genes and non-neuronal genes
  # use pipes
  # calculate ratio
  
  neuronal_counts <- cbg %>%
    dplyr::select(Slc17a7,
                  Gad2,
                  Whrn,
                  Th,
                  Erbb4,
                  Cpne9,
                  Pcp4l1,
                  Slc6a1,
                  Zcchc12,
                  Spock3,
                  Neurod2,
                  Grin2a,
                  Nrn1,
                  Adcy2,
                  Mpped1,
                  Chat,
                  Trpc3,
                  Nfib,
                  Nnat,
                  Arx)
  
  non_neuronal_count <- cbg %>%
    dplyr::select(Arhgef19,
                  Atp1b2,
                  Car2,
                  Cgnl1,
                  Cldn5,
                  Ctss,
                  Cxcl12,
                  Enpp2,
                  Gja1,
                  Mal,
                  Mfge8,
                  Mog,
                  Gfap,
                  Pdlim5,
                  Rgs5,
                  Rgs12,
                  S1pr1,
                  Sema6d,
                  Slc1a3,
                  Sox10,
                  Tbx3,
                  Tcf7l2,
                  Timp3,
                  Unc5b)
  
  metadata$neuronal_counts <- rowSums(neuronal_counts)
  metadata$non_neuronal_counts <- rowSums(non_neuronal_count)
  
  metadata$prop_neuronal_gene <- (metadata$neuronal_counts/(metadata$neuronal_counts+metadata$non_neuronal_count))*100
  metadata$prop_non_neuronal_gene <- (metadata$non_neuronal_counts/(metadata$neuronal_counts+metadata$non_neuronal_count))*100
  
  metadata <- metadata %>% 
    mutate(seg_qc = if_else(prop_neuronal_gene < 30 | prop_neuronal_gene > 70, "Good","Bad"))
  
  
  ggplot(metadata,aes(x=prop_neuronal_gene)) +
    geom_histogram(aes(y = ..density..), binwidth = 1) +
    geom_density() +
    ggtitle("Distribution of proprotion of neuronal marker genes") +
    ylab("# of cells") +
    xlab("Proportion of neuronal genes") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/seg_quality.pdf"))
  
  # add columns cell quality score
  
  upper_bound_area <- 3*(median(metadata$volume[metadata$volume>100]))
  
  
  metadata <- metadata %>% 
    mutate(cell_qc = if_else(genes_detected < min_genes | 
                               total_reads < min_total_reads| 
                               volume < min_vol |
                               volume > upper_bound_area |
                               genes_detected > upper_genes_read |
                               total_reads > upper_bound_reads, 
                             "Low","High"))
   
  
  # plot filtered data
  ggplot(subset(metadata,cell_qc %in% "High"), aes(x=corrected_x,y=corrected_y))+
    geom_point(size=0.5,stroke=0,shape=19,color="grey") +
    coord_fixed() +
    ggtitle("All cells") +
    scale_y_reverse() +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/filtered_data.pdf"))
  
  ggplot(subset(metadata,cell_qc %in% "High"),aes(x=prop_neuronal_gene)) +
    geom_histogram(aes(y = ..density..), binwidth = 1) +
    geom_density() +
    ggtitle("Distribution of proprotion of neuronal marker genes") +
    ylab("# of cells") +
    xlab("Proportion of neuronal genes") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/seg_quality_filtered_data.pdf"))
  
  
  ggplot(metadata,aes(x=volume)) +
    geom_histogram(bins=500) +
    ggtitle("Distribution of cell area") +
    ylab("# of cells") +
    xlab("cell volume [um_3]") +
    geom_vline(xintercept = min_vol, color="red") +
    geom_vline(xintercept = upper_bound_area, color="blue") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/cell_volume_distr.pdf"))
  
  
  ggplot(metadata,aes(x=genes_detected)) +
    geom_histogram(binwidth = 1) +
    ggtitle("Distribution of genes deteced per cell") +
    ylab("# of cells") +
    xlab("Genes detected per cell") +
    geom_vline(xintercept = min_genes, color="red") +
    geom_vline(xintercept = upper_genes_read, color="blue") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/gene_count_distr.pdf"))
  
  
  ggplot(metadata,aes(x=total_reads)) +
    geom_histogram(binwidth = 1) +
    ggtitle("Distribution of mRNA spots deteced per cell") +
    ylab("# of cells") +
    xlab("mRNA spots detected per cell") +
    geom_vline(xintercept = min_total_reads, color="red") +
    geom_vline(xintercept = upper_bound_reads, color="blue") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/spot_count_distr.pdf"))
  
  metadata$section <- section
  metadata$animal <- animal
  metadata$species <- species
  metadata$merscope <- merscope
  metadata$target_atlas_plate <- target_atlas_plate
  metadata$coverslip_batch <- coverslip_batch
  metadata$section_date <- section_date
  metadata$panel <- panel
  metadata$codebook <- codebook
  metadata$min_genes <- min_genes
  metadata$min_total_reads <- min_total_reads
  metadata$min_vol <- min_vol
  metadata$upper_bound_area <- upper_bound_area
  metadata$upper_bound_reads <- upper_bound_reads
  metadata$upper_genes_read <- upper_genes_read
  
  
  cbg <- as.matrix(cbg)
  cbg <- Matrix(cbg, sparse = TRUE)
  
  cbg_cpum <- (cbg / metadata$volume*1000)
  cbg_norm <- log2(cbg_cpum+1)
  
  # start mapping
  vizgen.dat <- t(cbg_cpum)
  vizgen.dat <- log2(vizgen.dat+1)
  
  genes <- rownames(vizgen.dat)
  useGenes <- intersect(rownames(cl.means),genes)
  
  train.cl.dat = cl.means[useGenes,]
  vizgen.dat = vizgen.dat[useGenes,]
  
  index.bs = build_train_index_bs(train.cl.dat, method="cor",fn = "fb.index")
  map.result = map_cells_knn_bs(vizgen.dat, train.index.bs=index.bs, method="cor")
  best.map.df = map.result$best.map.df
  best.map.df$best.cl <- as.integer(best.map.df$best.cl)
  cl.anno = best.map.df %>% left_join(train.cl.df[,c("cl","subclass_id","class_id")],by=c("best.cl"="cl"))
  
  cl.list = with(cl.anno, list(cl=setNames(best.cl, sample_id),subclass=setNames(subclass_id, sample_id),subclass=setNames(class_id, sample_id)))
  z.score=z_score(cl.list, val=with(best.map.df, setNames(avg.cor,sample_id)), min.sample=100)
  cl.anno$z.score = z.score[cl.anno$sample_id]
  
  anno_mfish <- merge(metadata,
                      cl.anno,
                      by.x = 0,
                      by.y="sample_id"
  )
  
  
  anno_mfish <- merge(anno_mfish,
                      cl.df_subset,
                      by.x = "best.cl",
                      by.y="cl",
                      all.x = TRUE,
                      ally.y = FALSE
  )
  
  anno_mfish <- merge(anno_mfish,
                      anno_anat,
                      by.x = "best.cl",
                      by.y="cl",
                      all.x = TRUE,
                      ally.y = FALSE
  )
  
  anno_mfish <- column_to_rownames(anno_mfish, var = "Row.names")
  anno_mfish <- anno_mfish[match(rownames(cbg_cpum),rownames(anno_mfish)),]
  
  #create some basic overview images
  # define color scheme
  class_colors <- anno_mfish[c("class_label","class_color")]
  class_colors <- unique(class_colors)
  aibs_color_scheme_class <- class_colors$class_color 
  names(aibs_color_scheme_class) <- class_colors$class_label
  
  subclass_colors <- anno_mfish[c("subclass_label","subclass_color")]
  subclass_colors <- unique(subclass_colors)
  aibs_color_scheme_subclass <- subclass_colors$subclass_color 
  names(aibs_color_scheme_subclass) <- subclass_colors$subclass_label
  
  cluster_colors <- anno_mfish[c("cluster_label","cluster_color")]
  cluster_colors <- unique(cluster_colors)
  aibs_color_scheme_cluster <- cluster_colors$cluster_color 
  names(aibs_color_scheme_cluster) <- cluster_colors$cluster_label
  
  ggplot(subset(anno_mfish,cell_qc %in% "High"),aes(x=avg.cor)) +
    geom_histogram(aes(y = ..density..), binwidth = .01) +
    geom_density() +
    ggtitle("Distribution of cluster correlation") +
    ylab("Cell density") +
    xlab("Correlation coefficient") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/corr_coeff.pdf"))
  
  ggplot(subset(anno_mfish,cell_qc %in% "High"),aes(x=avg.cor,fill=class_label)) +
    geom_histogram(aes(y = ..density..), binwidth = .01) +
    geom_density() +
    ggtitle("Distribution of cluster correlation") +
    ylab("Cell density") +
    xlab("Correlation coefficient") +
    facet_wrap(~class_label) +
    scale_fill_manual(values=aibs_color_scheme_class) +
    guides(color = "none", alpha = "none") +
    theme_minimal()
  
  ggsave2(paste0(outputFolder,i,"/corr_coeff_by_class.pdf"))
  
  # save the following files, cbg, blank, cbg_cpum, cbg_norm, metadata_processed, 
  # and everything wrapped up in a hdf5 (anndata)
  
  # save individual files as csv
  write.csv(anno_mfish,paste0(outputFolder,i,"/metadata_processed.csv"),row.names = TRUE)
  write.csv(as.matrix(cbg_cpum),paste0(outputFolder,i,"/cbg_cpum.csv"),row.names = TRUE)
  write.csv(as.matrix(cbg_norm),paste0(outputFolder,i,"/cbg_norm.csv"),row.names = TRUE)
  write.csv(blanks,paste0(outputFolder,i,"/blanks.csv"),row.names = TRUE)
  # save all files as rda
  save(anno_mfish,cbg_cpum,cbg_norm,blanks,file=paste0(outputFolder,i,"/processed_files.rda")) 
  

  
  # XY_filename
  coordinates <- anno_mfish %>% 
    dplyr::filter(cell_qc == "High") %>% 
    dplyr::select(corrected_x,corrected_y)
  
  offset_x <- target_atlas_plate - 47
  
  coordinates$corrected_y = -(coordinates$corrected_y - mean(coordinates$corrected_y))
  coordinates$corrected_x = coordinates$corrected_x - min(coordinates$corrected_x)
  coordinates$corrected_x = coordinates$corrected_x + (offset_x*10000)
  counter = counter+1
  
  to_keep <- intersect(rownames(cbg),rownames(subset(anno_mfish,cell_qc %in% "High")))
  cbg_filtered <- cbg_norm[to_keep,]
  cbg_filtered <- as.matrix(cbg_filtered)
  
  
  # prepare data for anndata format 
  uns <- c("rotation","animal","species","merscope","target_atlas_plate","coverslip_batch","section_date","panel","codebook","min_genes","min_total_reads","min_vol","upper_bound_area","upper_bound_reads","upper_genes_read")
  anno_mfish$subclass_label <- paste(anno_mfish$subclass_id,anno_mfish$subclass_label, sep="_")
  
  anno_mfish_subset <- anno_mfish %>% 
    dplyr::filter(cell_qc == "High") %>% 
    dplyr::select(class_label,
                  subclass_label,
                  cluster_label,
                  region_label,
                  animal,
                  section,
                  merscope,
                  target_atlas_plate,
                  seg_qc,
                  avg.cor,
                  genes_detected,
                  total_reads,
                  prob)
                                     
  
  umap_mfish <- umap(cbg_filtered,
                     n_neighbors = 25,
                     n_components = 2,
                     metric = "euclidean",
                     min_dist = 0.4)
  
  blanks_filtered <- log2(blanks[to_keep,]+1)
  
  ad <- AnnData(
    X = cbg_filtered,
    obs = anno_mfish_subset,
    obsm = list(
      blanks = as.matrix(blanks_filtered),
      spatial = as.matrix(coordinates),
      X_umap = umap_mfish
    ),
    uns = list(
      species = species,
      merscope = merscope,
      gene_panel = gene_panel,
      min_genes = min_genes,
      min_total_reads = min_total_reads,
      min_vol = min_vol,
      upper_bound_area = upper_bound_area,
      upper_bound_reads = upper_bound_reads,
      upper_genes_read = upper_genes_read
    )
  )
  
  write_h5ad(ad,paste0(inputFolder,"/processed/",experiment,".h5ad"))
}

cirro_folder <- "/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/for_cirro/"

files <- list.files("/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/atlas/", 
                    recursive = TRUE, 
                    pattern = ".h5ad",
                    full.names = TRUE)

anndatas <- lapply(files,read_h5ad)

combined_ad <- concat(
  anndatas)

umap_mfish <- umap(as.data.frame(combined_ad$X),
                   n_neighbors = 25,
                   n_components = 2,
                   metric = "euclidean",
                   min_dist = 0.4)

combined_ad$obsm[['X_umap']] <- umap_mfish  

write_h5ad(combined_ad,paste0(cirro_folder,"atlas_brain_1.h5ad"))









