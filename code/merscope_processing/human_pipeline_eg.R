####################### load all data and process #####################

suppressPackageStartupMessages({
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
  library(feather)
})

options(stringsAsFactors = FALSE)
options(scipen=999)
options(mc.cores = 20)

########################## code to process cell by gene table #######################

inputFolder <- "~/surveyNAS05/scratch/vizgen_download/analyzed_data/"
run_tracker <- read_xlsx("~/surveyNAS05/scratch/MERSCOPE Run tracker_211102.xlsx", sheet=1)

run_tracker_human <- run_tracker %>% 
  dplyr::filter(Species == "Human") %>% 
  drop_na(`Experiment name`) 

run_tracker_human$`Experiment name` <- str_replace_all(run_tracker_human$`Experiment name`,"mtg", "MTG") 
run_tracker_human$`Experiment name` <- str_replace_all(run_tracker_human$`Experiment name`,"cx", "Cx") 
run_tracker_human$`Experiment name` <- str_replace_all(run_tracker_human$`Experiment name`,"CX", "Cx") 

min_genes <- 3
min_total_reads <- 30
min_vol <- 100

upper_bound_reads <- 4000
upper_genes_read <- 130

gene_panel <- "VZG167a"

# load files for mapping
source("~/surveyNAS05/scratch/map_knn.R")

#mappingFolder <- "/allen/programs/celltypes/workgroups/rnaseqanalysis/Nik/Analyses_for_great_apes_paper/Step9_final_taxonomies_after_manual_subclass_confirmation/individual_species/human/"    

# read count matrix for macaque rnaseq data
mat_rnaseq <- readRDS(paste0("~/surveyNAS05/scratch/human_all_raw_UMI_expression_matrix.RDS"))
# read in annotation
metadata_rnaseq <- readRDS(paste0("~/surveyNAS05/scratch/Master_metadata_for_plots_and_sharing_12_16_21.RDS"))
#filter out smart-seq data from metadata
metadata_rnaseq <- metadata_rnaseq %>% 
  filter(species_tech=="human_10x")
# remove smart-seq cells from count matrix
mat_rnaseq_subset <- mat_rnaseq[,metadata_rnaseq$sample_id]


# generate anno file
train.cl.df <- metadata_rnaseq %>% 
  dplyr::select(cluster,
                cluster_color,
                subclass,
                subclass_color,
                neighborhood,
                cross_species_cluster,
                cross_species_cluster_color,
                class) %>% 
  distinct(cluster, .keep_all = TRUE)

folder <- list.dirs("~/surveyNAS05/scratch/vizgen_download/analyzed_data", full.names = TRUE, recursive = FALSE)

counter = 0

for (i in folder) {
  inputFolder <- i
  file_name <- basename(inputFolder)
  #below are changes to accommodate vizgen processed data and folder structure
  if (file_name != 'Assembled') {
    #file_name_parts <- str_split(file_name, pattern = "_")
    # experiment <- file_name_parts[[1]][2]
    # experiment <- str_replace_all(experiment,"mtg", "MTG") 
    # experiment <- str_replace_all(experiment,"cx", "Cx") 
    # experiment <- str_replace_all(experiment,"CX", "Cx") 
    # idx <- which(run_tracker_human$`Experiment name` == experiment)
    # species <- run_tracker_human[[idx,2]]
    # section <- run_tracker_human[[idx,1]]
    # merscope <- file_name_parts[[1]][3]
    # collection_year <- substring(file_name_parts[[1]][2],2,3)
    # source <- substring(file_name_parts[[1]][2],4,5)
    experiment <- file_name
    idx <- counter
    species <- 'human'
    section <- 'vizgen_run'
    merscope <- 'vizgen_run'
    collection_year <- '2022'
    source <- 'vizgen_run'
    #above are changes to accommodate vizgen processed data and folder structure
    
    cbg <- read.csv(paste0(inputFolder,"/region_0/cell_by_gene.csv"), 
                    header=TRUE, 
                    row.names = 1,
                    check.names = FALSE)
    
    
    metadata <- read.csv(paste0(inputFolder,"/region_0/cell_metadata.csv"), 
                         header=TRUE,
                         row.names = 1,
                         check.names = FALSE)
    
    metadata <- metadata[match(rownames(cbg),rownames(metadata)),]
    
    blanks <- dplyr::select(cbg,contains("Blank"))
    cbg <- dplyr::select(cbg,-contains("Blank"))
    
    metadata$genes_detected <- rowSums(cbg!=0)
    metadata$total_reads <- rowSums(cbg)
    
    dir.create(file.path(paste0(inputFolder,"/processed/")), showWarnings = FALSE)
    dir.create(file.path(paste0(inputFolder,"/plots/")), showWarnings = FALSE)
    
    # calculate sum of neuron specific genes and non-neuronal genes
    
    neuronal_counts <- cbg %>%
      dplyr::select(SATB2,
                    CUX2,
                    CDH6,
                    FEZF2,
                    PDGFD,
                    LAMP5,
                    VIP,
                    LHX6,
                    PRRT4,
                    ANK1,
                    SLC32A1,
                    BTBD11,
                    GRIP2,
                    DLX1,
                    GAD2,
                    DCN,
                    CALB1,
                    TH,
                    CBLN2,
                    NXPH2,
                    SMYD1,
                    SULF1,
                    NDNF,
                    SLIT3,
                    KCNIP4,
                    KCNMB2,
                    SEMA3E,
                    TLL1,
                    ROBO2,
                    TSHZ2,
                    NRG1,
                    SV2C,
                    FRMPD4,
                    PALMD,
                    DCLK1,
                    FGF13,
                    ADAMTS3,
                    CLSTN2,
                    GALNTL6,
                    GRM8,
                    GRIN3A,
                    SEMA6D,
                    HS3ST2,
                    ZMAT4,
                    GRIP1,
                    DACH1,
                    ADAMTSL1,
                    RBFOX3,
                    THEMIS,
                    HCN1,
                    GRIN2A)
    
    non_neuronal_count <- cbg %>%
      dplyr::select(CD22,
                    MOG,
                    ETNPPL,
                    ID3,
                    FBXL7)
    
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

    ggsave2(paste0(inputFolder,"/plots/seg_quality.pdf"))
    
    metadata$species <- species
    metadata$collection_year <- collection_year
    metadata$source <- source
    metadata$merscope <- merscope
    metadata$gene_panel <- gene_panel
    metadata$section <- section
    
    ggplot(metadata, aes(x=center_x,y=center_y))+
      geom_point(size=0.1,stroke=0,shape=19,color="grey") +
      coord_fixed() +
      scale_y_reverse() +
      ggtitle("All cells") +
      theme_minimal()

    ggsave2(paste0(inputFolder,"/plots/original.pdf"))

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
    ggplot(subset(metadata,cell_qc %in% "High"), aes(x=center_x,y=center_y))+
      geom_point(size=0.5,stroke=0,shape=19,color="grey") +
      coord_fixed() +
      ggtitle("All cells") +
      scale_y_reverse() +
      theme_minimal()

    ggsave2(paste0(inputFolder,"/plots/filtered_data.pdf"))

    ggplot(subset(metadata,cell_qc %in% "High"),aes(x=prop_neuronal_gene)) +
      geom_histogram(aes(y = ..density..), binwidth = 1) +
      geom_density() +
      ggtitle("Distribution of proprotion of neuronal marker genes") +
      ylab("# of cells") +
      xlab("Proportion of neuronal genes") +
      theme_minimal()

    ggsave2(paste0(inputFolder,"/plots/seg_quality_filtered_data.pdf"))


    ggplot(metadata,aes(x=volume)) +
      geom_histogram(bins=500) +
      ggtitle("Distribution of cell area") +
      ylab("# of cells") +
      xlab("cell volume [um_3]") +
      geom_vline(xintercept = min_vol, color="red") +
      geom_vline(xintercept = upper_bound_area, color="blue") +
      theme_minimal()

    ggsave2(paste0(inputFolder,"/plots/cell_volume_distr.pdf"))


    ggplot(metadata,aes(x=genes_detected)) +
      geom_histogram(binwidth = 1) +
      ggtitle("Distribution of genes deteced per cell") +
      ylab("# of cells") +
      xlab("Genes detected per cell") +
      geom_vline(xintercept = min_genes, color="red") +
      geom_vline(xintercept = upper_genes_read, color="blue") +
      theme_minimal()

    ggsave2(paste0(inputFolder,"/plots/gene_count_distr.pdf"))


    ggplot(metadata,aes(x=total_reads)) +
      geom_histogram(binwidth = 1) +
      ggtitle("Distribution of mRNA spots deteced per cell") +
      ylab("# of cells") +
      xlab("mRNA spots detected per cell") +
      geom_vline(xintercept = min_total_reads, color="red") +
      geom_vline(xintercept = upper_bound_reads, color="blue") +
      theme_minimal()

    ggsave2(paste0(inputFolder,"/plots/spot_count_distr.pdf"))
    
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
    useGenes <- intersect(rownames(mat_rnaseq_subset),genes)

    mat_rnaseq_subset <- mat_rnaseq_subset[rownames(mat_rnaseq_subset) %in% useGenes, ]
    norm_dat_rnaseq <- log2(mat_rnaseq_subset+1)

    cl          = metadata_rnaseq$cluster
    names(cl)   = metadata_rnaseq$sample_id
     
    cl.means <- get_cl_means(norm_dat_rnaseq,cl)

    train.cl.dat = cl.means[useGenes,]
    vizgen.dat = vizgen.dat[useGenes,]

    index.bs = build_train_index_bs(train.cl.dat, method="Annoy.Cosine",fn = "fb.index")
    map.result = map_cells_knn_bs(vizgen.dat, train.index.bs=index.bs, method="Annoy.Cosine", mc.cores=10)
    best.map.df = map.result$best.map.df
    cl.anno = best.map.df %>% left_join(train.cl.df,by=c("best.cl"="cluster"))
    
    cl.list = with(cl.anno, list(cl=setNames(best.cl, sample_id),subclass=setNames(subclass, sample_id),subclass=setNames(best.cl, sample_id)))
    z.score=z_score(cl.list, val=with(best.map.df, setNames(avg.cor,sample_id)), min.sample=100)
    cl.anno$z.score = z.score[cl.anno$sample_id]

    anno_mfish <- merge(metadata,
                       cl.anno,
                       by.x = 0,
                       by.y="sample_id"
                       )
    
    anno_mfish <- column_to_rownames(anno_mfish, var = "Row.names")
    anno_mfish <- anno_mfish[match(rownames(cbg_cpum),rownames(anno_mfish)),]
    
    ggplot(subset(anno_mfish,cell_qc %in% "High"),aes(x=avg.cor)) +
      geom_histogram(aes(y = ..density..), binwidth = .01) +
      geom_density() +
      ggtitle("Distribution of cluster correlation") +
      ylab("Cell density") +
      xlab("Correlation coefficient") +
      theme_minimal()
    
    ggsave2(paste0(inputFolder,"/plots/corr_coeff.pdf"))
    
    ggplot(subset(anno_mfish,cell_qc %in% "High"),aes(x=avg.cor,fill=neighborhood)) +
      geom_histogram(aes(y = ..density..), binwidth = .01) +
      geom_density() +
      ggtitle("Distribution of cluster correlation") +
      ylab("Cell density") +
      xlab("Correlation coefficient") +
      facet_wrap(~neighborhood) +
      #scale_fill_manual(values=aibs_color_scheme_class) +
      guides(color = "none", alpha = "none") +
      theme_minimal()
    
    ggsave2(paste0(inputFolder,"/plots/corr_coeff_by_class.pdf"))
    
    write.csv(anno_mfish,paste0(inputFolder,"/processed/metadata_processed.csv"),row.names = TRUE)
    write.csv(as.matrix(cbg_cpum),paste0(inputFolder,"/processed/cbg_cpum.csv"),row.names = TRUE)
    write.csv(as.matrix(cbg_norm),paste0(inputFolder,"/processed/cbg_norm.csv"),row.names = TRUE)
    write.csv(blanks,paste0(inputFolder,"/processed/blanks.csv"),row.names = TRUE)
    
    save(anno_mfish,cbg_cpum,cbg_norm,blanks,file=paste0(inputFolder,"/processed/processed_files.rda"))
    
    to_keep <- intersect(rownames(cbg),rownames(subset(anno_mfish,cell_qc %in% "High")))
    cbg_filtered <- cbg_norm[to_keep,]
    cbg_filtered <- as.matrix(cbg_filtered)
    
    coordinates <- anno_mfish %>%
      dplyr::filter(cell_qc == "High") %>%
      dplyr::select(center_x,center_y)
    
    coordinates_cirro <- anno_mfish %>%
      dplyr::filter(cell_qc == "High") %>%
      dplyr::select(center_x,center_y)
    
    coordinates_cirro$center_y = -(coordinates_cirro$center_y - mean(coordinates_cirro$center_y))
    coordinates_cirro$center_x = coordinates_cirro$center_x - min(coordinates_cirro$center_x)
    coordinates_cirro$center_x = coordinates_cirro$center_x + (counter*10000)
    counter = counter+1
    
    colnames(anno_mfish)[29] <- "cluster"
    # prepare data for anndata format
    uns <- c("species","merscope","gene_panel","min_genes","min_total_reads","min_vol","upper_bound_area","upper_bound_reads","upper_genes_read")
    anno_mfish_subset <- anno_mfish %>%
      dplyr::filter(cell_qc == "High") %>%
      dplyr::select(cluster,cross_species_cluster,subclass, neighborhood,class,merscope,avg.cor,genes_detected,total_reads,prob,volume)
    umap_mfish <- umap(cbg_filtered,n_neighbors = 25,n_components = 2,metric = "euclidean",min_dist = 0.4,pca = 50)
    
    blanks_filtered <- log2(blanks[to_keep,]+1)
    
    ad <- AnnData(
      X = cbg_filtered,
      obs = anno_mfish_subset,
      obsm = list(
        blanks = as.matrix(blanks_filtered),
        spatial = as.matrix(coordinates),
        spatial_cirro = as.matrix(coordinates_cirro),
        X_umap = umap_mfish
        ),
      uns = list(
        species = species,
        section = section,
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
    write_h5ad(ad,paste0(inputFolder,"/processed/",file_name,".h5ad"))
  }
}

