library(anndata)
library(uwot)

save_folder <- "/home/imaging_mfish/surveyNAS05/scratch/"

files <- list('/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202112101342_H1930001Cx46MTG0202007205_VMSC01001/processed/202112101342_H1930001Cx46MTG0202007205_VMSC01001.h5ad', 
              '/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202112021529_H1930001Cx46MTG202007203_VMSC01001/processed/202112021529_H1930001Cx46MTG202007203_VMSC01001.h5ad', 
              '/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202112021532_H1930001Cx46MTG202007204_vmsc00401/processed/202112021532_H1930001Cx46MTG202007204_vmsc00401.h5ad', 
              '/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202201271619_H1930002Cx46MTG202007102_VMSC01001/processed/202201271619_H1930002Cx46MTG202007102_VMSC01001.h5ad', 
              '/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202201281113_H1930002Cx46MTG202007104_vmsc00401/processed/202201281113_H1930002Cx46MTG202007104_vmsc00401.h5ad', 
              '/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202201271624_H1930002Cx46MTG202007105_vmsc00401/processed/202201271624_H1930002Cx46MTG202007105_vmsc00401.h5ad', 
              '/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202201311715_H2133013Cx24MTG02007106_VMSC01001/processed/202201311715_H2133013Cx24MTG02007106_VMSC01001.h5ad',
              '/home/imaging_mfish/MERSCOPENAS02_data/human/merfish_output/202201281456_H2133013Cx24MTG02007105_VMSC01001/processed/202201281456_H2133013Cx24MTG02007105_VMSC01001.h5ad', 
              '/home/imaging_mfish/surveyNAS05/scratch/vizgen_download/analyzed_data/HuBrain_C30_VS47_VZG167a_5hrPhoto_V3_JH_3-15-2022/processed/HuBrain_C30_VS47_VZG167a_5hrPhoto_V3_JH_3-15-2022.h5ad',
              '/home/imaging_mfish/surveyNAS05/scratch/vizgen_download/analyzed_data/HuBrain_C30_VS47_VZG167a_5hrPhoto_V12_JH_3-15-2022/processed/HuBrain_C30_VS47_VZG167a_5hrPhoto_V12_JH_3-15-2022.h5ad',
              '/home/imaging_mfish/surveyNAS05/scratch/vizgen_download/analyzed_data/HuBrain_C26_VS47_VZG167a_5hrbleach_V6_03-12-2022/processed/HuBrain_C26_VS47_VZG167a_5hrbleach_V6_03-12-2022.h5ad',
              '/home/imaging_mfish/surveyNAS05/scratch/vizgen_download/analyzed_data/HuBrain_C26_VS47_VZG167a_AIBS72hr_V3_03-12-2022/processed/HuBrain_C26_VS47_VZG167a_AIBS72hr_V3_03-12-2022.h5ad',
              '/home/imaging_mfish/surveyNAS05/scratch/vizgen_download/analyzed_data/HuBrain_C26_VS47_VZG167a_AIBS72hrPhoto_V11_JH_3-15-2022/processed/HuBrain_C26_VS47_VZG167a_AIBS72hrPhoto_V11_JH_3-15-2022.h5ad',
              '/home/imaging_mfish/MERSCOPENAS04_data/human/atlas/merfish_output/202204141051_H2133016Cx26MTG0200720101_VMSC02501/processed/202204141051_H2133016Cx26MTG0200720101_VMSC02501.h5ad',
              '/home/imaging_mfish/MERSCOPENAS04_data/human/atlas/merfish_output/202204151539_H2133016Cx26MTG0200720102_VMSC02501/processed/202204151539_H2133016Cx26MTG0200720102_VMSC02501.h5ad',
              '/home/imaging_mfish/MERSCOPENAS04_data/human/atlas/merfish_output/202204181242_H2133016Cx26MTG0200720103_VMSC02501/processed/202204181242_H2133016Cx26MTG0200720103_VMSC02501.h5ad'
              )

# files <- list.files(save_folder,
#                     recursive = TRUE,
#                     pattern = ".h5ad",
#                     full.names = TRUE)

anndatas <- lapply(files,read_h5ad)

combined_ad <- concat(anndatas)

umap_mfish <- umap(as.data.frame(combined_ad$X),
                   n_neighbors = 25,
                   n_components = 2,
                   metric = "euclidean",
                   min_dist = 0.4,
                   pca = 50)

combined_ad$obsm[['X_umap']] <- umap_mfish

write_h5ad(combined_ad,paste0(save_folder,"human_SAC_newtaxonomy_042822", ".h5ad"))
