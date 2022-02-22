################ script to adjust orientation ###############
library(tidyverse)
library(cowplot)
library(spdep)


options(stringsAsFactors = FALSE)
options(scipen=999)
options(mc.cores = 20)

inputFolder <- "/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/atlas/202202181223_60988206_VMSC00401/"

dir.create(file.path(paste0(inputFolder,"processed/")), showWarnings = FALSE)
dir.create(file.path(paste0(inputFolder,"plots/")), showWarnings = FALSE)

metadata <- read.csv(paste0(inputFolder,"/cell_metadata.csv"), 
                     header=TRUE,
                     row.names = 1,
                     check.names = FALSE)

ggplot(metadata, aes(x=center_x,y=center_y))+
  geom_point(size=0.1,stroke=0,shape=19,color="grey") +
  coord_fixed() +
  scale_y_reverse() +
  ggtitle("All cells") +
  theme_minimal()

ggsave2(paste0(inputFolder,"plots/original_orientation.pdf"))


coord <- metadata %>% 
  dplyr::select(center_x,center_y) 

colnames(coord) <- c("center_x","center_y")

rotation <- -106

coord_adj <- Rotation(coord, rotation*pi/180)
coord_adj <- as.data.frame(coord_adj)
colnames(coord_adj) <- c("corrected_x","corrected_y")

# plot data to look at changed orientation
ggplot(coord_adj, aes(x=corrected_x,y=corrected_y))+
  geom_point(size=0.5,stroke=0,shape=19,color="grey") +
  coord_fixed() +
  scale_y_reverse() +
  ggtitle("All cells") +
  theme_minimal()

ggsave2(paste0(inputFolder,"plots/adjusted_orientation.pdf"))

metadata <- merge(metadata,
                  coord_adj,
                  by=0)

metadata$rotation <- rotation

metadata <- tibble::column_to_rownames(metadata,var = "Row.names")

write.csv(metadata,paste0(inputFolder,"processed/metadata_correct_coord.csv"))
