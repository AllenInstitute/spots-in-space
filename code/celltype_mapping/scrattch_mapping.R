#
# make sure you use docker image docker://alleninst/arrow_scrattch_on_hpc
#
library(scrattch.mapping)
library(anndata)
library(optparse)

inputs = list(make_option("--dat_path", type="character", default="", 
              help="path to data", metavar="character"))

opt_parser = OptionParser(option_list=inputs)
opts = parse_args(opt_parser)

out_path <- opts$dat_path
print(out_path)

# load a log normalized count matrix (gene x cell)
ad <- read_h5ad(paste0(out_path, "/scrattch_map_temp.h5ad"))
qdat <- t(ad$X)

taxonomy_path <- ad$uns['taxonomy_path']
print(taxonomy_path)
taxonomy_anndat <- loadTaxonomy(taxonomyDir=taxonomy_path)

mapped = taxonomy_mapping(
                           AIT.anndata=taxonomy_anndat,
                           query.data=qdat,
                           label.cols=c("level1.class_label", "level2.neighborhood_label", "level3.subclass_label", "cluster_label"),
                           corr.map=TRUE,
                           tree.map=FALSE,
                           seurat.map=FALSE
                           )

# mapped = run_mapping_on_taxonomy(qdat,
#                                  Taxonomy="AIT20.0_macaque",
#                                  prefix="cDNAD5",               
#                                  prebuild=FALSE,
#                                  newbuild=TRUE,
#                                  mapping.method='hierarchy',
#                                  mc.cores=10,
#                                  iter=10,
#                                  blocksize=5000)

# results = mapped$cl.df[match(mapped$best.map.df$best.cl, mapped$cl.df$cl),]
# write.csv(results, paste0(out_path, "/scrattch_map_results.csv"))

uns <- list(
    taxonomy = 'AIT11.5_macaque',
    method = 'corr',
    iterations = 100
    )
ad_map <- AnnData(
  X = t(qdat),
  obs = ad$obs,
  var = ad$var,
  obsm = list(
    mapping_results = mapped
    ),
  uns = uns
)

write_h5ad(ad_map, paste0(out_path, "/scrattch_map_temp.h5ad"))