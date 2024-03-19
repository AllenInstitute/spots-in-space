#
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

if (!is.null(ad$uns['taxonomy_name'])) {
    tax_name <- ad$uns['taxonomy_name'] 
} else {
    tax_name <- taxonomy_path
}
print(tax_name)

if (!is.null(ad$uns['taxonomy_cols'])) {
    label_cols <- unlist(ad$uns['taxonomy_cols'])
} else {
    label_cols <- c('cluster_label')
    print('no label columns specified, only returning cluster_label')
}
print(label_cols)

if (!is.null(ad$uns['taxonomy_file'])) {
    tax_file <- ad$uns['taxonomy_file'] 
} else {
    tax_file <- 'AI_taxonomy.h5ad'
}
print(tax_file)

taxonomy_anndat <- loadTaxonomy(taxonomyDir=taxonomy_path, anndata_file=tax_file)


mapped = taxonomy_mapping(
                           AIT.anndata=taxonomy_anndat,
                           query.data=qdat,
                           label.cols=label_cols,
                           corr.map=TRUE,
                           tree.map=FALSE,
                           seurat.map=FALSE
                           )

print('building results anndata')
uns <- list(
    taxonomy = tax_name,
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

print('Saving results...')
write_h5ad(ad_map, paste0(out_path, "/scrattch_map_temp.h5ad"))