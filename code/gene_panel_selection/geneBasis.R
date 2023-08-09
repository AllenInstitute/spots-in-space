# use with /allen/programs/celltypes/workgroups/rnaseqanalysis/bicore/singularity/genebasis.sif
library(geneBasisR)
library(SingleCellExperiment)
install.packages('optparse', lib="/home/stephanies/R/x86_64-pc-linux-gnu-library/4.2")
library(optparse)
install.packages('rjson', lib="/home/stephanies/R/x86_64-pc-linux-gnu-library/4.2")
library(rjson)

inputs = list(make_option("--dat_path", type="character", default="", 
              help="path to data", metavar="character"),
            make_option("--size", type="integer", default=10,
                help="size of gene panel")
)

opt_parser = OptionParser(option_list=inputs)
opts = parse_args(opt_parser)

out_path <- opts$dat_path
print(out_path)

system.time({
    exp_data_file <- paste0(out_path, "/expression_data.csv")
    anno_data_file <- paste0(out_path, "/annotation_data.csv")

    print('converting to sce...')
    sce <- raw_to_sce(counts_dir=exp_data_file, counts_type='logcounts', transform_counts_to_logcounts = FALSE, meta_dir = anno_data_file, header=TRUE, sep=",", batch=NULL)

    gB_args <- fromJSON(file=paste0(out_path, "/geneBasis_args.json"))
    
    size <- opts$size
    print('building gene panel...')
    gene_panel <- gene_search(sce, n_genes_total=size, genes_base= gB_args$genes_base, verbose = TRUE)

    write.csv(gene_panel, paste0(out_path, "/gene_list.csv"))
    print(paste0(out_path, "/gene_list.csv"))
})