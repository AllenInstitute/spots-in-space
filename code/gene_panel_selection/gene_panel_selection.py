from ast import Expr
from calendar import c
from operator import ne
import pandas
import math
import os
import numpy as np
import pickle as pkl
from datetime import datetime
import inspect

_r_init_done = False
rinterface = None
def importr(lib):
    global _r_init_done, rinterface
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr as rpy2_importr
    import rpy2.rinterface as rinterface
    from rpy2.robjects import pandas2ri
    if not _r_init_done:
        pandas2ri.activate()
        readr = rpy2_importr('readr')
    return rpy2_importr(lib)


class ExpressionDataset:
    """Encapsulates gene expression data from scRNAseq

    Parameters
    ----------
    expression_data : pandas dataframe
        Dataframe containing count of genes detected per cell. Columns are genes, with gene names contained in the column names.
        Rows are samples (cells), with the sample IDs contained in the index.
    annotation_data : pandas dataframe | None
        Dataframe containing per-cell annotations. Each row is one cell, with sample name contained in the index.
        Columns should include 'cluster', 'subclass', and 'class'.
    region : string
        Region of the brain this data comes from
    save_path : string
        Directory to save date for later recall
    logcounts : bool
        Whether the gene counts are log normalized or not
    """
    def __init__(self, expression_data: pandas.DataFrame, annotation_data: pandas.DataFrame=None, region: str=None, save_path: str=None, logcounts: bool=False):
        self.expression_data = expression_data
        self.annotation_data = annotation_data
        self.log_exp = logcounts
        self.region = region

        if save_path is not None:
            #initiate save directory and dump ExpressionDataset
            dir_ts = '{:0.3f}'.format(datetime.now().timestamp())
            self.run_directory = os.path.join(save_path, dir_ts)
            os.mkdir(self.run_directory)
            file = open(os.path.join(self.run_directory, 'expression_dataset'), 'wb')
            pkl.dump(self, file)
            file.close()
            print(f'gene panel UID: {dir_ts}')

    @classmethod
    def load_arrow(cls, gene_file: str, annotation_file: str, expression_file: str):
        """Return dataset loaded from 3 arrow files

        * ``expression_file`` contains one gene per row, one cell per column.
        * ``gene_file`` contains a column 'gene' with the same length and order as rows in expression_file.
        * ``annotation_file`` contains columns 'sample_id', 'cluster', 'subclass', and 'class'. 
        """
        annotations = pandas.read_feather(annotation_file).set_index('sample_id')
        genes = pandas.read_feather(gene_file)
        expression = pandas.read_feather(expression_file)
        expression = expression.set_index(genes['gene']).T

        exp_data = ExpressionDataset(
            expression_data=expression,
            annotation_data=annotations,
        )

        return exp_data

    @classmethod
    def load_from_timestamp(cls, directory: str, timestamp: str):
        path = os.path.join(directory, timestamp, 'expression_dataset')
        if os.path.exists(path) is False:
            print(f'expression dataset file does not exist for timestamp {timestamp} in directory {directory}')
            return
        with open(path, 'rb') as file:
            exp_data = pkl.load(file)
            file.close()
        #consider making a new directory as if this is a new run? or hav an input to indicate if this is a new run but using a
        #previously saved expression dataset?
        return exp_data
    
    def logcounts(self):
        self.expression_data = self.expression_data.applymap(lambda x: math.log(x+1, 2), na_action='ignore')
        self.log_exp = True
    

class GenePanelSelection:
    """Contains a gene panel selection as well as information about how the selection was performed.
    """
    def __init__(self, exp_data: ExpressionDataset, gene_panel: list, method: 'GenePanelMethod', args: dict):
        self.exp_data = exp_data
        self.gene_panel = gene_panel
        self.method = method
        self.args = args
        self.run_directory = self.exp_data.run_directory
        file_name = self.args.get('file_name', 'gene_panel_selection')
        file = open(os.path.join(self.run_directory, file_name), 'wb')
        pkl.dump(self, file)
        file.close()

    def __getstate__(self):
        return {k:v for (k, v) in self.__dict__.items() if not inspect.ismethod(getattr(self, k))}

    def expression_dataset(self):
        """Return a new ExpressionDataset containing only the genes selected in this panel.
        """
        exp_data = ExpressionDataset(
            expression_data=self.exp_data.expression_data[self.exp_data.expression_data.index.isin(self.gene_panel)],
            annotation_data=self.exp_data.annotation_data,
            logcounts = self.exp_data.log_exp,
        )

        if hasattr(self.exp_data, 'run_directory'):
            exp_data.run_directory = self.exp_data.run_directory
        
        return exp_data
    
    def report(self):
        print(self.args)


    @classmethod
    def load_gene_panel_selection(cls, directory='str', timestamp='str', filename='str'):
        path = os.path.join(directory, timestamp, filename)
        if os.path.exists(path) is False:
            print(f'{filename} file does not exist for timestamp {timestamp} in directory {directory}')
            return
        with open(path, 'rb') as file:
            selection = pkl.load(file)
            file.close()
        
        return selection

class GenePanelMethod():
    def select_gene_panel(self, size: int, data: ExpressionDataset, args: dict={}) -> GenePanelSelection:
        raise NotImplementedError('select_gene_panel must be implementd in a subclass')
    
    def evaluate_gene_panel(self, selection: GenePanelSelection):
        raise NotImplementedError('evaluate_gene_panel must be implemented in a subclass')
    
    def __getstate__(self):
        return {k:v for (k, v) in self.__dict__.items() if not inspect.ismethod(getattr(self, k))}


class ScranMethod(GenePanelMethod):

    def __init__(self, exp_data = ExpressionDataset):
        self.exp_data = exp_data
        self.scran = importr('scran')

    def select_gene_panel(self, size: int, args:dict):
        exp_var = self.scran.modelGeneVar(self.exp_data.expression_data)
        var_thresh = args.get('var_thresh', 0)
        top_hvgs = list(self.scran.getTopHVGs(exp_var, var_threshold=var_thresh))
        if size is not None and np.isfinite(size):
            top_hvgs = top_hvgs[:size]

        gps = GenePanelSelection(
            exp_data = self.exp_data,
            gene_panel = top_hvgs,
            method = ScranMethod,
            args = {
                'n_genes_selected': len(top_hvgs),
                'variance_threshold': args.get('var_thresh', 0),
                'used_log_counts': self.exp_data.log_exp,
                'file_name': 'hvg_selection',
            }
        )


        return gps

class GeneBasisMethod(GenePanelMethod):
    def __init__(self, exp_data: ExpressionDataset):
        self.exp_data = exp_data
        anno = exp_data.annotation_data
        anno.index.rename('cell', inplace=True) # <- renaming sample_id to 'cell' is important for geneBasis ability to read the file and align with the expression data
        self.exp_data.annotation_data = anno
        self.sce = None
        self.gB = importr('geneBasisR')
        self.SummarizedExperiment = importr('SummarizedExperiment')

    def df_to_sce(self, args:dict={}):
        print('saving expression and annotation data to csv...')
        self.save_to_csv()
        print('converting data to SCE format...')
        self.raw_to_sce(args=args)

    def save_to_csv(self):
        path = self.exp_data.run_directory
        expression_data_filename = os.path.join(path, 'expression_data.csv')
        if os.path.exists(expression_data_filename):
            print(f'{expression_data_filename} already exists')
        else:   
            self.exp_data.expression_data.to_csv(expression_data_filename, index_label=False) # having an index label throws off the matchup between the expression and anno files
        self.expression_data_file = expression_data_filename    
        annotation_data_filename = os.path.join(path, 'annotation_data.csv')
        if os.path.exists(annotation_data_filename):
            print(f'{annotation_data_filename} already exists')
        else:
            self.exp_data.annotation_data.to_csv(annotation_data_filename)
        self.annotation_data_file = annotation_data_filename
    
    def raw_to_sce(self, args: dict):
        # used in geneBasis to convert csv data files into SingleCellExperiment object used by geneBasis to select and evaluate panel
        default_args = {
            'counts_type': 'logcounts',
            'transform_to_logcounts': False,
            'header': True, 
            'sep': ',',
            'batch': rinterface.NULL,
        }

        if bool(args) is False:
            args = default_args
        self.sce = self.gB.raw_to_sce(counts_dir=self.expression_data_file, meta_dir=self.annotation_data_file, verbose=True, **args)
    
    def select_gene_panel(self, size: int, args:dict={}):
        gene_panel = self.gB.gene_search(self.sce, n_genes_total = size, **args, verbose = True)

        gps = GenePanelSelection(
            exp_data = self.exp_data,
            gene_panel = gene_panel,
            method = GeneBasisMethod,
            args = {
                'n_genes_selected': len(gene_panel),
                'used_log_counts': self.exp_data.log_exp,
                'file_name': 'gene_panel_selection',
            }
        )

        return gps

    def neighborhood_score(self, gene_panel: GenePanelSelection):
        if self.sce is None:
            self.df_to_sce()
        neighbor_score = self.gB.evaluate_library(
            self.sce, 
            genes_selection=gene_panel.gene_panel['gene'],  
            celltype_id = rinterface.NULL,
            return_cell_score_stat = True, 
            return_gene_score_stat = False, 
            return_celltype_stat = False, 
            verbose = True
            )
        neigh_df = neighbor_score.rx2['cell_score_stat'].set_index('cell').rename(columns={'cell_score': f'neighborhood_cell_score'})
        neigh_df = neigh_df.merge(gene_panel.exp_data.annotation_data, left_index=True, right_index=True)
        return neigh_df
    
    def celltype_mapping(self, gene_panel: GenePanelSelection, levels: list=['class', 'subclass', 'cluster']):
        print('Performing celltype mapping')
        if self.sce is None:
            self.df_to_sce()
        frac_mapped_dict = {}
        confusion_dict = {}
        for level in levels:
            print(f'Level: {level}')
            cell_mapping = self.gB.get_celltype_mapping(
                self.sce, 
                genes_selection=gene_panel.gene_panel['gene'],  
                celltype_id = level,
                return_stat = True, 
                )
            frac_mapped_dict[level] = cell_mapping.rx2['stat']
            df = cell_mapping.rx2['mapping']
            confusion_matrix = pandas.pivot_table(df, values='cell', index='mapped_celltype', columns='celltype', aggfunc='count', margins=True)
            confusion_dict[level] = confusion_matrix

        frac_mapped_df = pandas.concat(frac_mapped_dict, axis=1)
        confusion_matrix_df = pandas.concat(confusion_dict, axis=1)
        return frac_mapped_df, confusion_matrix_df

    def panel_eval(self, gene_panel: GenePanelSelection, levels: list=['class', 'subclass', 'cluster']):
        neighbor_score = self.neighborhood_score(gene_panel)
        frac_mapped, confusion_matrix = self.celltype_mapping(gene_panel, levels)
        return neighbor_score, frac_mapped, confusion_matrix

def gene_basis_panel_eval(gene_panel: GenePanelSelection, levels: list=['class', 'subclass', 'cluster']):
    """Peform gene panel evaluation using geneBasis. 
    Peforms two evaluations at various cell group levels:
        1) Neighborhood score
        2) Confusion matrix / fraction mapped correctly

    Parameters
    ----------
    gene_panel: GenePanelSelection

    levels: list | ['class', 'subclass', 'cluster']
        List of cell group levels to perform confusion matrix. Defaults to ['class', 'subclass', 'cluster']
    """
    gene_basis = GeneBasisMethod(gene_panel.exp_data)
    panel_eval = gene_basis.panel_eval(gene_panel, levels)
    with open(os.path.join(gene_panel.run_directory, 'gene_basis_evaluation'), 'wb') as file:
        pkl.dump(panel_eval, file)
        file.close()

    return panel_eval

def marker_gene_eval(gene_panel: GenePanelSelection, marker_gene_files: dict):
    """Evaluates how many known marker genes are present in the gene panel

    Parameters
    ----------
    gene_panel : GenePanelSelection
        GenePanelSelection with list of genes to perform evaluation on
    marker_gene_files: dictionary
        Dictionary where keys are the level (cluster, subclass, class) of the marker genes
        and values are csv files with at least one column called 'cluster' that specifies the
        cell group and 'gene' specifying a marker gene for that group
    """

    marker_gene_dict = {}
    for level, marker_gene_file in marker_gene_files.items():
        marker_genes = pandas.read_csv(marker_gene_file)
        marker_genes = marker_genes[['cluster', 'gene']]
        marker_genes['in_gene_panel'] = marker_genes.apply(lambda x: x['gene'] in gene_panel.gene_panel['gene'].to_list(), axis=1)
        marker_gene_dict[level] = marker_genes
    
    marker_genes_df = pandas.concat(marker_gene_dict, axis=1)
    return marker_genes_df

def norm_confusion_matrix(confusion_matrix: pandas.DataFrame):
    """ Normalize confusion matrix to fraction of mapped cells

    Parameters:
    -----------
    confusion_matrix: pandas.Dataframe or pandas.MultiIndex
        Dataframe or MultiIndex of pivot tables with known cell types
        on the columns and mapped cell types on the rows with elements
        containing counts. Marginal counts in a row call 'All' also
        required to calculate the fraction of correctly mapped cells.
        Included when calling pandas.DataFrame.pivot_table(margins=True)
    """
    if isinstance(confusion_matrix.columns, pandas.MultiIndex):
        norm_matrix = {}
        for level in confusion_matrix.columns.levels[0].to_list():
            table = confusion_matrix[level].dropna(how='all')
            norm_table = table.apply(lambda x: x/table.loc['All'], axis=1)
            norm_table.drop(columns='All', inplace=True)
            norm_table.drop(labels='All', inplace=True)
            norm_matrix[level] = norm_table
        return pandas.concat(norm_matrix, axis=1)
    else:
        table = confusion_matrix
        norm_matrix = table.apply(lambda x: x/table.iloc['All'], axis=1)
        norm_matrix.drop(columns='All', inplace=True)
        norm_matrix.drop(labels='All', inplace=True)

        return norm_matrix