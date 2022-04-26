from ast import Expr
import pandas
import math
import os
import numpy as np
import pickle as pkl
from datetime import datetime

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.rinterface as rinterface
from rpy2.robjects import pandas2ri
pandas2ri.activate()
readr = importr('readr')


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
    """
    def __init__(self, expression_data: pandas.DataFrame, annotation_data: pandas.DataFrame=None, save_path: str=None, logcounts: bool=False):
        self.expression_data = expression_data
        self.annotation_data = annotation_data
        self.log_exp = logcounts

        if save_path is not None:
            #initiate save directory and dump ExpressionDataset
            dir_ts = '{:0.3f}'.format(datetime.now().timestamp())
            self.run_directory = os.path.join(save_path, dir_ts)
            os.mkdir(self.run_directory)
            file = open(os.path.join(self.run_directory, 'expression_dataset'), 'wb')
            pkl.dump(self, file)
            file.close()

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
        path = os.path.join(directory, timestamp, '/expression_dataset')
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
    def __init__(self, exp_data: ExpressionDataset, gene_panel: list, method: GenePanelMethod, args: dict):
        self.exp_data = exp_data
        self.gene_panel = gene_panel
        self.method = method
        self.args = args
        self.run_directory = self.exp_data.run_directory
        file_name = args.get('file_name', 'gene_panel_selection')
        file = open(os.path.join(self.run_directory, file_name), 'wb')
        pkl.dump(self, file)
        file.close()


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
    
    def evaluate_panel(self, method: GenePanelMethod):
        raise NotImplementedError()


class GenePanelMethod():
    def select_gene_panel(size: int, data: ExpressionDataset, args: dict={}) -> GenePanelSelection:
        raise NotImplementedError('select_gene_panel must be implementd in a subclass')


class ScranMethod(GenePanelMethod):
    scran = importr('scran')

    def __init__(self, exp_data = ExpressionDataset):
        self.exp_data = exp_data
    
    def select_gene_panel(self, size: int, args:dict):
        exp_var = self.scran.modelGeneVar(self.exp_data.expression_data)
        var_thresh = args.get('var_thresh', 0)
        top_hvgs = list(self.scran.getTopHVGs(exp_var, var_threshold=var_thresh))
        if size is not None and np.isfinite(size):
            top_hvgs = top_hvgs[:size]

        gps = GenePanelSelection(
            exp_data = self.exp_data,
            gene_panel = top_hvgs,
            method = self,
            args = {
                'n_genes_selected': len(top_hvgs),
                'variance_threshold': args.get('var_thresh', 0),
                'used_log_counts': self.exp_data.log_exp,
                'file_name': 'hvg_selection',
            }
        )

        return gps

class GeneBasisMethod(GenePanelMethod):
    gB = importr('geneBasisR')
    SummarizedExperiment = importr('SummarizedExperiment')

    def __init__(self, exp_data: ExpressionDataset):
        self.exp_data = exp_data
        anno = exp_data.annotation_data
        anno.index.rename('cell', inplace=True) # <- renaming sample_id to 'cell' is important for geneBasis ability to read the file and align with the expression data
        self.exp_data.annotation_data = anno

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
        # used in geneBasis to convert csv data files into SingleCellExperiment object used by geneBasis to select panel
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
            method = self,
            args = {
                'n_genes_selected': len(gene_panel),
                'used_log_counts': self.exp_data.log_exp,
                'file_name': 'gene_panel_selection',
            }
        )

        return gps

    