from ast import Expr
from calendar import c
from operator import ne
import pandas
import math
import os
import typing
import numpy as np
import pickle as pkl
from datetime import datetime

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
    """
    def __init__(self, expression_data: pandas.DataFrame, annotation_data: pandas.DataFrame=None,  cpm_data: pandas.DataFrame=None, save_path: str=None, logcounts: bool=False):
        self.expression_data = expression_data
        self.annotation_data = annotation_data
        self._cpm_data = cpm_data
        self.log_exp = logcounts

        if save_path is not None:
            #initiate save directory and dump ExpressionDataset
            dir_ts = '{:0.3f}'.format(datetime.now().timestamp())
            self.run_directory = os.path.join(save_path, dir_ts)
            os.mkdir(self.run_directory)
            file = open(os.path.join(self.run_directory, 'expression_dataset'), 'wb')
            pkl.dump(self, file)
            file.close()

    @property
    def genes(self):
        return self.expression_data.columns

    @property
    def cpm_data(self):
        if self._cpm_data is None:
            # todo: generate from raw data
            raise NotImplementedError()
        return self._cpm_data

    @classmethod
    def load_arrow(cls, gene_file: str, annotation_file: str, expression_file: str, cpm_file: typing.Optional[str]):
        """Return dataset loaded from 3 arrow files

        * ``expression_file`` contains one gene per row, one cell per column.
        * ``gene_file`` contains a column 'gene' with the same length and order as rows in expression_file.
        * ``annotation_file`` contains columns 'sample_id', 'cluster', 'subclass', and 'class'. 
        """
        annotations = pandas.read_feather(annotation_file).set_index('sample_id')
        genes = pandas.read_feather(gene_file)
        expression = pandas.read_feather(expression_file)
        expression = expression.set_index(genes['gene']).T
        if cpm_file is not None:
            cpm = pandas.read_feather(cpm_file)
            cpm = cpm.set_index(genes['gene']).T
        else:
            cpm = None

        exp_data = ExpressionDataset(
            expression_data=expression,
            annotation_data=annotations,
            cpm_data=cpm,
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
    def __init__(self, exp_data: ExpressionDataset, gene_panel: list, method: 'GenePanelMethod', args: dict):
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
    def select_gene_panel(self, size: int, data: ExpressionDataset, args: dict={}) -> GenePanelSelection:
        raise NotImplementedError('select_gene_panel must be implementd in a subclass')
    
    def evaluate_gene_panel(self, selection: GenePanelSelection):
        raise NotImplementedError('evaluate_gene_panel must be implemented in a subclass')


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
            method = self,
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
    
    def celltype_mapping(self, gene_panel: GenePanelSelection):
        print('Performing celltype mapping')
        if self.sce is None:
            self.df_to_sce()
        frac_mapped_dict = {}
        confusion_dict = {}
        for level in ['class', 'subclass', 'cluster']:
            print(f'Level: {level}')
            cell_mapping = self.gB.get_celltype_mapping(
                self.sce, 
                genes_selection=gene_panel.gene_panel['gene'],  
                celltype_id = level,
                return_stat = True, 
                )
            frac_mapped_dict[level] = cell_mapping.rx2['stat']
            confusion_matrix = pandas.pivot_table(cell_mapping.rx2['mapping'], values='cell', index='mapped_celltype', columns='celltype', aggfunc='count')
            confusion_dict[level] = confusion_matrix

        frac_mapped_df = pandas.concat(frac_mapped_dict, axis=1)
        confusion_matrix_df = pandas.concat(confusion_dict, axis=1)
        return frac_mapped_df, confusion_matrix_df

    def panel_eval(self, gene_panel: GenePanelSelection):
        neighbor_score = self.neighborhood_score(gene_panel)
        frac_mapped, confusion_matrix = self.celltype_mapping(gene_panel)
        return neighbor_score, frac_mapped, confusion_matrix

def gene_basis_panel_eval(gene_panel: GenePanelSelection):
    gene_basis = GeneBasisMethod(gene_panel.exp_data)
    panel_eval = gene_basis.panel_eval(gene_panel)
    with open(os.path.join(gene_panel.run_directory, 'gene_basis_evaluation'), 'wb') as file:
        pkl.dump(panel_eval, file)
        file.close()

    return panel_eval



class PROPOSEMethod(GenePanelMethod):
    def select_gene_panel(self, size: int, data: ExpressionDataset, cuda_device=0, train_fraction=0.8, test_fraction=0.1, pre_eliminate=500) -> GenePanelSelection:
        """
        Paramters
        ---------
        size : int
            Size of gene panel to select
        data : ExpressionDataset
            Gene expression dataset from which to derive panel
        cuda_device : int
            Index of cuda GPU device to use
        train_fraction : float
            Fraction of expression data to use for training (samples not used for training and testing are used for validation)
        test_fraction : float
            Fraction of expression data to hold out for testing
        pre_eliminate : int
            Maximum size of gene set to process with model
        """
        from propose import PROPOSE, HurdleLoss
        from propose import ExpressionDataset as ProposeExpressionDataset
        import torch
        import torch.nn as nn

        # Arrange raw and CPM data
        raw = data.expression_data.values.astype(np.float32)
        cpm = data.cpm_data.values.astype(np.float32)

        # Generate logarithmized and binarized data
        binary = (raw > 0).astype(np.float32)
        log = np.log(1 + raw)
        logcpm = np.log(1 + cpm)

        # For data splitting
        n = len(raw)
        n_train = int(train_fraction * n)
        n_test = int(test_fraction * n)
        all_rows = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(all_rows)
        train_inds = all_rows[:n_train]
        val_inds = all_rows[n_train:-n_test]
        test_inds = all_rows[-n_test:]
        print(f'{n} total examples, {len(train_inds)} training examples, {len(val_inds)} validation examples, {len(test_inds)} test examples')

        # Set up datasets
        train_dataset = ProposeExpressionDataset(binary[train_inds], logcpm[train_inds])
        val_dataset = ProposeExpressionDataset(binary[val_inds], logcpm[val_inds])

        selector = PROPOSE(
            train_dataset,
            val_dataset,
            loss_fn=HurdleLoss(),
            device=torch.device('cuda', cuda_device),
            hidden=[128, 128]
        )

        # Eliminate many candidates
        candidates, model = selector.eliminate(target=pre_eliminate, mbsize=128, max_nepochs=500)

        # Select specific number of genes
        inds, model = selector.select(num_genes=size, mbsize=128, max_nepochs=500)

        gps = GenePanelSelection(
            exp_data = data,
            gene_panel = data.genes[inds],
            method = self,
            args = {
                'n_genes_selected': len(inds),
                'train_inds': train_inds,
                'validation_inds': val_inds,
                'test_inds': test_inds,
                'pre_candidates': candidates,
                'model': model,
                'train_fraction': train_fraction,
                'test_fraction': test_fraction,
                'pre_eliminate': pre_eliminate,
            },
        )
