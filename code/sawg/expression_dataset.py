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
import inspect
import anndata as ad
import json
from sawg.celltype_mapping import CellTypeMapping
from sawg.hpc import run_slurm

_r_init_done = False
rinterface = None
def importr(lib):
    # os.environ['R_HOME'] = 'C:\\Users\stephanies\AppData\Local\Continuum\miniconda3\envs\r_env\lib\R'
    # os.environ['R_HOME'] = "C:/PROGRA~1/R/R-43~1.0"
    global _r_init_done, rinterface, ro
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr as rpy2_importr
    import rpy2.rinterface as rinterface
    from rpy2.robjects import pandas2ri
    if not _r_init_done:
        pandas2ri.activate()
        print('pandas2ri activated')
        readr = rpy2_importr('readr')
    return rpy2_importr(lib)


class ExpressionDataset:
    """Encapsulates gene expression data from scRNAseq

    Parameters
    ----------
    expression_data : AnnData 
        AnnData with following attributes:
            X: cell x gene count
            obs: cell annotations with at least 'class', 'subclass', 'cluster' annotations
            var: gene names
    expression_type : str
        Indicates the kind of data in *expression_data.X*. Options are "raw" (raw expression counts), "cpm", "logcpm", "binary"
    region : string
        Region of the brain this data comes from
    save_path : string
        Directory to save date for later recall
    """
    def __init__(self, expression_data: ad.AnnData, expression_type,  region: str=None, save_path: str=None):
        self.expression_data = expression_data
        self.expression_type = expression_type
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
        else:
            self.run_directory = None

    @property
    def genes(self):
        return self.expression_data.columns

    def normalize(self, expression_type: str):
        ## Needs updating to be compatible with expression_dataset = AnnData
        """Return a normalized copy of this dataset.

        Parameters
        ----------
        expression_type : str
            Type of data to return. Options are 'cpm', 'logcpm', and 'binary'
        """        
        if expression_type == self.expression_type:
            return self.copy()
        
        if expression_type == 'cpm':
            assert self.expression_type == 'raw'
            raise NotImplementedError()
        elif expression_type == 'logcpm':
            assert self.expression_type in ['raw', 'cpm']
            if self.expression_type == 'raw':
                return self.normalize('cpm').normalize('logcpm')
            elif self.expression_type == 'cpm':
                expression = np.log2(1 + self.expression_data)
                return self.copy(expression_data=expression, expression_type='logcpm')
        elif expression_type == 'binary':
            assert self.expression_type in ['raw', 'cpm']
            return self.copy(expression_data=(self.expression_data > 0), expression_type='binary')

    @classmethod
    def load_arrow(cls, gene_file: str, annotation_file: str, expression_file: str, expression_type: str, max_n_samples: typing.Optional[int]=None):
        """Return dataset loaded from 3 arrow files

        Parameters
        ----------
        gene_file : str
            Name of arrow file containing a column 'gene' with the same length and order as rows in expression_file.
        annotation_file : str
            Name of arrow file containing columns 'sample_id', 'cluster', 'subclass', and 'class'. 
        expression_file : str
            Name of arrow file containing one gene per row, one cell per column.
        expression_type : str
            Type of expression data provided in *expression_file* (see ExpressionDataset.__init__(expression_type))
        """
        annotations = pandas.read_feather(annotation_file).set_index('sample_id')
        genes = pandas.read_feather(gene_file)
        expression = pandas.read_feather(expression_file)
        expression = expression.set_index(genes['gene']).T
        assert len(expression) == len(annotations)

        if max_n_samples is None:
            max_n_samples = len(expression)

        anndata = ad.AnnData(
            X = expression.iloc[:max_n_samples],
            obs = annotations.iloc[:max_n_samples],
            var = pandas.DataFrame(index=expression.columns)
        )

        exp_data = ExpressionDataset(
            expression_data=anndata,
            expression_type=expression_type,
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
    
    def select_genes(self, genes: list) -> 'ExpressionDataset':
        exp_data = self.copy(expression_data=self.expression_data[:, genes])
        if hasattr(self, 'run_directory'):
            exp_data.run_directory = self.run_directory
        
        return exp_data

    def copy(self, **kwds):
        """Create a copy of this ExpressionDataset with some arguments changed.
        """
        default_kwds = dict(
            expression_data=self.expression_data,
            expression_type=self.expression_type,
        )
        default_kwds.update(kwds)
        exp_data =  ExpressionDataset(**default_kwds)
        exp_data.run_directory = self.run_directory
        return exp_data


class GenePanelSelection:
    """Contains a gene panel selection as well as information about how the selection was performed.
    """
    def __init__(self, exp_data: ExpressionDataset, gene_panel: list, method: 'GenePanelMethod', args: dict={}):
        self.exp_data = exp_data
        self.gene_panel = gene_panel
        self.method = method
        self.args = args
        self.run_directory = self.exp_data.run_directory
        file_name = self.args.get('file_name', 'gene_panel_selection')
        self.save_selection(file_name = file_name)

    def __getstate__(self):
        return {k:v for (k, v) in self.__dict__.items() if not inspect.ismethod(getattr(self, k))}

    def expression_dataset(self):
        """Return a new ExpressionDataset containing only the genes selected in this panel.
        """
        return self.exp_data.select_genes(self.gene_panel)
    
    def report(self):
        print(self.args)

    def save_selection(self, save_path:str=None, file_name:str='gene_panel_selection', meta:dict={}):
        self.args.update(meta)
        if self.run_directory is None and save_path is None:
            print('Must specify save path or set GenePanelSelection.run_directory to path')
            return

        path = save_path if save_path is not None else self.run_directory
        file = open(os.path.join(path, file_name), 'wb')
        pkl.dump(self, file)
        file.close()

    @classmethod
    def load_from_timestamp(cls, directory:str, timestamp:str, filename:str='gene_panel_selection'):
        path = os.path.join(directory, timestamp, filename)
        if os.path.exists(path) is False:
            print(f'{filename} file does not exist for timestamp {timestamp} in directory {directory}')
            return
        with open(path, 'rb') as file:
            selection = pkl.load(file)
            file.close()
        
        return selection

    def eval_method(self, method: CellTypeMapping):
        self.eval = method
        self.eval._set_run_directory(save_path=self.run_directory)
    
    def get_confusion_matrix(self, **args):
        # instantiate in evaluation method of CellTypeMapping class
        if self.eval is not None:
            self.confusion_matrix = self.eval.get_confusion_matrix(self.eval, **args)
        else:
            raise NotImplementedError('No GenePanelSelection evaluation method is defined')
    
    def get_fraction_mapped(self, **args):
        # pull fraction mapped from diagonal of confusion matrix
        if self.confusion_matrix is None:
            if self.eval is None:
                raise NotImplementedError('No GenePanelSelection evaluation method is defined to generate confusion matrix')
            try:
                self.confusion_matrix = self.eval.get_confusion_matrix(self.eval, **args)
            except:
                raise("Tried generating confusion matrix but don't have appropriate arguments")
                
            self.fraction_mapped = self.eval.get_fraction_mapped(self.eval, self.confusion_matrix, **args)
        else:
            raise NotImplementedError('No GenePanelSelection evaluation method is defined')
        

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

    def select_gene_panel(self, size: int, args:dict, file_name: str='hvg_selection'):
        expression_data = self.exp_data.expression_data.T
        exp_var = self.scran.modelGeneVar(expression_data)
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
                'file_name': file_name,
            }
        )


        return gps


class GeneBasisMethod(GenePanelMethod):

    def __init__(self, exp_data: ExpressionDataset):
        exp_data.expression_data.obs.index.rename('cell', inplace=True) # <- renaming sample_id to 'cell' is important for geneBasis ability to read the file and align with the expression data
        self.exp_data = exp_data
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
            expression_data = self.exp_data.expression_data.to_df().T  
            expression_data.to_csv(expression_data_filename, index_label=False) # having an index label throws off the matchup between the expression and anno files
        self.expression_data_file = expression_data_filename    
        annotation_data_filename = os.path.join(path, 'annotation_data.csv')
        if os.path.exists(annotation_data_filename):
            print(f'{annotation_data_filename} already exists')
        else:
            self.exp_data.expression_data.obs['cell'].to_csv(annotation_data_filename)
        self.annotation_data_file = annotation_data_filename
    
    def raw_to_sce(self, args: dict):
        """ Used in geneBasis to convert csv data files into SingleCellExperiment 
        object used by geneBasis to select and evaluate panel

        Parameters:
        -----------
        args : dict
            Optional arguments to pass to geneBasis.raw_to_sce
        """
        default_args = {
            'counts_type': 'logcounts',
            'transform_to_logcounts': False,
            'header': True, 
            'sep': ',',
            'batch': rinterface.NULL,
        }

        args.update(default_args)
        self.sce = self.gB.raw_to_sce(counts_dir=self.expression_data_file, meta_dir=self.annotation_data_file, verbose=True, **args)
    
    def select_gene_panel(self, size: int, file_name: str='gene_panel_selection', args:dict={}, run_on_hpc=False, hpc_args={}):
        """
        Paramters
        ---------
        size : int
            Size of gene panel to select
        data : ExpressionDataset
            Gene expression dataset from which to derive panel
        args : dict | {}
            Optional arguments to pass to geneBasis.gene_search
        """

        # check some R conversions to optional arguments
        if bool(args):
                for k, v in args.items():
                    if v is None:
                        args[k] = rinterface.NULL
                    if type(v) is list and all(isinstance(element, str) for element in v):
                        args[k] = ro.vectors.StrVector(v)

        if run_on_hpc is True:
            job = self.run_on_hpc(size=size, hpc_args=hpc_args, geneBasis_args=args)
            return job
        
        else:
            # run locally
            gene_panel = self.gB.gene_search(self.sce, n_genes_total = size, **args, verbose = True)

            gps = GenePanelSelection(
                exp_data = self.exp_data,
                gene_panel = gene_panel['gene'].to_list(),
                method = GeneBasisMethod,
                args = {
                    'n_genes_selected': len(gene_panel),
                    'expression_data_type': self.exp_data.expression_type,
                    'file_name': file_name,
                }
            )

            return gps
        
    def run_on_hpc(self, size, hpc_args:dict={}, geneBasis_args:dict={}):
        with open(self.exp_data.run_directory + '/geneBasis_args.json', 'w') as file:
            json.dump(geneBasis_args, file)

        job_path = hpc_args.get('job_path', '//allen/programs/celltypes/workgroups/rnaseqanalysis/NHP_spatial/gene_panel_runs/')
        docker = hpc_args.get('docker', 'singularity exec /allen/programs/celltypes/workgroups/rnaseqanalysis/bicore/singularity/genebasis.sif')
        r_script = hpc_args.get('r_script', 'geneBasis.R')

        print('building HPC job')
        hpc_default = {
            'hpc_host': 'hpc-login',
            'job_path': job_path,
            'partition': 'celltypes',
            'job_name': 'geneBasis',
            'nodes': 1,
            'mincpus': 10,
            'mem': '500G',
            'time':'24:00:00',
            'mail_user': None,
            'output': job_path + 'hpc_logs/%j.out',
            'error': job_path + 'hpc_logs/%j.err',
        }

        hpc_default.update(hpc_args)
        
        command = f"""
                cd {job_path}

                {docker} Rscript {r_script} --dat_path {self.exp_data.run_directory} --size {size}
            """
        
        assert 'command' not in hpc_args
        hpc_default['command'] = command
        job = run_slurm(**hpc_default)
        print(job.state(), job.job_id)
        return job
    
    def load_gene_panel(self, args: dict={}):
        gene_list = pandas.read_csv(self.run_directory + '/gene_list.csv', names=['gene'], header=0)
        
        args.update({
                'n_genes_selected': len(gene_list),
                'expression_data_type': self.exp_data.expression_type,
                'file_name': 'gene_panel_selection',
                })

        gps = GenePanelSelection(
            exp_data = self.exp_data,
            gene_panel = gene_list['gene'].to_list(),
            method = self,
            args=args,
        )

        return gps

    def neighborhood_score(self, gene_panel: GenePanelSelection):
        if self.sce is None:
            self.df_to_sce()
        neighbor_score = self.gB.evaluate_library(
            self.sce, 
            genes_selection=ro.vectors.StrVector(gene_panel.gene_panel),  
            celltype_id = rinterface.NULL,
            return_cell_score_stat = True, 
            return_gene_score_stat = False, 
            return_celltype_stat = False, 
            verbose = True
            )
        neigh_df = neighbor_score.rx2['cell_score_stat']
        if not isinstance(neigh_df, pandas.DataFrame):
            try:
                neigh_df = pandas2ri.rpy2py(neigh_df)
            except AttributeError:
                neigh_df = pandas2ri.ri2py(neigh_df)
        neigh_df = neigh_df.set_index('cell').rename(columns={'cell_score': f'neighborhood_cell_score'})
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
                genes_selection=ro.vectors.StrVector(gene_panel.gene_panel),  
                celltype_id = level,
                return_stat = True, 
                )
            frac_mapped_dict[level] = pandas2ri.rpy2py(cell_mapping.rx2['stat'])
            df = pandas2ri.rpy2py(cell_mapping.rx2['mapping'])
            confusion_matrix = pandas.pivot_table(df, values='cell', index='mapped_celltype', columns='celltype', aggfunc='count', margins=True)
            confusion_dict[level] = confusion_matrix

        frac_mapped_df = pandas.concat(frac_mapped_dict, axis=1)
        confusion_matrix_df = pandas.concat(confusion_dict, axis=1)
        return frac_mapped_df, confusion_matrix_df

    def panel_eval(self, gene_panel: GenePanelSelection, levels: list=['class', 'subclass', 'cluster']):
        neighbor_score = self.neighborhood_score(gene_panel)
        frac_mapped, confusion_matrix = self.celltype_mapping(gene_panel, levels)
        return neighbor_score, frac_mapped, confusion_matrix


class PROPOSEMethod(GenePanelMethod):
    def select_gene_panel(self, size: int, data: ExpressionDataset, use_classes=True, annotation_column='cluster', cuda_device=0, train_fraction=0.8, test_fraction=0.1, pre_eliminate=500) -> GenePanelSelection:
        """

        Paramters
        ---------
        size : int
            Size of gene panel to select
        data : ExpressionDataset
            Gene expression dataset from which to derive panel
        use_classes : bool
            If True, optimize for predicting classes rather than gene expression
        annotation_column : str
            Name of annotation column that holds classes (when use_clases is True)
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
    
        # For data splitting
        n = len(data.expression_data)
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
        binary = data.normalize('binary').expression_data.to_numpy(dtype='float32')
        if use_classes:
            classes = data.annotation_data[annotation_column].to_numpy()
            self.class_codes = pandas.Categorical(classes).codes
            train_dataset = ProposeExpressionDataset(binary[train_inds], self.class_codes[train_inds])
            val_dataset = ProposeExpressionDataset(binary[val_inds], self.class_codes[val_inds])
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            logcpm = data.normalize('logcpm').expression_data.to_numpy(dtype='float32')
            train_dataset = ProposeExpressionDataset(binary[train_inds], logcpm[train_inds])
            val_dataset = ProposeExpressionDataset(binary[val_inds], logcpm[val_inds])
            loss_fn = HurdleLoss()

        selector = PROPOSE(
            train_dataset,
            val_dataset,
            loss_fn=loss_fn,
            device=torch.device('cuda', cuda_device),
            hidden=[128, 128]
        )

        # Eliminate many candidates
        candidates, model = selector.eliminate(target=pre_eliminate, mbsize=128, max_nepochs=500, lam_init=0.00001)

        # Select specific number of genes
        inds, model = selector.select(num_genes=size, mbsize=128, max_nepochs=500)

        gps = GenePanelSelection(
            exp_data = data,
            gene_panel = data.genes[inds],
            method = self,
            args = {
                'use_classes': use_classes,
                'annotation_column': annotation_column,
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

        return gps

class mFISHtoolsMethod(GenePanelMethod):
    def __init__(self, exp_data: ExpressionDataset):
        self.exp_data = exp_data
        self.run_directory = self.exp_data.run_directory
        self.filter_genes_params = None

    def filter_genes(self, starting_genes: list=[], num_binary_genes: int=3000, **args):
        filter_genes_params = {
            'startingGenes': starting_genes,
            'numBinaryGenes': num_binary_genes,
        }

        filter_genes_params.update(**args)
        
        self.filter_genes_params = filter_genes_params

    def select_gene_panel(self, size: int, cluster_label: str, panel_args:dict={}, hpc_args:dict={}, 
                          docker: str='singularity exec --cleanenv docker://bicore/scrattch_mapping:latest', r_script: str='mFISHTools.R'):
        
        print('building temp ad_h5ad')
        self.build_panel_ad(size, cluster_label, panel_args)

        job_path = hpc_args.get('job_path', '//allen/programs/celltypes/workgroups/rnaseqanalysis/NHP_spatial/gene_panel_runs/')

        print('building HPC job')
        hpc_default = {
            'hpc_host': 'hpc-login',
            'job_path': job_path,
            'partition': 'celltypes',
            'job_name': 'mfishtools',
            'nodes': 1,
            'mincpus': 10,
            'mem': '500G',
            'time':'3-0:00:00',
            'mail_user': None,
            'output': job_path + 'hpc_logs/%j.out',
            'error': job_path + 'hpc_logs/%j.err',
        }

        hpc_default.update(hpc_args)
        
        command = f"""
                cd {job_path}

                {docker} Rscript {r_script} --dat_path {self.run_directory}
            """
        
        assert 'command' not in hpc_args
        hpc_default['command'] = command
        job = run_slurm(**hpc_default)
        print(job.state(), job.job_id)
        return job
        
    def build_panel_ad(self, size: int, cluster_label: str, panel_args:dict={}):
        output_ad = self.exp_data.expression_data.copy()
        output_ad.obs.rename(columns={cluster_label: 'cluster_label'}, inplace=True) # controlled strings for mFISHtools
        build_panel_params = {
            'panelSize': size,
            'currentPanel': self.filter_genes_params['startingGenes'] if self.filter_genes_params is not None else []
        }
        build_panel_params.update(**panel_args)
        self.build_panel_params = build_panel_params

        output_ad.uns['build_panel'] = build_panel_params
        if self.filter_genes_params is not None:
            output_ad.uns['filter_panel'] = self.filter_genes_params

        output_ad.write_h5ad(self.run_directory + '/mfishtools_ad_temp.h5ad')

    def load_gene_panel(self, args: dict={}):
        gene_list = pandas.read_csv(self.run_directory + '/gene_list.csv', names=['gene', 'frac_mapped'], header=0)
        
        args.update({
                'n_genes_selected': len(gene_list),
                'expression_data_type': self.exp_data.expression_type,
                'file_name': 'gene_panel_selection',
                'frac_mapped': gene_list
                })

        gps = GenePanelSelection(
            exp_data = self.exp_data,
            gene_panel = gene_list['gene'].to_list(),
            method = self,
            args=args,
        )

        print('deleting mfishtools temp file...')
        os.remove(self.run_directory + '\mfishtools_ad_temp.h5ad')

        return gps

class SeuratMethod(GenePanelMethod):
    def select_gene_panel(self, size: int, data: ExpressionDataset, flavor='seurat_v3') -> GenePanelSelection:
        """
        Paramters
        ---------
        size : int
            Size of gene panel to select
        data : ExpressionDataset
            Gene expression dataset from which to derive panel
        """
        if flavor == 'seurat_v3':
            assert data.expression_type == 'raw'
        else:
            assert data.expression_type == 'logcpm'

        import scanpy
        from anndata import AnnData
        ann = AnnData(X=data.expression_data.values.astype('float32'))
        annotations = scanpy.pp.highly_variable_genes(ann, n_top_genes=size, flavor=flavor, inplace=False)

        if flavor == 'seurat_v3':
            inds = np.where(~np.isnan(annotations['highly_variable_rank'].values))[0]
        else:
            inds = np.where(annotations['highly_variable'].values)[0]

        gps = GenePanelSelection(
            exp_data = data,
            gene_panel = data.genes[inds],
            method = self,
            args = {
                'n_genes_selected': len(inds),
            },
        )
        return gps


def gene_basis_panel_eval(gene_panel: GenePanelSelection, levels: list=['class', 'subclass', 'cluster'], file_name: str='gene_basis_evaluation'):
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
    with open(os.path.join(gene_panel.run_directory, file_name), 'wb') as file:
        pkl.dump(panel_eval, file)
        file.close()

    return panel_eval

def marker_gene_eval(gene_panel: GenePanelSelection, marker_gene_files: dict, cluster_col='cluster', gene_col='gene'):
    """Evaluates how many known marker genes are present in the gene panel

    Parameters
    ----------
    gene_panel : GenePanelSelection
        GenePanelSelection with list of genes to perform evaluation on
    marker_gene_files: dictionary
        Dictionary where keys are the level (cluster, subclass, class) of the marker genes
        and values are csv files with at one column designating the cell group (cluster_col)  
        and another column specifying a marker gene for that group (gene_col)
    """

    marker_gene_dict = {}
    for level, marker_gene_file in marker_gene_files.items():
        marker_genes = pandas.read_csv(marker_gene_file)
        marker_genes = marker_genes[[cluster_col, gene_col]]
        marker_genes['in_gene_panel'] = marker_genes.apply(lambda x: x[gene_col] in gene_panel.gene_panel, axis=1)
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
            if 'All' not in table.T.columns:
                table.loc['All'] = table.sum(axis=0)
            norm_table = table.apply(lambda x: x/table.loc['All'], axis=1)
            norm_table.drop(columns='All', inplace=True)
            norm_table.drop(labels='All', inplace=True)
            norm_matrix[level] = norm_table
        return pandas.concat(norm_matrix, axis=1)
    else:
        table = confusion_matrix
        if 'All' not in table.T.columns:
            table.loc['All'] = table.sum(axis=0)
        norm_matrix = table.apply(lambda x: x/table.loc['All'], axis=1)
        norm_matrix.drop(columns='All', inplace=True)
        norm_matrix.drop(labels='All', inplace=True)
        return norm_matrix

def load_panel_eval(path: str):
    with open(path, 'rb') as file:
        eval = pkl.load(file)
        file.close()
    
    return eval