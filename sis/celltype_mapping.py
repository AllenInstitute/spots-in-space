from __future__ import annotations
import pandas as pd
# from code.gene_panel_selection.gene_panel_selection import ExpressionDataset
# from code.segmentation.segmentation import SpotTable
import numpy as np
from datetime import datetime
import os
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import json
import anndata as ad
import umap
from sis.hpc import run_slurm

try:
    import tangram as tg
except ImportError:
    pass

class CellTypeMapping():
    """Container class for cell type mapping methods
    """
    @property
    def ordered_labels(self, reload=False):
        if hasattr(self, '_ordered_labels') is False or self._ordered_labels is None or reload is True:
            with open('//allen/programs/celltypes/workgroups/rnaseqanalysis/NHP_spatial/snRNAseq_data/ordered_celltype_labels.json') as file:
                self._ordered_labels = json.load(file)
        return self._ordered_labels

    def _create_run_directory(self, save_path):
        dir_ts = '{:0.3f}'.format(datetime.now().timestamp())
        self.run_directory = os.path.join(save_path, dir_ts)
        os.mkdir(self.run_directory)
        print(f'Mapping UID: {dir_ts}')

    def _set_run_directory(self, save_path):
        self.run_directory = save_path

    def save_mapping(self, save_path: str|None=None, file_name:str='mapping', meta:dict={}, replace=False):
        self.meta.update(meta)
        if self.run_directory is None or replace is False:
            if save_path is None:
                print('Must set a directory path')
                return
            self._create_run_directory(save_path)
        with open(os.path.join(self.run_directory, 'meta.json'), "w") as f:
            json.dump(self.meta, f)
        with bz2.BZ2File(os.path.join(self.run_directory, file_name + '.pbz2'), 'w') as f: 
            cPickle.dump(self, f)

    @classmethod
    def load_from_timestamp(cls, directory: str, timestamp: str, file_name: str='mapping', meta_only=False):
        path = os.path.join(directory, timestamp)
        meta_file = os.path.join(path, 'meta.json')
        if meta_only and os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
                return meta
        data_file = os.path.join(path, file_name + '.pbz2')
        if os.path.exists(data_file) is False:
            print(f'analysis file {file_name} does not exist for timestamp {timestamp} in directory {directory}')
            return
        ct_map = bz2.BZ2File(data_file, 'rb')
        ct_map = cPickle.load(ct_map)

        return ct_map
    
    def reset_qc(self):
        # reset qc status 
        self.ad_map.obs['mapping_qc'] = 'pass'
        self.ad_map.uns['qc_params'] = None

    def qc_mapping(self, qc_params: dict):
        # QC mapping results using thresholds for various performance metrics in self.ad_map.obs

        qc_mask = None
        self.reset_qc()
        for param, thresh in qc_params.items():
            if qc_mask is None:
                qc_mask = (self.ad_map.obs[param] < thresh)
            else:
                qc_mask = qc_mask & (self.ad_map.obs[param] < thresh)
            
        self.ad_map.obs.loc[qc_mask, 'mapping_qc'] = 'fail'
        self.ad_map.uns['qc_params'] = qc_params
    
    def spatial_umap(self, attr: str='ad_sp', umap_args: dict|None=None):
        # UMAP spatial gene expression. Assumes X of anndata object is a cell x gene
        umap_attr = getattr(self, attr)

        if umap_args is not None:
            mapper = umap.UMAP(**umap_args)
        else:
            mapper = umap.UMAP()
        embedding = mapper.fit_transform(umap_attr.X)

        umap_attr.obs['umap_x'] = embedding[:, 0]
        umap_attr.obs['umap_y'] = embedding[:, 1]

    def mapping(self):
        raise NotImplementedError('mapping must be implementd in a subclass')

class TangramMapping(CellTypeMapping):
    def __init__(self, sc_data: 'AnnData'|str, sp_data: 'AnnData'|str, meta: dict={}):
        """
        sc_data: AnnData object or filename of AnnData object representing single-cell or singcle-nucleus data or None
        sp_data: AnnData object or filename of AnnData object representing spatial transcriptomics data or None
        """
        import scanpy as sc

        self.run_directory = None
       
        if isinstance(sc_data, str):
            ad_sc = ad.read_h5ad(sc_data)
        else:
            ad_sc = sc_data
        if isinstance(sp_data, str):
            ad_sp = ad.read_h5ad(sp_data)
        else:
            ad_sp = sp_data

        # library size correction, normalize count within each cell to fixed number
        sc.pp.normalize_total(ad_sc)

        meta.update({'mapping_method': 'Tangram'})

        self.ad_sc = ad_sc
        self.ad_sp = ad_sp
        self.meta = meta

        self._ordered_labels = None

    def set_training_genes(self, training_genes: list, meta: dict={}):
       
        print(f'starting with {len(training_genes)} training genes..')
        tg.pp_adatas(self.ad_sc, self.ad_sp, genes=training_genes)
        self.meta.update(meta)

    def mapping(self, device: str='cpu', mode: str='clusters', cluster_label: 'None|str'='subclass', args:dict={}, meta:dict={}):
        """ main mapping function using tangram.map_cells_to_space. Default to cluster mode at subclass level.
        Pass other argumaents to tangram.map_cells_to_space via args
        """

        ad_map = tg.map_cells_to_space(
            adata_sc = self.ad_sc,
            adata_sp = self.ad_sp,
            device = device,
            mode = mode,
            cluster_label = cluster_label,
            **args,
        )

        self.ad_map = ad_map
        self.meta.update({'mode': mode, 'cluster_label': cluster_label})
        self.meta.update(meta)

    def project_genes(self, args:dict={}):
        """Project gene expression from RNAseq data to spatial data
        """

        ad_ge = tg.project_genes(self.ad_map, self.ad_sc, cluster_label=self.meta['cluster_label'], **args)
        self.ad_ge = ad_ge

    def evaluate_mapping(self, level:str|None=None, args:dict={}):
        """Run a standard set of mapping evaluations
        """
        print('Running core analysis set..')

        print('1) plot training scores of mapping')
        tg.plot_training_scores(self.ad_map, bins=10, alpha=.5)

        print('2) plot mapping probability at each spatial position for each cluster label')
        self.plot_cell_probability(args=args)

        print('3) get and plot the maximum cluster label for each spatial position')
        # if self.meta.cluster_label == subclass group into excitatory, inhibitory, non-neuronal
        if level is None:
            level = self.meta['cluster_label']
        if level  == 'subclass':
            class_groups = self.ad_sc.obs['class'].unique()
            groups = {group: self.ad_sc.obs[self.ad_sc.obs['class']==(group)]['subclass'].unique().to_list() for group in class_groups}
            self.plot_discrete_mapping_probability(level=level, groups=groups, args=args)
            self.plot_discrete_mapping(level=level, groups=groups, args=args)
        else:
            self.plot_discrete_mapping_probability(level=level, args=args)
            self.plot_discrete_mapping(args=args)

        print('4) predict spatial gene expression, plot canonical markers and histogram of scores')
        self.compare_spatial_gene_exp()

    def plot_cell_probability(self, ncols: int=5, args: dict={}):
        annotation = self.meta['cluster_label']
        nrows = int(np.ceil(len(self.ad_map.obs[annotation]) / ncols))
        tg.plot_cell_annotation(self.ad_map, self.ad_sp, annotation=annotation, nrows=nrows, ncols=ncols, invert_y=False, **args)

    def get_discrete_cell_mapping(self, threshold: float=0, levels: list=['class_label', 'neighborhood_label','subclass_label', 'supertype_label', 'cluster_label']):
        """Return mapped cell type that has the highest probability for each cell. Optionally set a lower probability threshold.
        """
        mapping_level = self.meta['cluster_label']
        if 'tangram_ct_pred' not in self.ad_sp.obsm.keys():
            tg.project_cell_annotations(self.ad_map, self.ad_sp, annotation=mapping_level)
        
        spatial_prob = self.ad_sp.obsm['tangram_ct_pred']
        spatial_prob_norm = (spatial_prob - spatial_prob.min()) / (spatial_prob.max() - spatial_prob.min())
        self.ad_sp.obsm['tangram_ct_pred_norm'] = spatial_prob_norm

        max_prob = spatial_prob_norm.max(axis=1)
        max_anno = spatial_prob_norm.idxmax(axis=1)
        max_anno = max_anno.mask(max_prob < threshold)
        max_prob = max_prob.mask(max_prob < threshold)   
        max_prob.name = mapping_level + '_prob'
        max_anno.name = mapping_level

        discrete_mapping = pd.DataFrame(max_anno)
        discrete_mapping = discrete_mapping.dropna()
        

        discrete_mapping = discrete_mapping.merge(pd.DataFrame(max_prob), left_index=True, right_index=True)
        discrete_mapping = discrete_mapping.merge(self.ad_sp.obs[['center_x', 'center_y']], left_index=True, right_index=True)
        
        # fill in other levels of the hierarchy from known mappings in refence data.
        levels = [l for l in levels if l in self.ad_sc.obs.columns]
        if len(levels)==0:
            print('No other heirarchy levels match reference data. Check self.ad_sc.obs.columns and set levels accordingly')
        else:
            labels = self.ad_sc.obs[levels].drop_duplicates()
            labels.set_index(mapping_level, inplace=True, drop=True)
            discrete_mapping = discrete_mapping.merge(labels, left_on=mapping_level, right_index=True)

        discrete_mapping = discrete_mapping.reindex(self.ad_sp.obs.index)
        self.ad_sp.obsm['discrete_ct_pred'] = discrete_mapping
    
    def plot_discrete_mapping(self, level: str|None=None, groups: dict|None=None, args: dict={}):
        """Plot the discrete mapping produced from get_discrete_cell_mapping at hierarchy level. Optionally provide a dictionary with keys specifying group
        names and keys a list of labels to plot separately for easier visualization. If 'level' is None defaults to self.meta['cluster_label'] 
        """
        # checking for tangram_ct_pred_norm for backwards compatability 
        if 'discrete_ct_pred' not in self.ad_sp.obsm.keys() or 'tangram_ct_pred_norm' not in self.ad_sp.obsm.keys(): 
            self.get_discrete_cell_mapping()

        if level is None:
            level = self.meta['cluster_label']
        
        scatter = {'s': 10, 'alpha': 0.7, 'linewidth': 0}
        scatter.update(args)

        data = self.ad_sp.obsm['discrete_ct_pred']
        if groups is not None:
            fig, ax = plt.subplots(len(groups), 1, figsize=(8, len(groups)*8))
            for i, (group_name, group) in enumerate(groups.items()):
                sns.scatterplot(data=data, x='center_x', y='center_y', ax=ax[i], color='grey', s=3, alpha=0.1, lw=0)
                subdata = data[data[level].isin(group)]
                sns.scatterplot(data=subdata, x='center_x', y='center_y', hue=level, ax=ax[i], hue_order=group, **scatter)
                ax[i].set_title(group_name)
                ax[i].legend(bbox_to_anchor=(1,1))
            
            fig.set_tight_layout(True)
        
        
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.scatterplot(data=data, x='center_x', y='center_y', hue=level,ax=ax, **scatter)
            ax.legend(bbox_to_anchor=(1,1))
        
        return fig

    def plot_discrete_mapping_probability(self, groups: dict|None=None, level: str|None=None, args: dict={}):
        # checking for tangram_ct_pred_norm for backwards compatability 
        if 'discrete_ct_pred' not in self.ad_sp.obsm.keys() or 'tangram_ct_pred_norm' not in self.ad_sp.obsm.keys(): 
            self.get_discrete_cell_mapping()
        
        if level is None:
            level = self.meta['cluster_label']

        violin = {'cut': 0}
        violin.update(args)

        data = self.ad_sp.obsm['discrete_ct_pred']
        if groups is not None:
            fig, ax = plt.subplots(len(groups), 1, figsize=(15, len(groups)*4))
            for i, (group_name, group) in enumerate(groups.items()):
                subdata = data[data[level].isin(group)]
                
                sns.violinplot(data=subdata, x=level, y=self.meta['cluster_label'] + '_prob', ax=ax[i], order=group, **violin)
                ax[i].set_ylabel('Max probability')
                ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation = 45, ha='right')
                ax[i].set_title(group_name)
            
            fig.set_tight_layout(True)
        
        else:
            fig, ax = plt.subplots(figsize=(15, 3))
            sns.violinplot(data=data, x=level, y=self.meta['cluster_label'] + '_prob', ax=ax, sort=True, **violin)
            ax.set_ylabel('Max probability')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha='right')
            
        return fig

    def compare_spatial_gene_exp(self, genes: list=['cux2', 'rorb', 'fezf2', 'lhx6', 'pvalb', 'grik1', 'lamp5', 'gfap', 'opalin', 'mog'], args: dict={}):
        if not hasattr(self, 'ad_ge'):
            self.project_genes()

        genes = [g.lower() for g in genes]
        tg.plot_genes(genes, self.ad_sp, self.ad_ge, invert_y=False, **args)
        spatial_score = tg.compare_spatial_geneexp(self.ad_ge, self.ad_sp, adata_sc=self.ad_sc)
        self.spatial_score = spatial_score
        print(spatial_score[spatial_score.index.isin(genes)])
        
        fig, ax = plt.subplots()
        sns.histplot(spatial_score, x='score', hue='is_training', ax=ax)


class CKMapping(CellTypeMapping):
    """
    Currently for CKmapping mapping occurs on HPC through a separate script. This class loads in results for analysis
    """
    def __init__(self, sp_data: 'AnnData'|str, mapping_result_path: str, meta: dict={}):
        """
        sp_data: AnnData object or filename of AnnData object representing spatial transcriptomics data or None
        mapping_result_path: file path to CK mapping result which should be a .rda file
        """
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        self.run_directory = mapping_result_path
       
        if isinstance(sp_data, str):
            print('loading spatial data...')
            ad_sp = ad.read_h5ad(sp_data)
        else:
            ad_sp = sp_data

        print('loading mapping...')
        r_file = ro.r.load(os.path.join(mapping_result_path, 'mapped.rda'))
        map_results = ro.r['mapped']

        # Extract mapping data from R-object. This returns a list of dataframes indexed as such:
        # [0] map.freq (dataframe)
            # cl     : all clusters a sample is mapped in N iterations of mapping with sub-sampled markers
            # freq : frequencies a sample is mapped to each cluster, cl
            # dist  : distance to cluster template centroid (mean marker gene count)
            # path.cor : correlation of markers along the path of the hierarchy to the terminal node(cluster), in hierarchical mapping
        # [1] best.map.df (dataframe)
            # best.cl  : the cluster a sample is mapped with highest freq in map.freq
            # prob     : probablity of a sample being mapped to best.cl cluster out of  N iterations
            # avg.dist : distance to the template cluster mean
            # avg.path.cor : correlation of markers along the path of the hierarchy to the terminal node(cluster), in hierarchical mapping
            # avg.cor  : correlation to template cluster mean
            # cor.zscore : z-normalized value of over avg.cor
        # [2] cl.df : taxonomy cluster annotation matched by "cl" (mapped[["best.map.df"]]$best.cl) (dataframe)
            # map_freq = pandas2ri.rpy2py(map_results[0])
            # map_freq.set_index('sample_id', inplace=True)
        # [3] all.markers: list of marker genes used for mapping (list)
        # [4] meta: metadata about how mapping was done, created by me not in CK's code (list)
            # Taxonomy = the taxonomy that was used for mapping
            # method = Flat or Hierarchical mapping
            # iterations = number of iterations of mapping

        print('extracting mapping results...')
        map_freq = pandas2ri.rpy2py(map_results[0])
        map_freq.set_index('sample_id', inplace=True)
            
        best_mapping = pandas2ri.rpy2py(map_results[1])
        best_mapping.set_index('sample_id', inplace=True)

        clusters = pandas2ri.rpy2py(map_results[2])
        clusters.set_index('cluster_id', inplace=True)

        # add type labels across the hierarchy from clusters relationship
        labels = [l for l in clusters.columns if l.endswith('label')]
        for label in labels:
            map_freq[label] = map_freq.apply(lambda x: clusters[clusters['cl']==x['cl']][label].iloc[0], axis=1)
            best_mapping[label] = best_mapping.apply(lambda x: clusters[clusters['cl']==x['best.cl']][label].iloc[0], axis=1)

        uns = {}
        if len(map_results) == 5:
            markers = list(map_results[3])
            uns['all.markers'] = markers
            map_meta = list(map_results[4])
            uns['Taxonomy'] = map_meta[0]
            uns['method'] = map_meta[1]
            uns['iterations']  = map_meta[2] if len(map_meta)==3 else None
        elif len(map_results) == 4:
            map_meta = list(map_results[3])
            uns['Taxonomy'] = map_meta[0]
            uns['method'] = map_meta[1]
            uns['iterations']  = map_meta[2] if len(map_meta)==3 else None
        else:
            print("Length of mapping results doesn't match known structures")
        
        uns.update({
                'map.freq': map_freq,
                'cl.df': clusters,
            })

        # Set X to cell x gene reduced to genes used for mapping from all.markers if available otherwise use all genes
        if 'all.markers' in uns.keys(): 
            X = ad_sp[:, uns['all.markers']].to_df()
        else:
            X = ad_sp.to_df()

        # make quasi confusion matrix (cell x celltype prob)
        # AnnData refuses to believe that map_freq.pivot and best_mapping have equal indexes even though
        # they do so hack around to force it so that AnnData doesn't error
    
        obsm_vals = map_freq.pivot(columns='cluster_label', values='freq').fillna(0)
        obsm_index = pd.DataFrame(index=ad_sp.obs.index)
        obsm = obsm_index.merge(obsm_vals, left_index=True, right_index=True)

        ad_map = ad.AnnData(
            X = X,
            obs = ad_sp.obs.merge(best_mapping, left_index=True, right_index=True),
            obsm = {'map_prob_matrix': obsm},
            uns = uns
        )
       
        if len(map_results) == 5:
            markers = list(map_results[3])
            ad_map.uns['all.markers'] = markers
            map_meta = list(map_results[4])
            ad_map.uns['Taxonomy'] = map_meta[0]
            ad_map.uns['method'] = map_meta[1]
            ad_map.uns['iterations']  = map_meta[2] if len(map_meta)==3 else None
        elif len(map_results) == 4:
            map_meta = list(map_results[3])
            ad_map.uns['Taxonomy'] = map_meta[0]
            ad_map.uns['method'] = map_meta[1]
            ad_map.uns['iterations']  = map_meta[2] if len(map_meta)==3 else None
        else:
            print("Length of mapping results doesn't match known structures")

        meta.update({
            'mapping_method': 'CK mapping',
            'flat or heirarchical': ad_map.uns.get('method', 'unknown')
        })
        
        self.ad_sp = ad_sp
        self.ad_map = ad_map
        self.meta = meta

        self.save_mapping(save_path=self.run_directory, file_name='ck_mapping', replace=True)
    
    def plot_mapping_performance(self):
        fig = sns.jointplot(data = self.ad_map.obs, x='avg.cor', y='prob', hue='class_label', alpha=0.2)
        
        return fig

    def plot_best_mapping_probability(self, level: str, groups: dict|None=None, args: dict={}):
    
        violin = {'cut': 0}
        violin.update(args)

        data = self.ad_map.obs

        if groups is not None:
            fig, ax = plt.subplots(len(groups), 1, figsize=(15, len(groups)*4))
            for i, (group_name, group) in enumerate(groups.items()):
                subdata = data[data[level].isin(group)]
                
                sns.violinplot(data=subdata, x=level, y='prob', ax=ax[i], order=group, **violin)
                ax[i].set_ylabel('Probability')
                ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation = 45, ha='right')
                ax[i].set_title(group_name)
            
            fig.set_tight_layout(True)
        
        else:
            fig, ax = plt.subplots(figsize=(15, 3))
            sns.violinplot(data=data, x=level, y= 'prob', ax=ax, sort=True, **violin)
            ax.set_ylabel('Probability')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha='right')
            
        return fig

    def plot_best_mapping_corr(self, level: str, groups: dict|None=None, args: dict={}):

        violin = {'cut': 0}
        violin.update(args)

        data = self.ad_map.obs

        if groups is not None:
            fig, ax = plt.subplots(len(groups), 1, figsize=(15, len(groups)*4))
            for i, (group_name, group) in enumerate(groups.items()):
                subdata = data[data[level].isin(group)]

                sns.violinplot(data=subdata, x=level, y='avg.cor', ax=ax[i], order=group, **violin)
                ax[i].set_ylabel('Avg Correlation')
                ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation = 45, ha='right')
                ax[i].set_title(group_name)

            fig.set_tight_layout(True)

        else:
            fig, ax = plt.subplots(figsize=(15, 3))
            sns.violinplot(data=data, x=level, y= 'avg.cor', ax=ax, sort=True, **violin)
            ax.set_ylabel('Avg Correlation')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha='right')

        return fig

    def plot_best_mapping(self, level: str, groups: dict|None=None, args: dict={}):
            """Plot the best mapping at particular heirarchy level. Optionally provide a dictionary with keys specifying group
            names and keys a list of labels to plot separately for easier visualization.  
            """
            scatter = {'s': 10, 'alpha': 0.7, 'linewidth': 0}
            scatter.update(args)

            data = self.ad_map.obs

            if groups is not None:
                fig, ax = plt.subplots(len(groups), 1, figsize=(8, len(groups)*6))
                for i, (group_name, group) in enumerate(groups.items()):
                    sns.scatterplot(data=data, x='center_x', y='center_y', ax=ax[i], color='grey', s=3, alpha=0.1, lw=0)
                    subdata = data[data[level].isin(group)]
                    sns.scatterplot(data=subdata, x='center_x', y='center_y', hue=level, ax=ax[i], hue_order=group, **scatter)
                    ax[i].set_title(group_name)
                    ax[i].legend(bbox_to_anchor=(1,1))
                
                fig.set_tight_layout(True)
            
            
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.scatterplot(data=data, x='center_x', y='center_y', hue=level,ax=ax, **scatter)
                ax.legend(bbox_to_anchor=(1,1))
            
            return fig

class ScrattchMapping(CellTypeMapping):
    def __init__(self, sp_data: 'AnnData'|str, taxonomy_path: str, meta: dict={}):
        """
        sp_data: AnnData object or filename of AnnData object representing spatial transcriptomics data or None
        taxonomy_path: file path to where taxonomy lives on isolon
        sc_data: AnnData object or filename of AnnData object representing reference RNAseq data
        """
           
        if not isinstance(sp_data, ad.AnnData):
            print(f'loading spatial data from {sp_data}...')
            ad_sp = ad.read_h5ad(sp_data)
        else:
            ad_sp = sp_data

        meta.update({'mapping_method': 'scrattch mapping',
                     'taxonomy': taxonomy_path,
                      'sp_data_uns': str(dict(ad_sp.uns)),
                    })

        self.ad_sp = ad_sp
        self.taxonomy_path = taxonomy_path
        self.run_directory = None
        self.meta = meta

    def run_on_hpc(self, ad_map_args:dict={}, hpc_args:dict={}, docker: str='singularity exec --cleanenv docker://bicore/scrattch_mapping:latest', r_script: str='scrattch_mapping.R'):

        self.make_mapping_anndata(**ad_map_args)

        print('building HPC job')
        job_path = hpc_args.get('job_path', None)
        if job_path is None:
            print('must specify a job path')
            return
        hpc_default = {
            'hpc_host': 'hpc-login',
            'job_path': job_path,
            'partition': 'celltypes',
            'job_name': 'scrattch-mapping',
            'nodes': 1,
            'mincpus': 10,
            'mem': '300G',
            'time':'1:00:00',
            'mail_user': None,
            'output': job_path + 'hpc_logs/%j.out',
            'error': job_path + 'hpc_logs/%j.err',
        }

        hpc_default.update(hpc_args)
        
        dat_path = self.run_directory.replace('\\', '/') # make sure format is correct for linux

        command = f"""
                cd {job_path}

                {docker} Rscript {r_script} --dat_path {dat_path}
        """
        
        assert 'command' not in hpc_args
        hpc_default['command'] = command
        job = run_slurm(**hpc_default)
        print(job.state(), job.job_id)
        return job
    
    def make_mapping_anndata(self, save_path: str|None=None, ad_sp_layer: str|None=None, training_genes: list|None=None, cell_qc: str|None=None, meta: dict={}):
        """
        save_path: where to save out mapping anndata object to load into R script
        ad_sp_layer: scrattch_mapping requires log normalized data, identify where in the ad_sp object
                    this data resides. If None it will be assumed the data is ad_sp.X otherwise identify
                    key for ad_sp.layers
        training_genes: list of genes to use for mapping, if None will use all genes in ad_sp.var_names
        cell_qc: column to use in ad_sp.obs to filter cells for mapping. Values in column must be boolean. If None all cells will be used
        """
        if training_genes is not None:
            genes = [tg for tg in training_genes if tg in self.ad_sp.var_names]
            if len(genes) == 0:
                print('No genes overlap between training_genes and ad_sp.var_names')
                return
            ad_map = self.ad_sp[:, genes].copy()
        else:
            # remove blanks if they exist
            try:
                ad_map = self.ad_sp[:, self.ad_sp.var[self.ad_sp.var['probe_type']=='gene'].index.to_list()].copy()
            except KeyError:
                print('probe type not set in ad_sp.var, assuming only genes in ad_sp.var...')
                ad_map = self.ad_sp.copy()

        # scrattch_map catches if the gene names are only listed in the index. Make them a column
        if ad_map.var.shape[1]==0:
            ad_map.var['gene'] = ad_map.var_names

        if ad_sp_layer is not None:
            ad_map.layers['raw_counts'] = ad_map.X
            ad_map.X = ad_map.layers[ad_sp_layer]
            self.meta['counts'] = ad_sp_layer
        else:
            self.meta['counts'] = 'raw'

        # Script doesn't work if AnnData contains sparse matrices, so convert 
        if isinstance(ad_map.X, sparse.spmatrix):
            ad_map.X = ad_map.X.toarray()
        
        # All layers must be converted, not just X
        for l_name in ad_map.layers.keys():
            if isinstance(ad_map.layers[l_name], sparse.spmatrix):
                ad_map.layers[l_name] = ad_map.layers[l_name].toarray()

        if cell_qc is not None:
            ad_map = ad_map[ad_map.obs[ad_map.obs[cell_qc]].index.to_list(), :]

        ad_map.uns['taxonomy_path'] = self.taxonomy_path
        if 'taxonomy_name' in self.meta.keys():
            ad_map.uns['taxonomy_name'] = self.meta['taxonomy_name']
        if 'taxonomy_cols' in self.meta.keys():
            ad_map.uns['taxonomy_cols'] = [prefix + 'label' for prefix in self.meta['taxonomy_cols']]
        
        self.meta.update(meta)
        self.ad_map = ad_map
        if save_path is None and self.run_directory is None:
            print('Must set a save path')
        if save_path is not None:
            print(f'Setting run directory to {save_path}')
            self._create_run_directory(save_path)

        ad_map.write_h5ad(os.path.join(self.run_directory, 'scrattch_map_temp.h5ad'))

    def load_scrattch_mapping_results(self):
        # use to load scrattch mapping results back in for analysis
        # results will be saved in obsm field of anndata in R script
        print('loading results...')
        results_file = os.path.join(self.run_directory, 'scrattch_map_temp.h5ad')
        if os.path.exists(results_file) is False:
            print('No scrattch_map_temp.h5ad file found in run directory, checking for previously saved results...')
            try:
                mapping = ScrattchMapping.load_from_timestamp(os.path.dirname(self.run_directory), timestamp=os.path.basename(self.run_directory))
                self = mapping
            except:
                pass
        else:
            scrattch_map_results = ad.read_h5ad(results_file)
            if 'mapping_results' not in scrattch_map_results.obsm.keys():
                print('No mapping_results found check hpc logs for errors')
                return
            self.ad_map.obs = self.ad_map.obs.merge(scrattch_map_results.obsm['mapping_results'], left_index=True, right_index=True, suffixes=('_drop', ''))
            self.ad_map.obs.drop([col for col in self.ad_map.obs.columns if 'drop' in col], axis=1, inplace=True)
            self.ad_map.uns.update(scrattch_map_results.uns)
            self.save_mapping(save_path=self.run_directory, replace=True)
            # delete saved out anndata file
            print('deleting scrattch-mapping temp file...')
            os.remove(os.path.join(self.run_directory, 'scrattch_map_temp.h5ad'))
    
    def load_taxonomy_anndata(self):
        # the taxonomy anndatas are pretty big, only load if necessary and backed so no changes inadvertantly get made. 
        taxonomy_files = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(self.taxonomy_path) for filename in filenames if filename.endswith('taxonomy.h5ad')]
        if len(taxonomy_files) == 1:
            self.ad_sc = ad.read_h5ad(taxonomy_files[0], backed='r')
        elif len(taxonomy_files)==0:
            print(f'No taxonomy.h5ad file found that ends in path: {self.taxonomy_path}')
        else:
            print(f'Multiple taxonomy.h5ad files found:')
            _ = [print(f'{file}') for file in taxonomy_files]

    def plot_best_mapping_corr(self, level: str, groups: dict|None=None, args: dict={}, qc_pass=True):

        violin = {'cut': 0}
        violin.update(args)

        if qc_pass is True and 'mapping_qc' in self.ad_map.obs.columns:
            data = self.ad_map.obs[self.ad_map.obs['mapping_qc']=='pass']
        else:
            data = self.ad_map.obs

        if groups is not None:
            fig, ax = plt.subplots(len(groups), 1, figsize=(15, len(groups)*4))
            for i, (group_name, group) in enumerate(groups.items()):
                subdata = data[data[level].isin(group)]

                sns.violinplot(data=subdata, x=level, y='score.Corr', ax=ax[i], hue=level, order=group, **violin)
                ax[i].set_ylabel('Avg Correlation')
                ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation = 45, ha='right')
                ax[i].set_title(group_name)
                ax[i].set_aspect('equal', adjustable='box', anchor='C')

            fig.set_tight_layout(True)

        else:
            fig, ax = plt.subplots(figsize=(15, 3))
            sns.violinplot(data=data, x=level, y= 'score.Corr', ax=ax, **violin)
            ax.set_ylabel('Avg Correlation')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha='right')

        return fig

    def plot_best_mapping(self, level: str, groups: dict|None=None, args: dict={}, qc_pass=True):
            """Plot the best mapping at particular heirarchy level. Optionally provide a dictionary with keys specifying group
            names and keys a list of labels to plot separately for easier visualization.  
            """
            scatter = {'s': 10, 'alpha': 0.7, 'linewidth': 0}
            scatter.update(args)

            if qc_pass is True and 'mapping_qc' in self.ad_map.obs.columns:
                data = self.ad_map.obs[self.ad_map.obs['mapping_qc']=='pass']
            else:
                data = self.ad_map.obs

            if groups is not None:
                fig, ax = plt.subplots(len(groups), 1, figsize=(8, len(groups)*6))
                for i, (group_name, group) in enumerate(groups.items()):
                    sns.scatterplot(data=data, x='center_x', y='center_y', ax=ax[i], color='grey', s=3, alpha=0.1, lw=0)
                    subdata = data[data[level].isin(group)]
                    sns.scatterplot(data=subdata, x='center_x', y='center_y', hue=level, ax=ax[i], hue_order=group, **scatter)
                    ax[i].set_title(group_name)
                    ax[i].legend(bbox_to_anchor=(1,1))
                
                fig.set_tight_layout(True)
            
            
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.scatterplot(data=data, x='center_x', y='center_y', hue=level,ax=ax, **scatter)
                ax.legend(bbox_to_anchor=(1,1))
            
            return fig

    def plot_class_proportions(self, level, groups=None, args={}, qc_pass=True):

        if qc_pass is True and 'mapping_qc' in self.ad_map.obs.columns:
            data = self.ad_map.obs[self.ad_map.obs['mapping_qc']=='pass']
        else:
            data = self.ad_map.obs
        
        if groups is not None:
            cls_prop = None
            for group_name, group in groups.items():
                subdata = data[data[level].isin(group)]
                total_count = len(subdata)
                level_counts = subdata.groupby(level).count()
                level_counts['Prop of Class'] = level_counts.apply(lambda x: x['score.Corr'] / total_count, axis=1)
                level_counts['Class'] = group_name

                if cls_prop is None:
                    cls_prop = level_counts[['Prop of Class', 'Class']]
                else:
                    cls_prop = cls_prop.append(level_counts[['Prop of Class', 'Class']])
            pivot = cls_prop.pivot_table(index=cls_prop.index, columns='Class', values='Prop of Class')
            pivot = pivot.loc[cls_prop.index]
            fig, ax = plt.subplots(figsize=(5,5))
            _ = pivot.T.plot.bar(stacked=True, ax=ax, **args)
            ax.set_ylabel('Proportion of Class')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            total_count = len(data)
            level_counts = data.groupby(level).count()
            level_counts['Prop of Cells'] = level_counts.apply(lambda x: x['score.Corr'] / total_count, axis=1)
            fig, ax = plt.subplots(figsize=(5,5))
            _ = level_counts['Prop of Cells'].T.plot.bar(ax=ax, **args)
            ax.set_ylabel('Proportion of Cells')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        return fig
        
    def get_confusion_matrix(self, pivot_cols, norm=True):
        from sis.expression_dataset import norm_confusion_matrix
        if type(pivot_cols)==dict:
            conf_dict = {}
            for level, pivot in pivot_cols.items():
                conf_matrix_level = self.ad_map.obs[pivot].pivot_table(columns=pivot[0], index=pivot[1], aggfunc=len, margins=True)
                conf_matrix_level = conf_matrix_level.loc[conf_matrix_level.columns]
                conf_matrix_level.index.rename('Mapped', inplace=True)
                conf_matrix_level.columns.name = 'Annotated'
                conf_dict[level] = conf_matrix_level
            conf_matrix = pd.concat(conf_dict, axis=1)
        else:
                conf_matrix = self.ad_map.obs[pivot_cols].pivot_table(columns=pivot_cols[0], index=pivot_cols[1], aggfunc=len, margins=True)
                conf_matrix = conf_matrix.loc[conf_matrix.columns]
                conf_matrix.index.rename('Mapped', inplace=True)
                conf_matrix.columns.name = 'Annotated'
        
        if norm is True:
            norm_conf_matrix = norm_confusion_matrix(conf_matrix)
            return norm_conf_matrix
        else:
            return conf_matrix
    

