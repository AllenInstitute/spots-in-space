from __future__ import annotations
from lib2to3.pgen2 import driver
import pandas as pd
# from code.gene_panel_selection.gene_panel_selection import ExpressionDataset
# from code.segmentation.segmentation import SpotTable
import numpy as np
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle

try:
    import tangram as tg
except ImportError:
    pass

class CellTypeMapping():
    """Container class for cell type mapping methods
    """
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
            ad_sc = sc.read_h5ad(sc_data)
        else:
            ad_sc = sc_data
        if isinstance(sp_data, str):
            ad_sp = sc.read_h5ad(sp_data)
        else:
            ad_sp = sp_data

        # library size correction, normalize count within each cell to fixed number
        sc.pp.normalize_total(ad_sc)

        self.ad_sc = ad_sc
        self.ad_sp = ad_sp
        self.meta = meta
        
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

    def evaluate_mapping(self, args:dict={}):
        """Run a standard set of mapping evaluations
        """
        print('Running core analysis set..')

        print('1) plot training scores of mapping')
        tg.plot_training_scores(self.ad_map, bins=10, alpha=.5)

        print('2) plot mapping probability at each spatial position for each cluster label')
        self.plot_cell_probability(args=args)

        print('3) get and plot the maximum cluster label for each spatial position')
        # if self.meta.cluster_label == subclass group into excitatory, inhibitory, non-neuronal
        if self.meta['cluster_label'] == 'subclass':
            class_groups = self.ad_sc.obs['class'].unique()
            groups = {group: self.ad_sc.obs[self.ad_sc.obs['class']==(group)]['subclass'].unique().to_list() for group in class_groups}
            self.plot_discrete_mapping(groups=groups, args=args)
        else:
            self.plot_discrete_mapping(args=args)

        print('4) predict spatial gene expression, plot canonical markers and histogram of scores')
        self.compare_spatial_gene_exp()

    def plot_cell_probability(self, ncols: int=5, args: dict={}):
        annotation = self.meta['cluster_label']
        nrows = int(np.ceil(len(self.ad_map.obs[annotation]) / ncols))
        tg.plot_cell_annotation(self.ad_map, self.ad_sp, annotation=annotation, nrows=nrows, ncols=ncols, invert_y=False, **args)

    def get_discrete_cell_mapping(self, threshold: float=0):
        """Return mapped cell type that has the highest proportion for each cell. Optionally set a lower probability threshold.
        """
        if 'tangram_ct_pred' not in self.ad_sp.obsm.keys():
            tg.project_cell_annotations(self.ad_map, self.ad_sp, annotation=self.meta['cluster_label'])
        
        spatial_prob = self.ad_sp.obsm['tangram_ct_pred']
        spatial_prob_norm = (spatial_prob - spatial_prob.min()) / (spatial_prob.max() - spatial_prob.min())
       
        max_prop = spatial_prob_norm.max(axis=1)
        max_anno = spatial_prob_norm.idxmax(axis=1)
        max_anno = max_anno.mask(max_prop < threshold)
        max_prop = max_prop.mask(max_prop < threshold)   
        max_prop.name = 'cluster_prop'
        max_anno.name = 'cluster'

        spatial_prob_norm = spatial_prob_norm.merge(max_anno, left_index=True, right_index=True)
        spatial_prob_norm = spatial_prob_norm.merge(max_prop, left_index=True, right_index=True)
        spatial_prob_norm = spatial_prob_norm.merge(self.ad_sp.obs[['x', 'y']], left_index=True, right_index=True)
        self.ad_sp.obsm['discrete_ct_pred'] = spatial_prob_norm
    
    def plot_discrete_mapping(self, groups: dict|None=None, args: dict={}):
        """Plot the discrete mapping produced from get_discrete_cell_mapping. Optionally provide a dictionary with keys specifying group
        names and keys a list of labels in self.meta['cluster_label'] to plot separately for easier visualization.
        """
        if 'discrete_ct_pred' not in self.ad_sp.obsm.keys():
            self.get_discrete_cell_mapping()
        
        violin = {'cut': 0}
        violin.update(args)
        scatter = {'s': 10, 'alpha': 0.7, 'lw': 0}
        scatter.update(args)

        data = self.ad_sp.obsm['discrete_ct_pred']
        if groups is not None:
            fig1, ax1 = plt.subplots(len(groups), 1, figsize=(15, len(groups)*4))
            fig2, ax2 = plt.subplots(len(groups), 1, figsize=(8, len(groups)*8))
            for i, (group_name, group) in enumerate(groups.items()):
                subdata = data[data['cluster'].isin(group)]
                
                sns.violinplot(data=subdata, x='cluster', y='cluster_prop', ax=ax1[i], sort=True, **violin)
                ax1[i].set_ylabel('Max probability')
                ax1[i].tick_params(axis='x', labelrotation = 45)
                ax1[i].set_title(group_name)

                sns.scatterplot(data=subdata, x='x', y='y', hue='cluster', ax=ax2[i], hue_order=group, **scatter)
                ax2[i].set_title(group_name)
            
            fig1.set_tight_layout(True)
        
        else:
            fig1, ax1 = plt.subplots(figsize=(15, 3))
            sns.violinplot(data=data, x='cluster', y='cluster_prop', ax=ax1, sort=True, **violin)
            ax1.set_ylabel('Max probability')
            ax1.tick_params(axis='x', labelrotation = 45)
            
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            sns.scatterplot(data=data, x='x', y='y', hue='cluster',ax=ax2, **scatter)

    
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

    def save_mapping(self, save_path: str, file_name:str='tangram_mapping', meta:dict={}, replace=False):

        if self.run_directory is None or replace is False:
            dir_ts = '{:0.3f}'.format(datetime.now().timestamp())
            self.run_directory = os.path.join(save_path, dir_ts)
            os.mkdir(self.run_directory)
        with bz2.BZ2File(os.path.join(self.run_directory, file_name + '.pbz2'), 'w') as f: 
            cPickle.dump(self, f)
        print(f'analysis UID: {os.path.basename(self.run_directory)}')

    @classmethod
    def load_from_timestamp(cls, directory: str, timestamp: str, file_name: str='tangram_mapping'):
        path = os.path.join(directory, timestamp, file_name + '.pbz2')
        if os.path.exists(path) is False:
            print(f'tangram analysis file {file_name} does not exist for timestamp {timestamp} in directory {directory}')
            return
        tg_map = bz2.BZ2File(path, 'rb')
        tg_map = cPickle.load(tg_map)

        return tg_map
    

def convert_to_anndata(sc_data: ExpressionDataset|None=None, sp_data: SpotTable|None=None, binsize:int|None=None, 
                       annotation_levels:dict={'cluster': 'cluster_label', 'subclass': 'subclass_label', 'class':'class_label'}):
    """Convert RNAseq and/or spatial data to AnnData format. 
    Parameters:
    -----------
    sc_data: ExpressionDataset|None
        ExpressionDataset containing RNAseq data formatted with cells as rows and genes as columns
    sp_data: SpotTable|None
        SpotTable describing spatial transcriptomics data with x, y position
    binsize: int|None
        bin size to use for creating "cell" by gene table of spatial data
    annotation_levels: dict
        annotations to collect from rnaSeq data. Keys are label you want for the new AnnData structure, values are labels from
        ExpressionDataset.annotation_data columns
    """
    import anndata as ad
    
    ad_sc = None
    ad_sp = None
    
    if sc_data is not None:
        exp_matrix = sc_data.expression_data.to_numpy(dtype='float64')
        ad_sc = ad.AnnData(exp_matrix)
        cells = sc_data.expression_data.index.to_numpy(dtype='object')
        genes = sc_data.expression_data.columns.to_numpy(dtype='object')
        
        ad_sc.obs_names = cells
        ad_sc.var_names = genes
        
        for label1, label2 in annotation_levels.items():
            ad_sc.obs[label1] = sc_data.annotation_data[label2]
        
    if sp_data is not None and binsize is not None:
        bin_by_gene, xys = bin_gene_table(sp_data, binsize=binsize)
        
        ad_sp = ad.AnnData(bin_by_gene, dtype=float)
        bin_names = [f'bin:{int(x)}_{int(y)}' for x, y in xys]
        genes = sp_data.map_gene_ids_to_names(np.unique(sp_data.gene_ids))
        
        ad_sp.obs_names = bin_names
        ad_sp.var_names = genes
        ad_sp.obs['x'] = xys[:, 0]
        ad_sp.obs['y'] = xys[:, 1]
        
    return ad_sc, ad_sp

def bin_gene_table(table, binsize:int =100):
    from tqdm.notebook import tqdm
    """Construct bin by gene table from SpotTable at binsize resolution
    """

    x = table.x
    y = table.y
    gene = table.gene_ids

    xrange = x.min(), x.max()
    yrange = y.min(), y.max()
    gene_inds = len(table.gene_id_to_name)

    x_bins = int(np.ceil((xrange[1] - xrange[0]) / binsize))
    y_bins = int(np.ceil((yrange[1] - yrange[0]) / binsize))
    n_cells = (x_bins*y_bins)

    bin_by_gene = np.zeros((n_cells, gene_inds), dtype='uint32')
    bins = np.empty((n_cells, 2))
    row = 0
    print(f'Binning SpotTable at bin size {binsize}..')
    for i in tqdm(range(x_bins)):
        xbin = (xrange[0] + i * binsize), (xrange[0] + (i+1) * binsize)
        xmask = (x > xbin[0]) & (x < xbin[1])
        y2 = y[xmask]
        g2 = gene[xmask]
        for j in range(y_bins):
            ybin = (yrange[0] + j * binsize), (yrange[0] + (j+1) * binsize)
            ymask = (y2 > ybin[0]) & (y2 < ybin[1])
            g3 = g2[ymask]
            bins[row] = [xbin[0], ybin[0]]
            for g, c in zip(*np.unique(g3, return_counts=True)):
                bin_by_gene[row][g] = c
            row += 1    

    mask = np.where(bin_by_gene.any(axis=1))[0]
    bin_gene = bin_by_gene[mask]
    bins2 = bins[mask]

    return bin_gene, bins2
