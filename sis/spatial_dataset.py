from __future__ import annotations

## Class SpatialDataset to organize spatial datasets from variety of platforms
## includes subclasses for each platform to encompass data processing steps
## use to version data processing
## These classes can just store pointers to segmentation, mapping, etc and not copies of the data itself. This will allow the tools already
## built to still access those files without having to load the whole class which could get burdensome
## utilizes new Allen Services for data grabbing 
## spatial_config.yml to store datapaths and other configurable properties


from sis.celltype_mapping import ScrattchMapping, CellTypeMapping
from sis.hpc import run_slurm_func
from sis.qc import run_doublet_detection, calc_n_transcripts, calc_n_genes, calc_n_blanks
from sis.segmentation import MerscopeSegmentationRun
from sis.spot_table import SpotTable
from sis.util import load_config
import anndata as ad
import os, sys, datetime, glob, pickle, time
import json
from abc import abstractmethod
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from contextlib import redirect_stdout
from .optional_import import optional_import

widgets = optional_import('ipywidgets')
interact_manual = optional_import('ipywidgets', names='interact_manual')


## Allen Services
pts_schema = optional_import('allen_services_api.pts.schema.pts_schema', names= 'pts_schema')
DatsClient = optional_import('allen_services_api.dats.client.dats_client', names=  "DatsClient")
PtsClient = optional_import('allen_services_api.pts.client.pts_client', names=  "PtsClient")
BersClient = optional_import( 'allen_services_api.bers.client.bers_client', names =  "BersClient")


SPATIALDATASET_VERSION = 2


class SpatialDataset:
    # wrapper class
    def __init__(self, barcode):
        self.barcode = barcode
        self.version = SPATIALDATASET_VERSION
        self.config = load_config()
        
    @classmethod
    def load_from_barcode(cls, barcode: str, dataset_type: 'SpatialDataset', file_name='spatial_dataset'):
        config = load_config()
        if dataset_type == MERSCOPESection:
            data_path = Path(config['merscope_save_path']).joinpath(str(barcode), file_name)
        if dataset_type == StereoSeqSection:
            data_path = Path(config['stereoseq_save_path']).joinpath(str(barcode), file_name)
        # other config paths would be added here as they come about
        if not data_path.is_file():
            print(f'SpatialDataset {data_path} does not exist')
            return None
        with open(data_path, 'rb') as file:
            dataset = pkl.load(file)
            file.close()
            dataset._convert_str_to_paths()
            print(f'SpatialDataset {data_path} loaded...')
            if dataset.version != SPATIALDATASET_VERSION:
                print(f'Warning: SpatialDataset version {dataset.version} does not match current version {SPATIALDATASET_VERSION}')
        return dataset

    def save_dataset(self, file_name: str='spatial_dataset'):
        self._convert_paths_to_str()
        save_path = Path(self.save_path)
        file = open(save_path.joinpath(file_name), 'wb')
        pkl.dump(self, file)
        file.close()
        # convert back to paths to continue using the object (bit hacky)
        self._convert_str_to_paths()

    def _get_path_attrs(self):
        path_attrs = [k for k, v in self.__dict__.items() if k.endswith('_path') or k.endswith('_file') or k.endswith('_cache')]
        return path_attrs

    def _convert_paths_to_str(self):
        path_attrs = self._get_path_attrs()
        for attr in path_attrs:
            path = getattr(self, attr)
            if isinstance(path, str):
                continue
            assert isinstance(path, Path)
            if path.is_file():
                setattr(self, attr, path.as_posix())
            elif path.is_dir():
                setattr(self, attr, f'{path.as_posix()}/')

    def _convert_str_to_paths(self):
        path_attrs = self._get_path_attrs()
        for attr in path_attrs:
            path = getattr(self, attr)
            assert isinstance(path, str)
            setattr(self, attr, Path(path))
 
    def get_analysis_status(self):
        # check for segmentation and mapping files
        if hasattr(self, 'corr_to_bulk'):
            print(f'Correlation to bulk already calculated: {self.corr_to_bulk:.2f}')
        else:
            print('Correlation to bulk not calculated')

        seg_status = self.check_segmentation_status()
        if seg_status is None:
            print(f'No segmentations available in {self.segmentation_path}')
        else:
            print('Segmentation timestamps: \t Segmentation info')
            for ts, info in seg_status.items():
                print(f'{ts}: \t {info}')

        mappings = self.get_mappings(print_output=False)
        if len(mappings) == 0:
            print('No mappings found')
        else:
            print('Mapping timestamps: \t Mapping info')
            for ts, info in mappings.items():
                print(f'{ts}: \t {info}')

    def check_segmentation_status(self):
        # Get all subdirectories in segmentation directory
        # Each should correspond to a segmentation run... unless there are
        # weird random folders that shouldn't be there...
        segmentation_path = self.segmentation_path
        seg_runs = [f.name for f in os.scandir(segmentation_path) if f.is_dir()]
        if len(seg_runs) == 0:
            status = None

        elif len(seg_runs) > 0:
            status = {}
            for run_name in seg_runs:
                run_path = os.path.join(segmentation_path, run_name)
                final_output = os.path.join(run_path, 'cell_by_gene.h5ad')
                seg_info = {'finished': os.path.exists(final_output)}
                
                try:
                    seg_config_file = os.path.join(run_path, 'seg_meta.json')
                    with open(seg_config_file, 'r') as f:
                        seg_config = json.load(f)

                    seg_info.update({k: seg_config[k] for k in ['subrgn', 'seg_method', 'seg_opts']})

                except FileNotFoundError:
                    pass

                status[run_name] = seg_info

        return status

    def get_mappings(self, print_output=True):
        mappings = {}
        mapping_dir = hasattr(self, 'mapping_path')
        if mapping_dir is False:
            print('No mapping directory specified')
            return mappings
        mapping_dir = self.mapping_path
        for mapping in mapping_dir.iterdir():
            ts = mapping.name
            try:
                with redirect_stdout(None):
                    meta = CellTypeMapping.load_from_timestamp(directory=mapping_dir, timestamp=ts, meta_only=True)
                mappings[ts] = meta
                if print_output is True:
                    print(f'{ts}: {meta}')
            except:
                mappings[ts] = None
                if print_output is True:
                    print(f'{ts}: None')
        
        return mappings
        
    def spatial_corr_to_bulk(self, spot_table=None, save_fig=True):
        if spot_table is None:
            spot_table = self.load_spottable()
        
        gene_ids, total_counts = np.unique(spot_table.gene_ids, return_counts=True)
        genes = spot_table.map_gene_ids_to_names(gene_ids)

        total_counts = pd.DataFrame({'gene': genes, 'spatial total counts': total_counts})
        total_counts.set_index('gene', inplace=True)
        
        bulk_data = pd.read_csv(self.config['bulkseq_file'])
        bulk_data.set_index('Gene Name', inplace=True)
        
        if self.broad_region is not None and self.broad_region in bulk_data['Broad Region'].unique():
            bulk_data_region = bulk_data[bulk_data['Broad Region']==self.broad_region]
        else:
            print(f'Section Broad Region ({self.broad_region}) does not match any in bulk data reference ({bulk_data["Broad Region"].unique()}). \
                Using all reference bulk sequencing data for spatial correlation.')
            bulk_data_region = bulk_data
                  
        total_counts = total_counts.merge(bulk_data_region['FPKM'], left_index=True, right_index=True, how='inner')
        for col in total_counts.columns:
            total_counts[col + ' log'] = np.log(total_counts[col])

        total_counts_corr = total_counts.corr()['spatial total counts log']['FPKM log']
        self.corr_to_bulk = total_counts_corr

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        reduced_spots = int(np.floor(len(spot_table)/100000))
        spot_table[::reduced_spots].scatter_plot(ax[0])
        sns.scatterplot(data=total_counts, x='spatial total counts', y='FPKM', alpha=0.5, lw=0, ax=ax[1])
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_title(f'Pearson corr: {total_counts_corr:0.3f}')
        plt.tight_layout()

        if save_fig is True:
            fig.savefig(self.save_path.joinpath('correlation_to_bulk.png'))
        
        self.save_dataset()

    def show_gene_transcripts(self, genes: list, subregions=None, spot_table=None, save_fig=True):
        if spot_table is None:
            spot_table = self.load_spottable()
            
        fig_rows = len(subregions) if subregions is not None else 1
        fig, ax = plt.subplots(fig_rows, 2, figsize=(10, 5*fig_rows))
        
        gene_ids = spot_table.map_gene_names_to_ids(genes)
        genes_table = spot_table.get_genes(gene_ids=gene_ids)
        reduced_spots = int(np.floor(len(spot_table)/100000))
        if fig_rows==1:
            spot_table[::reduced_spots].scatter_plot(ax[0])
            spot_table.gene_names
            df = genes_table.dataframe(cols=['x', 'y', 'gene_names'])
            sns.scatterplot(data=df, x='x', y='y', hue='gene_names', s=3, alpha=0.5, linewidth=0, ax=ax[1], palette='tab10', hue_order=genes)
            ax[1].legend(bbox_to_anchor=(1, 1))
        else:  
            for i, subregion in enumerate(subregions):
                spot_table[::reduced_spots].scatter_plot(ax[i, 0])
                sub_table = genes_table.get_subregion(xlim=subregion[0], ylim=subregion[1])
                sub_table.plot_rect(ax[i, 0], 'r')
                sub_table.gene_names
                df = sub_table.dataframe(cols=['x', 'y', 'gene_names'])
                sns.scatterplot(data=df, x='x', y='y', hue='gene_names', s=3, alpha=0.5, linewidth=0, ax=ax[i, 1], palette='tab10', hue_order=genes)
                ax[i, 1].legend(bbox_to_anchor=(1, 1))
                
        if save_fig is True:
            fig.savefig(self.save_path.joinpath('marker_gene_transcripts.png'))

    def view_mapping_results(self, ct_map, ct_level, score_thresh=None):
        if type(ct_map) == str:
            ct_map = CellTypeMapping.load_from_timestamp(directory=self.mapping_path, timestamp=ct_map)

        if isinstance(ct_map, ScrattchMapping):
            mapping_score_col = 'score.Corr'
            col_labels = [prefix + 'Corr' for prefix in self.config['taxonomy_info'][ct_map.meta['taxonomy_name']]['col_labels']]
            if mapping_score_col not in ct_map.ad_map.obs.columns:
                ct_map.load_scrattch_mapping_results()

        fig, ax = plt.subplots()
        sns.histplot(data=ct_map.ad_map.obs, x=mapping_score_col, ax=ax)
        if score_thresh is not None:
            qc_pass = True
            ct_map.qc_mapping(qc_params={mapping_score_col: score_thresh})
            ax.axvline(score_thresh, color='k', ls='--')
        else:
            qc_pass = False
        fig.savefig(Path(ct_map.run_directory).joinpath('score_hist.png'))

        cls_label = [label for label in col_labels if 'class' in label][0]
        ct_level_label = [label for label in col_labels if ct_level in label]
        if len(ct_level_label) != 1:
            print(f'{ct_level} does not have an obvious corresponding column in mapping output, check obs columns')
            return
        ct_level_label = ct_level_label[0]
        groups= {}
        for cls in ct_map.ad_map.obs[cls_label].unique():
            group = ct_map.ad_map.obs[ct_map.ad_map.obs[cls_label]==cls][ct_level_label].unique()
            groups[cls] = group

        fig = ct_map.plot_best_mapping_corr(level=ct_level_label, groups=groups, qc_pass=qc_pass, args={'scale': 'width'})
        fig.savefig(Path(ct_map.run_directory).joinpath(f'score_corr_{ct_level}.png'))

        fig = ct_map.plot_best_mapping(level=ct_level_label, groups=groups, qc_pass=qc_pass, args={'s': 5})
        fig.savefig(Path(ct_map.run_directory).joinpath(f'{ct_level}_spatial_map.png'))

class MERSCOPESection(SpatialDataset):


    def __init__(self, barcode):
        pts_qc_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="QCMetadata")))
        pts_request_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="MerscopeImagingRequestMetadata")))

        config = load_config()
        save_path = Path(config['merscope_save_path']).joinpath(str(barcode))
        if os.path.exists(save_path.joinpath('spatial_dataset')):
            print(f'SpatialDataset already exists and will be loaded. If you want to reprocess this dataset delete the file and start over')
            cached = SpatialDataset.load_from_barcode(barcode, MERSCOPESection)
            self.__dict__ = cached.__dict__
            self.get_analysis_status()
        
        else:
            SpatialDataset.__init__(self, barcode)
            self.save_path = save_path
            self.save_path.mkdir(exist_ok=True)
            
            self.mapping_path = self.save_path.joinpath(self.config['mapping_dir']) if self.config.get('mapping_dir') is not None else None
            if self.mapping_path is not None and not self.mapping_path.exists():
                self.mapping_path.mkdir()
            self.segmentation_path = self.save_path.joinpath(self.config['segmentation_dir']) if self.config.get('segmentation_dir') is not None else None
            if self.segmentation_path is not None and not self.segmentation_path.exists():
                self.segmentation_path.mkdir()
            self.qc_path = self.save_path.joinpath(self.config['qc_dir']) if self.config.get('qc_dir') is not None else None
            if self.qc_path is not None and not self.qc_path.exists():
                self.qc_path.mkdir()

            self.broad_region = None
            self.get_section_metadata()
            self.get_section_data_paths()
            
            self.save_dataset()

    def get_section_metadata(self):
        bers = BersClient()
        pts = PtsClient()
        
        bers_spec = bers.get_specimen_by_lims2_id(lims2_id = int(self.barcode))
        assert bers_spec.specimen_type.label == 'brain specimen section', f'not a LIMS2 Section specimen {bers_spec.name}'
        
        self.bers_id = bers_spec.id
        self.species = bers_spec.species
        self.lims_specimen_name = bers_spec.name
#         self.roi  # this could come from multiple places, LIMS.structure, PTS_merscope_request_metadata_roi
#         self.hemisphere # may need to get this from slab parent if it is None
        self.plane_of_section =  bers_spec.external_data.plane_of_section.name
        
        spec_pts = pts.get_processes_by_biological_entities(bers_spec.id)
        assert spec_pts.total_count == 1 and 'MerscopeCoverslipProcessing' in spec_pts.nodes[0].name, "not a MERSCOPE process"
        merscope_expt = spec_pts.nodes[0]
        
        self.qc_state = pts.get_process_metadata(process_id = merscope_expt.id, 
                                                 metadata_filter_input = MERSCOPESection.pts_qc_filt)[0].data['QC_State']
        self.gene_panel = pts.get_process_metadata(process_id = merscope_expt.id, 
                                                   metadata_filter_input = MERSCOPESection.pts_request_filt)[0].data['GenePanel']
        self.merscope_expt_pts_id = merscope_expt.id
        
#         self.section_thickness # not sure where this comes from?
#         self.z_coord # this will probably require some math and grabbing of parent z_coord - nothing currently in BERS 
#         self.parent_z_coord
        
    def _filter_dats_instances(self, instances):
        if len(instances) > 1:
            for instance in instances:
                if instance.storage['storage_provider'] == 'Isilon::POSIX':
                    return instance.download_url
        else:
            return instances[0].download_url

    def get_section_data_paths(self):      
        platform = sys.platform # don't like this but need to edit the file names to be read by Windows
        #dats
        dats = DatsClient()
        pts = PtsClient()
        macaque_dats_account = dats.get_account(name = self.config['merscope_lims_code'])
        
        merscope_expt = pts.get_process_by_id(self.merscope_expt_pts_id)
        spec_data_collection = None
        for output in merscope_expt.outputs: 
            try:
                dats_collection = dats.get_collection_by_id(account_id = macaque_dats_account.id, collection_id = output.external_id)
                if dats_collection.description == 'merfish_output' and dats_collection.type == 'File Bundle':
                    spec_data_collection = dats_collection
                    break
            except:
                pass
        
        assert spec_data_collection is not None, f'No merfish_output collection found for {self.barcode} with Isilon instances. Data paths cannot be determined'
        for asset in spec_data_collection.digital_assets:
            if asset.type == 'CSV' and 'detected_transcripts' in asset.name:
                self.detected_transcripts_file = self._filter_dats_instances(asset.instances)
                if platform.startswith('win'):
                    self.detected_transcripts_file = '/' + self.detected_transcripts_file
            if asset.type == 'Directory' and 'images' in asset.name:
                self.images_path = self._filter_dats_instances(asset.instances)
                if platform.startswith('win'):
                    self.images_path = '/' + self.images_path

        assert hasattr(self, 'detected_transcripts_file') and hasattr(self, 'images_path'), f'No detected_transcripts or images found for {self.barcode}, check Allen Services query'
                
    def load_spottable(self):
        if hasattr(self, 'detected_transcripts_cache') is False:
            self.detected_transcripts_cache = self.save_path.joinpath('detected_transcripts.npz')

        if hasattr(self, 'images_path'):
            spot_table = SpotTable.load_merscope(self.detected_transcripts_file, self.detected_transcripts_cache, self.images_path)

        else:
            spot_table = SpotTable.load_merscope(self.detected_transcripts_file, self.detected_transcripts_cache)

        return spot_table

    def run_segmentation_on_section(self, subrgn, seg_method, seg_opts, hpc_opts):
        # generate a new timestamp for segmentation
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        segmentation_path = self.segmentation_path
        individual_seg_dir = segmentation_path.joinpath(str(timestamp))
        assert not os.path.exists(individual_seg_dir)
        seg_run = MerscopeSegmentationRun.from_spatial_dataset(self, individual_seg_dir, subrgn, seg_method, seg_opts, hpc_opts)
        spot_table, cell_by_gene = seg_run.run(use_prod_cids=False)

        cell_by_gene.uns.update({
                    'seg_timestamp': timestamp,
                    'seg_params': self.check_segmentation_status()[timestamp],
                    })

        return timestamp, spot_table, cell_by_gene

    def load_segmentation_results(self, timestamp):
        # useful for attempting to load results from older segmentations
        # that don't have metadata pkl files
        segmentation_path = self.segmentation_path
        individual_seg_dir = segmentation_path.joinpath(str(timestamp))
        assert os.path.exists(individual_seg_dir)

        cid_path = individual_seg_dir.joinpath('segmentation.npy')
        cbg_path = individual_seg_dir.joinpath('cell_by_gene.h5ad')

        cell_ids = np.load(cid_path)
        cell_by_gene = ad.read_h5ad(cbg_path)
        spot_table = self.load_spottable()
        spot_table.cell_ids = cell_ids

        return spot_table, cell_by_gene

    def calc_cell_qc_metrics(self, cell_by_gene):
        """Calculate basic qc parameters and add them to the cell by gene
        table.
        """
        cbg_copy = cell_by_gene.copy()
        n_transcripts = calc_n_transcripts(cell_by_gene)
        n_genes = calc_n_genes(cell_by_gene)
        
        n_blanks = calc_n_blanks(cell_by_gene)
        pct_blanks = n_blanks*100 / n_transcripts
        
        cell_by_gene.var['probe_type'] = cell_by_gene.var_names.str.startswith('Blank')
        cbg_copy.obs['n_transcripts'] = n_transcripts
        cbg_copy.obs['n_genes'] = n_genes
        cbg_copy.obs['pct_counts_blank'] = pct_blanks

        return cbg_copy

    def run_qc_filtering_on_section(self, cell_by_gene, qc_params: dict, use_cols: list[str]|None=None):
        """Filter cells based on defined qc params. Keys of qc_params must be
        in the cell_by_gene.obs.

        Note that the cell_by_gene table is not subset; instead, a column
        cell_qc_pass is created that indicates whether cells passed or failed
        any qc thresholds.
        """
        cbg_copy = cell_by_gene.copy()
        for param, (lower, upper) in qc_params.items():
            cbg_copy.obs[f'{param}_toolow_qc'] = cbg_copy.obs[param] < lower
            cbg_copy.obs[f'{param}_toohigh_qc'] = cbg_copy.obs[param] > upper

        if use_cols is None:
            # use all qc params for filtering
            cbg_copy.obs['cell_qc_pass'] = ~cbg_copy.obs.loc[:, cbg_copy.obs.columns.str.endswith('_qc')].apply(np.any, axis=1)
        else:
            # use only indicated columns
            cbg_copy.obs['cell_qc_pass'] = ~cbg_copy.obs.loc[:, use_cols].apply(np.any, axis=1)

        cbg_copy.uns.update({'qc_params': qc_params})

        return cbg_copy

    def check_doublet_detection_status(self):
        doublet_dir = self.qc_path
        dd_runs = [f.name for f in os.scandir(doublet_dir) if f.is_dir()]
        if len(dd_runs) == 0:
            status = None

        elif len(dd_runs) > 0:
            status = {}
            for run_name in dd_runs:
                output = doublet_dir.joinpath(run_name, 'cell_by_gene_doublets.h5ad')
                dd_info = {'finished': output.exists()}

                try:
                    dd_meta_file = doublet_dir.joinpath(run_name, 'doublet_detection_params.json')
                    with open(dd_meta_file) as f:
                        dd_meta = json.load(f)
                    dd_info['doublet_params'] = dd_meta
                except FileNotFoundError:
                    pass

                status[run_name] = dd_info

        return status

    def run_doublet_detection_on_section(self, cell_by_gene, method, method_kwargs, hpc_args, filter_col):
        """Submits a job to run doublet detection on a GPU node on the HPC."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        doublet_dir = self.qc_path.joinpath(timestamp)
        doublet_dir.mkdir()

        run_spec = (run_doublet_detection, (), {
            'cell_by_gene': cell_by_gene, 
            'output_dir': f'{doublet_dir.as_posix()}/',
            'method': method, 
            'method_kwargs': method_kwargs,
            'filter_col': filter_col
            }
        )

        job_path = doublet_dir.joinpath('hpc_jobs')
        job_path.mkdir(exist_ok=True)
        hpc_config = {
                'run_spec': run_spec,
                'conda_env': '/allen/programs/celltypes/workgroups/rnaseqanalysis/NHP_spatial/seg_tests/conda-envs/sis_test/',
                'hpc_host': 'hpc-login',  # change to 'localhost' if running from hpc
                'job_path': f'{job_path.as_posix()}/',
                'partition': 'celltypes',
                'job_name': 'doublet_det',
                'nodes': 1,
                'ntasks': 1,
                'mincpus': 1,
                'gpus_per_node': 1,
                'mem': '20G',
                'time': '02:00:00',
                'mail_user': None,
        }
        hpc_config.update(**hpc_args)
        jobs = run_slurm_func(**hpc_config)

        print('Job ID: ', jobs.base_job_id)
        d_out = doublet_dir.joinpath('doublet_results.csv')
        while not d_out.exists():
            print(jobs.state())
            time.sleep(60)

        doublet_df = pd.read_csv(d_out, index_col=0)
        meta_file = doublet_dir.joinpath('doublet_detection_params.json')
        with open(meta_file, 'r') as f:
            doublet_meta = json.load(f)

        doublet_df.index = doublet_df.index.astype(str)
        cbg_copy = cell_by_gene.copy()
        cbg_copy.obs = cbg_copy.obs.merge(doublet_df['doublet_score'], how='left', right_index=True, left_index=True)
        cbg_copy.obs['doublet_score'] = cbg_copy.obs['doublet_score'].fillna('qc_failed')
        cbg_copy.obs['doublet_score'] = pd.Categorical(cbg_copy.obs['doublet_score'], categories=['qc_failed', 'singlet', 'doublet'])
        cbg_copy.uns.update({'doublet_timestamp': timestamp, 'doublet_params': doublet_meta}) 
        
        doublet_ad_path = doublet_dir.joinpath('cell_by_gene_doublets.h5ad')
        cbg_copy.write_h5ad(doublet_ad_path)

        return cbg_copy, timestamp

    def load_doublet_anndata(self, timestamp):
        doublet_ad_path = self.qc_path.joinpath(timestamp, 'cell_by_gene_doublets.h5ad')
        return ad.read_h5ad(doublet_ad_path)

    def make_anndata(self, cell_by_gene, file_name='sp_anndata.h5ad'):
        sp_anndata_path = self.save_path.joinpath(file_name)
        cell_by_gene.write_h5ad(sp_anndata_path)
        self.anndata_file = sp_anndata_path
        self.save_dataset()

    def run_mapping_on_section(self, method, taxonomy=None, method_args={}, hpc_args={}):
        if method == ScrattchMapping:
            mapping = ScrattchMapping(
                sp_data = self.anndata_file,
                taxonomy_path = self.config['taxonomy_info'][taxonomy]['path'],
                save_path=self.mapping_path,
                meta = {'taxonomy_name': taxonomy, 'taxonomy_cols': self.config['taxonomy_info'][taxonomy]['col_labels']}
            )
        else:
            print(f'Mapping method not instantiated for {method}')
            return
        
        hpc_args_default = {
            'job_path': f"{self.config['hpc_outputs']}/scripts/",
            'output':f"{self.config['hpc_outputs']}/%j.out",
            'error': f"{self.config['hpc_outputs']}/%j.err",
        }
        
        hpc_args_default.update(hpc_args)
        
        if 'docker' in hpc_args.keys():
            docker = hpc_args['docker']
            hpc_args_default.pop('docker')
            job = mapping.run_on_hpc(method_args, hpc_args_default, docker=docker)
        else:
            job = mapping.run_on_hpc(method_args, hpc_args_default)
        
        return job, mapping    

    def make_cirro(self, ct_map):
        if type(ct_map) == str:
            ct_map = CellTypeMapping.load_from_timestamp(directory=self.mapping_path, timestamp=ct_map)

        if isinstance(ct_map, ScrattchMapping):
            if hasattr(ct_map, 'ad_map') is False:
                ct_map.load_scrattch_mapping_results()
        
        if 'raw_counts' in ct_map.ad_map.layers.keys():
            raw_dat = ct_map.ad_map.to_df(layer='raw_counts')
        else:
            raw_dat = ct_map.ad_map.to_df()
            
        ct_map.spatial_umap(attr='ad_map')
        x_umap = ct_map.ad_map.obs[['umap_x', 'umap_y']].to_numpy()
        spatial = ct_map.ad_map.obs[['center_x', 'center_y']].to_numpy()

        obsm = {
                'spatial': spatial,
                'X_umap': x_umap,
            } 

        X = raw_dat[ct_map.ad_map.var.index.unique()]

        var = pd.DataFrame(index=ct_map.ad_map.var.index.unique())

        ad_cirro = ad.AnnData(
            X = X,
            obs = ct_map.ad_map.obs,
            var = var,
            obsm = obsm,
            uns = ct_map.ad_map.uns,
        )

        cirro_file = Path(ct_map.run_directory).joinpath(f'ad_cirro_{self.lims_specimen_name}.h5ad')
        ad_cirro.write_h5ad(cirro_file)
        print(cirro_file)

        return cirro_file

# class MERSCOPESectionCollection(MERSCOPESection):
#     #collection of MERSCOPE sections that go together

class StereoSeqSection(SpatialDataset):
    
    def __init__(self, barcode):
        config = load_config()
        save_path = Path(config['stereoseq_save_path']).joinpath(barcode)
        if Path.is_file(save_path.joinpath('spatial_dataset')):
            print(f'SpatialDataset already exists and will be loaded. If you want to reprocess this dataset delete the file and start over')
            cached = SpatialDataset.load_from_barcode(barcode, StereoSeqSection)
            self.__dict__ = cached.__dict__
            print('QC status:')
            print(self.qc)

        else:
            SpatialDataset.__init__(self, barcode)
            self.save_path = save_path

            if not Path.exists(self.save_path):
                Path.mkdir(self.save_path)
            
            self.xyscale = 0.5
            self.broad_region = None
            self.mapping_path = self.save_path.joinpath(self.config['mapping_dir']) if self.config.get('mapping_dir') is not None else None
            if self.mapping_path is not None and not Path.exists(self.mapping_path):
                Path.mkdir(self.mapping_path)
            self.segmentation_path = self.save_path.joinpath(self.config['segmentation_dir']) if self.config.get('segmentation_dir') is not None else None
            if self.segmentation_path is not None and not Path.exists(self.segmentation_path):
                Path.mkdir(self.segmentation_path)

            self.qc = {
                'bin200_median_transcript': None,
                'bin200_transcript_coverage': None,
                'corr_to_bulk': None,
                'corr_to_pair': None,
                'visual_inspection': None,
                'gene_expression': None,
                'celltype_mapping': None,
            }

            self.get_section_metadata()
            self.get_section_data_paths()

            self.save_dataset() 
        
    def set_partner_chips(self, barcodes):
        self.partner_chips = barcodes
        self.save_dataset()

    def get_section_data_paths(self):
        # hopefully this will get integrated into Allen Services but for now, hardcode paths
        
        self.detected_transcripts_file  = self.save_path.joinpath('gem_files', (self.barcode + '.tissue.gem'))
        self.bin_file =  self.save_path.joinpath('gem_files', (self.barcode + '.tissue.gef'))
        cellbin_file =  self.save_path.joinpath('gem_files', (self.barcode + '.cellbin.gef'))
        image_file = self.save_path.joinpath(f'ssDNA_{self.barcode}_regist.tif')
        # we might not always download the image file or the cellbin file:
        if image_file.is_file():
            self.image_file = image_file
        if cellbin_file.is_file():
            self.cellbin_file = cellbin_file
    
    def get_section_metadata(self):
        # hopefully this will get integrated into Allen Services but for now, hardcode 
        
        self.species = 'Macaque'
        
    def load_spottable(self):       
        if hasattr(self, 'detected_transcripts_cache') is False:
            self.detected_transcripts_cache = self.save_path.joinpath('detected_transcripts.npz')

        if hasattr(self, 'image_file') and Path(self.image_file).is_file():
            spot_table = SpotTable.load_stereoseq(self.detected_transcripts_file, self.detected_transcripts_cache, skiprows=7, image_file=self.image_file, image_channel='nuclear')

        else:
            spot_table = SpotTable.load_stereoseq(self.detected_transcripts_file, self.detected_transcripts_cache, skiprows=7)

        return spot_table
        
    def qc_widget(self, metric):
        interact_manual(self.evaluate_qc_metric, qc_metric=metric, qc=widgets.Dropdown(
                            options=[None,'Pass', 'Fail'],
                            description='QC result:',
                            value=None,
                            disabled=False,
                            style={'description_width': 'initial'},
                        ))
        
    def evaluate_qc_metric(self, qc_metric, qc): 
        self.qc[qc_metric] = qc
        self.save_dataset()
        print(f'{qc_metric}: {self.qc[qc_metric]}')

    def qc_gene_expression(self, gene_list=None, spot_table=None, save_fig=True):
        # optionally provide a spot table if one is already loaded
        if spot_table is None:
            spot_table = self.load_spottable()
        
        if gene_list is None:
            gene_list = pd.read_csv(self.config['stereoseq_qc_genes_file'])
            gene_list.sort_values('gene expression pattern', inplace=True)
            gene_list = gene_list['gene'].to_list()

        rows = 4
        cols = int(np.ceil(len(gene_list) / rows))

        fig, ax = plt.subplots(cols, rows, figsize=(16, 4*cols))
        row = 0
        col = 0
        for i, gene in enumerate(gene_list):
            if i != 0 and i % rows == 0:
                row += 1
                col = 0
            try:
                spot_table.get_genes(gene_names=[gene]).show_binned_heatmap(ax=ax[row, col])
            except KeyError:
                ax[row, col].axis('off')
            ax[row, col].set_title(gene)
            col += 1

        if save_fig is True:
            fig.savefig( self.save_path.joinpath('gene_expression_qc.png'))


        self.qc_widget(metric='gene_expression')
    
    def qc_spatial_corr_to_bulk(self, threshold=0.6, kwargs={}):
        self.spatial_corr_to_bulk(**kwargs)
        self.qc['corr_to_bulk'] = 'Pass' if self.corr_to_bulk >= threshold else 'Fail'
        print(f"corr_to_bulk: {self.qc['corr_to_bulk']}")
        
    def qc_bin200(self, median_transcript_thresh=18000, base_transcript_thresh=7000, save_fig=True):
        import stereo as st

        ad200_file = Path.joinpath(self.save_path, 'ad_sp_bin200.h5ad')
        if not Path.is_file(ad200_file):
            meta = {'chip_name': self.barcode,
                    'binsize': '200',
                    'species': self.species,
                    'region': self.broad_region,
                    'method': 'StereoSeq',
                   }  
            bin200_data = st.io.read_gef(file_path=str(self.bin_file), bin_size=200)
            bin200_data.tl.raw_checkpoint()
            bin200_data.tl.cal_qc()
            ad_bin200 = st.io.stereo_to_anndata(bin200_data, flavor='scanpy', reindex=True, output= ad200_file)
            ad_bin200.uns.update(meta)
            ad_bin200.obs.rename(columns={'x': 'center_x', 'y': 'center_y'}, inplace=True)
            ad_bin200.obs['center_x'] *= self.xyscale
            ad_bin200.obs['center_y'] *= self.xyscale
            ad_bin200.write_h5ad(ad200_file)
        else:
            ad_bin200 = ad.read_h5ad(ad200_file)

        self.bin200_median_transcript = ad_bin200.obs['total_counts'].median()
        self.bin200_transcript_coverage = len(ad_bin200.obs[ad_bin200.obs['total_counts']>base_transcript_thresh])/len(ad_bin200)*100

        self.evaluate_qc_metric('bin200_median_transcript', 'Pass' if self.bin200_median_transcript >= median_transcript_thresh else 'Fail')
        self.evaluate_qc_metric('bin200_transcript_coverage', 'Pass' if self.bin200_transcript_coverage >= 80 else 'Fail')

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        sns.violinplot(data=ad_bin200.obs, y='total_counts', ax=ax[0], cut=0)
        ax[0].axhline(median_transcript_thresh, color='orange', ls='--')
        ax[0].set_title(f'Median MID: {self.bin200_median_transcript}')
        sns.ecdfplot(data=ad_bin200.obs, x='total_counts', ax=ax[1])
        ax[1].axvline(base_transcript_thresh, color='orange', ls='--')
        ax[1].axhline(0.2, color='orange', ls='--')  
        ax[1].set_title(f'% Bins > Base MID: {self.bin200_transcript_coverage:.2f}%')
        sns.violinplot(data=ad_bin200.obs, y='n_genes_by_counts', ax=ax[2], cut=0)
        plt.tight_layout()

        if save_fig is True:
            fig.savefig( self.save_path.joinpath('Bin200_transcript_detection.png'))
    
    def run_mapping_on_section(self, method, binsize=50, taxonomy=None, training_genes='hvgs', method_args={}, hpc_args={}):
        import stereo as st

        bin_key = 'bin' + str(binsize) if binsize != 'cell' else binsize
        ad_file = Path.joinpath(self.save_path, 'ad_sp_' + bin_key + '.h5ad')
        if not Path.is_file(ad_file):
            uns = {
                'chip_name': self.barcode,
                'binsize': str(binsize),
                'species': self.species,
                'region': self.broad_region,
                'method': 'StereoSeq',
                }
            
            if binsize == 'cell':
                data = st.io.read_gef(file_path=str(self.cellbin_file), bin_type='cell_bins')
            else:
                data = st.io.read_gef(file_path=str(self.bin_file), bin_size=binsize)
            
            data.tl.raw_checkpoint()
            data.tl.cal_qc()
            ad_data = st.io.stereo_to_anndata(data, flavor='scanpy', reindex=True, output= ad_file)
            ad_data.uns.update(uns)
            ad_data.layers['logcounts'] = np.log1p(ad_data.X)
            ad_data.obs.rename(columns={'x': 'center_x', 'y': 'center_y'}, inplace=True)
            ad_data.obs['center_x'] *= self.xyscale
            ad_data.obs['center_y'] *= self.xyscale 
            ad_data.write_h5ad(ad_file)
        else:
            ad_data = ad.read_h5ad(ad_file)

        if taxonomy is not None:
            taxonomy_path = self.config['taxonomy_info'][taxonomy]['path']

        if method == ScrattchMapping:
            mapping = ScrattchMapping(
                sp_data = ad_file,
                taxonomy_path = taxonomy_path,
                save_path=self.mapping_path,
                meta = {'taxonomy_name': taxonomy, 'taxonomy_cols': self.config['taxonomy_info'][taxonomy]['col_labels']}
            )
        else:
            print(f'Mapping method not instantiated for {method}')
            return
        
        if training_genes == 'hvgs' and taxonomy is not None:
            taxonomy_ad = ad.read_h5ad(Path(taxonomy_path).joinpath('AI_taxonomy.h5ad'), backed='r')
            training_genes = taxonomy_ad.var[taxonomy_ad.var['highly_variable_genes']==True].index.to_list()
            meta = {'training_genes': 'taxonomy highly variable genes'}
       
        ad_map_args = {'ad_sp_layer': 'logcounts',
                    'training_genes': training_genes,
                    'meta': meta}
        ad_map_args.update(method_args)

        hpc_args_default = {
            'job_path': self.config['hpc_outputs'] + '/scripts/',
            'output': self.config['hpc_outputs'] + '/%j.out',
            'error': self.config['hpc_outputs'] + '/%j.err',
        }
        hpc_args_default.update(hpc_args)

        if 'docker' in hpc_args.keys():
            docker = hpc_args['docker']
            hpc_args_default.pop('docker')
            job = mapping.run_on_hpc(ad_map_args, hpc_args_default, docker=docker)
        else:
            job = mapping.run_on_hpc(ad_map_args, hpc_args_default)
        
        return job, mapping
    
    def qc_mapping_results(self, ct_map, score_thresh, map_thresh=0.6):
        if type(ct_map) == str:
            ct_map = CellTypeMapping.load_from_timestamp(directory=self.mapping_path, timestamp=ct_map)

        if isinstance(ct_map, ScrattchMapping):
            mapping_score_col = 'score.Corr'

        fig, ax = plt.subplots()
        sns.ecdfplot(data=ct_map.ad_map.obs, x=mapping_score_col, ax=ax)
        ax.axvline(score_thresh, color='orange', ls='--')
        ax.axhline(1-map_thresh, color='orange', ls='--')  
        
        ct_map.qc_mapping(qc_params={mapping_score_col: score_thresh})
        
        self.mapping_quality = sum(ct_map.ad_map.obs[mapping_score_col] >= score_thresh)/len(ct_map.ad_map.obs)
        self.evaluate_qc_metric('celltype_mapping', 'Pass' if self.mapping_quality >= map_thresh else 'Fail')
        ax.set_title(f'Mapping Quality: {self.mapping_quality:.2f}')
        fig.savefig(self.save_path.joinpath('mapping_qc.png'))   

    def qc_corr_to_pair(self, corr_thresh=0.8):
        if self.partner_chips is None:
            print('No partner chips set')
            return
        barcodes = self.partner_chips.copy()
        barcodes.append(self.barcode)
        collection = StereoSeqSectionCollection(barcode_list=barcodes)
        counts = collection.get_counts()
        log_cols = [col for col in counts.columns if 'log' in col]
        corr = counts[log_cols].corr()
        self.corr_to_pair = corr.loc[f'{self.barcode} log counts'][1:].to_dict()
        self.evaluate_qc_metric('corr_to_pair', 'Pass' if all([v >= corr_thresh for v in self.corr_to_pair.values()]) else 'Fail')
        
        g = sns.PairGrid(data=counts, vars=log_cols)
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot, s=10, alpha=0.2)

        ax = g.fig.add_axes([1, 0, 1, 1])

        sns.heatmap(counts[log_cols].corr(), cmap ='viridis', linewidths = 0.30, annot = True, ax=ax,
                    cbar_kws={'label': 'Pearson Corr'})

        g.savefig(self.save_path.joinpath('corr_to_pair_qc.png'))
        
class StereoSeqSectionCollection(StereoSeqSection):
    def __init__(self, barcode_list):
        spd_list = []
        for barcode in barcode_list:
            section = SpatialDataset.load_from_barcode(barcode, StereoSeqSection)
            if section is None:
                section = StereoSeqSection(barcode)
            spd_list.append(section)

        self.sections = spd_list

    def get_counts(self):
        total_counts_df = None
        for section in self.sections:
            spot_table = section.load_spottable()
            
            gene_ids, total_counts = np.unique(spot_table.gene_ids, return_counts=True)
            genes = spot_table.map_gene_ids_to_names(gene_ids)
            
            df = pd.DataFrame({'gene': genes, f'{section.barcode} total counts': total_counts, 
                            f'{section.barcode} log counts': np.log(total_counts)})
            df.set_index('gene', inplace=True)

            if total_counts_df is None:
                total_counts_df = df
            else:
                total_counts_df = total_counts_df.merge(df, left_index=True, right_index=True, how='outer')
        
        return total_counts_df

# class XeniumSection(SpatialDataset):
#     def __init__(self, barcode):
#         super().__init__(barcode)   
#         self.save_path = os.path.join(self.config['xenium_save_path'], barcode)
#         if not os.path.exists(self.save_path):
#             os.mkdir(self.save_path)


# class XeniumSectionCollection(XeniumSection):
