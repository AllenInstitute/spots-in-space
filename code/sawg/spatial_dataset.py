## Class SpatialDataset to organize spatial datasets from variety of platforms
## includes subclasses for each platform to encompass data processing steps
## use to version data processing
## These classes can just store pointers to segmentation, mapping, etc and not copies of the data itself. This will allow the tools already
## built to still access those files without having to load the whole class which could get burdensome
## utilizes new Allen Services for data grabbing 
## spatial_config.yml to store datapaths and other configurable properties

version = 1

from sawg.celltype_mapping import ScrattchMapping
from sawg.segmentation import run_segmentation, CellposeSegmentationMethod, CellposeSegmentationResult
from sawg.segmentation import get_segmentation_region, get_tiles, create_seg_run_spec, merge_segmentation_results
from sawg.spot_table import SpotTable
from sawg.util import load_config
import anndata as ad
import os, sys, datetime, glob, pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact_manual


## Allen Services
import allen_services_api.dats.schema.dats_schema as dats_schema
import allen_services_api.pts.schema.pts_schema as pts_schema
import allen_services_api.bers.schema.bers_schema as bers_schema
from allen_services_api.dats.client.dats_client import DatsClient
from allen_services_api.pts.client.pts_client import PtsClient
from allen_services_api.bers.client.bers_client import BersClient


class SpatialDataset:
    # wrapper class
    def __init__(self, barcode):
        self.barcode = barcode
        self.version = version
        self.config = load_config()
        
    @classmethod
    def load_from_barcode(cls, barcode: str, dataset_type: 'SpatialDataset', file_name='spatial_dataset'):
        config = load_config()
        if dataset_type == MERSCOPESection:
            data_path = os.path.join(config['merscope_save_path'], str(barcode), file_name)
        if dataset_type == StereoSeqSection:
            data_path = os.path.join(config['stereoseq_save_path'], str(barcode), file_name)
        # other config paths would be added here as they come about
        if not os.path.isfile(data_path):
            print(f'SpatialDataset {data_path} does not exist')
            return None
        with open(data_path, 'rb') as file:
            dataset = pkl.load(file)
            file.close()
            print(f'SpatialDataset {data_path} loaded...')
        return dataset
    
    def save_dataset(self, file_name: str='spatial_dataset'):
        file = open(os.path.join(self.save_path, file_name), 'wb')
        pkl.dump(self, file)
        file.close()
        
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
            fig.savefig(os.path.join(self.save_path, 'correlation_to_bulk.pdf'))
        
        self.save_dataset()

    def show_gene_transcripts(self, genes: list, subregions=None, spot_table=None, save_fig=True):
        if spot_table is None:
            spot_table = self.load_spottable()
            
        fig_rows = len(subregions) if subregions is not None else 1
        fig, ax = plt.subplots(fig_rows, 2, figsize=(10, 5*fig_rows))
        
        genes_table = spot_table.get_genes(genes)
        reduced_spots = int(np.floor(len(spot_table)/100000))
        if fig_rows==1:
            spot_table[::reduced_spots].scatter_plot(ax[0])
            df = genes_table.dataframe(cols=['x', 'y', 'gene_names'])
            sns.scatterplot(data=df, x='x', y='y', hue='gene_names', s=3, alpha=0.5, linewidth=0, ax=ax[1], palette='tab10')
            ax[1].legend(bbox_to_anchor=(1, 1))
        else:  
            for i, subregion in enumerate(subregions):
                spot_table[::reduced_spots].scatter_plot(ax[i, 0])
                sub_table = genes_table.get_subregion(xlim=subregion[0], ylim=subregion[1])
                sub_table.plot_rect(ax[i, 0], 'r')
                df = sub_table.dataframe(cols=['x', 'y', 'gene_names'])
                sns.scatterplot(data=df, x='x', y='y', hue='gene_names', s=3, alpha=0.5, linewidth=0, ax=ax[i, 1], palette='tab10')
                ax[i, 1].legend(bbox_to_anchor=(1, 1))
                
        if save_fig is True:
            fig.savefig(os.path.join(self.save_path, 'marker_gene_transcripts.pdf'))
        
class MERSCOPESection(SpatialDataset):

    pts_qc_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="QCMetadata")))
    pts_request_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="MerscopeImagingRequestMetadata")))

    def __init__(self, barcode):
        config = load_config()
        save_path = os.path.join(config['merscope_save_path'], str(barcode))
        if os.path.isfile(os.path.join(save_path, 'spatial_dataset')):
            print(f'SpatialDataset already exists and will be loaded. If you want to reprocess this dataset delete the file and start over')
            cached = SpatialDataset.load_from_barcode(barcode, MERSCOPESection)
            self.__dict__ = cached.__dict__
        
        else:
            SpatialDataset.__init__(self, barcode)
            self.save_path = save_path
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            
            self.broad_region = None
            self.get_section_metadata()
            self.get_section_data_paths()
            
            self.save_dataset()
        
        #self.seg_dir = os.path.join(save_path, config['segmentation_dir'])
        # the above doesn't work if the segmentation dir has a leading slash
        self.seg_dir = save_path + config['segmentation_dir']
        if not os.path.exists(self.seg_dir):
            os.mkdir(self.seg_dir)
        # Get all subdirectories in segmentation directory
        # Each should correspond to a segmentation run... unless there are
        # weird random folders that shouldn't be there...
        self.seg_runs = [f.name for f in os.scandir(self.seg_dir) if f.is_dir()]

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
        
    def get_section_data_paths(self):
        platform = sys.platform # don't like this but need to edit the file names to be read by Windows
        #dats
        dats = DatsClient()
        pts = PtsClient()
        macaque_dats_account = dats.get_account(name = self.config['merscope_lims_code'])
        
        merscope_expt = pts.get_process_by_id(self.merscope_expt_pts_id)
        for output in merscope_expt.outputs: 
            dats_collection = dats.get_collection_by_id(account_id = macaque_dats_account.id, collection_id = output.external_id)
            if dats_collection.type == 'File Bundle':
                spec_data_collection = dats_collection
                break

        for asset in spec_data_collection.digital_assets:
            if asset.type == 'CSV' and 'detected_transcripts' in asset.name:
                assert len(asset.instances) == 1, f'more than one instance of asset {asset.name}'
                self.detected_transcripts_file = asset.instances[0].download_url
                if platform.startswith('win'):
                    self.detected_transcripts_file = '/' + self.detected_transcripts_file
            if asset.type == 'Directory' and 'images' in asset.name:
                assert len(asset.instances) == 1, f'more than one instance of asset {asset.name}'
                self.images_path = asset.instances[0].download_url
                if platform.startswith('win'):
                    self.images_path = '/' + self.images_path
                
    def load_spottable(self):
        if hasattr(self, 'detected_transcripts_cache') is False:
            self.detected_transcripts_cache = os.path.join(self.save_path, 'detected_transcripts.npz')
        spot_table = SpotTable.load_merscope(self.detected_transcripts_file, self.detected_transcripts_cache)
        return spot_table

    def set_segmentation_path(self, timestamp = None):
        """Set the path for an individual segmentation."""
        seg_dir = self.seg_dir 

        # does it make sense to set these as None?
        # we could also set what they would be by default
        # and then future steps could check whether they exist
        self.regions_path = None
        self.run_spec_path = None
        self.cid_path = None

        if timestamp is not None:
            # load from existing segmentation
            this_seg_dir = os.path.join(seg_dir, timestamp)
            assert timestamp in self.seg_runs and os.path.exists(this_seg_dir)
            regions_path = os.path.join(this_seg_dir, 'regions.json')
            run_spec_path = os.path.join(this_seg_dir, 'run_spec.pkl')
            cid_path = os.path.join(this_seg_dir, 'segmentation.npy')

            if os.path.exists(regions_path):
                self.regions_path = regions_path
            if os.path.exists(run_spec_path):
                self.run_spec_path = run_spec_path
            if os.path.exists(cid_path):
                self.cid_path = cid_path

        else:
            # create a new subdirectory for segmentation
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            this_seg_dir = os.path.join(seg_dir, timestamp)
            os.mkdir(this_seg_dir)
            self.seg_runs.append(timestamp)

        self.this_seg_dir = this_seg_dir # does this make sense?

    def check_segmentation_status(self):
        if len(self.seg_runs) == 0:
            print(f'No segmentations available in {self.seg_dir}')
            return

        status = {}
        if len(self.seg_runs) > 0:
            for run_name in self.seg_runs:
                run_path = os.path.join(self.seg_dir, run_name)
                final_output = os.path.join(run_path, 'segmentation.npy')
                status[run_name] = os.path.exists(final_output)

        return status

    def get_load_func(self):
        """Get the function to load a spot table."""
        return SpotTable.load_merscope

    def get_load_args(self):
        """Get args to pass to loading function (e.g. when submitting jobs to hpc)."""
        load_args = {
                'image_path': self.images_path,
                'csv_file': self.detected_transcripts_file,
                'cache_file': self.detected_transcripts_cache,
                'max_rows': None
        }
        return load_args

    def save_regions(self, regions):
        regions_path = os.path.join(self.this_seg_dir, 'regions.json')
        regions_df = pd.DataFrame(regions, columns=['xlim', 'ylim'])
        regions_df.to_json(regions_path)
        self.regions_path = regions_path

    def load_regions(self):
        assert self.regions_path is not None and os.path.exists(self.regions_path)
        regions = pd.read_json(self.regions_path).values
        return regions

    def save_run_spec(self, run_spec):
        run_spec_path = os.path.join(self.this_seg_dir, 'run_spec.pkl')
        with open(run_spec_path, 'wb') as f:
            pickle.dump(run_spec, f)
        self.run_spec_path = run_spec_path

    def load_run_spec(self):
        with open(self.run_spec_path, 'rb') as f:
            run_spec = pickle.load(f)

        return run_spec

    def save_cell_ids(self, cell_ids):
        cid_path = os.path.join(self.this_seg_dir, 'segmentation.npy')
        np.save(cid_path, cell_ids)
        self.cid_path = cid_path

    def load_cell_ids(self):
        assert self.cid_path is not None and os.path.exists(self.cid_path)
        cell_ids = np.load(self.cid_path)
        return cell_ids

    def tile_section(self, subrgn='DAPI', **kwargs):
        load_func = self.get_load_func()
        load_args = self.get_load_args()
        table = load_func(**load_args)
        subtable = get_segmentation_region(table, subrgn)
        tiles, regions = get_tiles(subtable, **kwargs)
        
        # save regions
        self.save_regions(regions)

        return tiles, regions

    def get_seg_run_spec(self, regions, seg_method, seg_opts):
        tile_save_path = os.path.join(self.this_seg_dir, 'seg_tiles')
        if not os.path.exists(tile_save_path):
            os.mkdir(tile_save_path)

        run_spec = create_seg_run_spec(
                regions,
                self.get_load_func(),
                self.get_load_args(),
                seg_method,
                seg_opts,
                tile_save_path
        )

        # save run_spec
        self.save_run_spec(run_spec)

        return run_spec

    def submit_seg_jobs(self, run_spec, conda_env, hpc_host, job_path):
        jobs = run_segmentation_on_hpc(run_spec, conda_env, hpc_host, job_path)
        return jobs

    def merge_segmented_tiles(self, run_spec, tiles, subrgn='DAPI'):
        # this stuff repeated from above...
        load_func = self.get_load_func()
        load_args = self.get_load_args()
        table = load_func(**load_args)
        subtable = get_segmentation_region(table, subrgn)

        cell_ids, merge_results, skipped = merge_segmentation_results(subtable, run_spec, tiles)

        # save cell_ids...?
        self.save_cell_ids(cell_ids)

        return cell_ids, merge_results, skipped



# class MERSCOPESection(SpatialDataset):

#     pts_qc_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="QCMetadata")))
#     pst_request_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="MerscopeImagingRequestMetadata")))

#     def __init__(self):
#         self.save_path # get from config file?  
#         self.bers_id
#         self.get_section_metadata()
#         self.get_section_data_paths()
    
#     def get_section_data_paths(self):
#         #dats
#         self.images_path
#         self.detected_transcripts_path
#         self.expt_json
#         # do we even need these 2 if we're just going to re-segment?
#         self.cbg_path
#         self.cbg_meta_path

#     def get_section_metadata(self):
#         self.section_thickness # not sure where this comes from?
#         self.z_coord # this will probably require some math and grabbing of parent z_coord - nothing currently in BERS
#         self.gene_panel #PTS
#         self.qc_state #PTS
#         self.lims_specimen_name #BERS
#         self.species #BERS

#     def make_section_anndata(self):
#         # this may be a part of the segmentation? 
#         self.anndata = anndata

#     def qc_cells(self):
#         # question of when this step happens. 
#         # should it replace self.anndata? if the segmentation can easy recreate the original, or we save that as a csv then we can
#         # or could have self.qc_anndata
#         # it's too bad that anndata can't have layers of different dimensions

#     def run_segmentation_on_section(self, method):
#         self.segmentation = CellposeSegmentationResult
        
#     def run_mapping_on_section(self, method):

# class MERSCOPESectionCollection(MERSCOPESection):
#     #collection of MERSCOPE sections that go together

class StereoSeqSection(SpatialDataset):
    
    def __init__(self, barcode):
        config = load_config()
        save_path = os.path.join(config['stereoseq_save_path'], barcode)
        if os.path.isfile(os.path.join(save_path, 'spatial_dataset')):
            print(f'SpatialDataset already exists and will be loaded. If you want to reprocess this dataset delete the file and start over')
            cached = SpatialDataset.load_from_barcode(barcode, StereoSeqSection)
            self.__dict__ = cached.__dict__
        
        else:
            SpatialDataset.__init__(self, barcode)
            self.save_path = save_path

            if not os.path.exists(self.save_path):
                os.path.mkdir(self.save_path)
                
            self.xyscale = 0.5
            self.broad_region = None

            self.qc = {
                'bin200_median_transcript': None,
                'bin200_transcript_coverage': None,
                'corr_to_bulk': None,
                'corr_to_pair': None,
                'visual_inspection': None,
                'gene_expression': None,
                'celltype_mapping': None,
            }

            self.bin_files = {}
            self.ad_files = {}

            self.get_section_data_paths()

            self.save_dataset() 
        
    def get_section_data_paths(self):
        # hopefully this will get integrated into Allen Services but for now, hardcode paths
        
        self.detected_transcripts_file  = os.path.join(self.save_path, 'gem_files', (self.barcode + '.tissue.gem'))
        
    def load_spottable(self):       
        if hasattr(self, 'detected_transcripts_cache') is False:
            self.detected_transcripts_cache = os.path.join(self.save_path, 'detected_transcripts.npz')
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

    def evaluate_gene_expression(self, gene_list=None, spot_table=None, save_fig=True):
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
            fig.savefig(os.path.join(self.save_path, 'gene_expression_qc.pdf'))


        self.qc_widget(metric='gene_expression_qc')
    
    def spatial_corr_to_bulk_qc(self, threshold=0.6, kwargs={}):
        self.spatial_corr_to_bulk(**kwargs)
        self.qc['corr_to_bulk'] = 'Pass' if self.corr_to_bulk >= threshold else 'Fail'
        print(f"corr_to_bulk: {self.qc['corr_to_bulk']}")
        
    def evaluate_bin200(self, spot_table=None, median_transcript_thresh=18000, base_transcript_thresh=7000, save_fig=True):
        bin200_file = self.bin_files.get('bin200', None)
        if bin200_file is None:
            bin200_file = os.path.join(self.save_path, 'bin200.npz')
            self.bin_files['bin200'] = bin200_file
        ad200_file = self.ad_files.get('bin200', None)
        if ad200_file is None:
            ad200_file = os.path.join(self.save_path, 'ad_sp_bin200.h5ad')
            self.ad_files['bin200'] = ad200_file

        if not os.path.isfile(ad200_file):
            if spot_table is None:
                self.load_spottable()
            ad_bin200 = spot_table.bin_by_gene_anndata(binsize=200*self.xyscale, cache_file=bin200_file, ad_file = ad200_file)
        else:
            ad_bin200 = ad.read_h5ad(ad200_file)

        self.bin200_median_transcript = ad_bin200.obs['n_transcripts'].median()
        self.bin200_transcript_coverage = len(ad_bin200.obs[ad_bin200.obs['n_transcripts']>base_transcript_thresh])/len(ad_bin200)*100

        self.qc['bin200_median_transcript'] = 'Pass' if self.bin200_median_transcript >= median_transcript_thresh else 'Fail'
        print(f"bin200_median_transcript: {self.qc['bin200_median_transcript']}")

        self.qc['bin200_transcript_coverage'] = 'Pass' if self.bin200_transcript_coverage >= 0.8 else 'Fail'
        print(f"bin200_transcript_coverage: {self.qc['bin200_transcript_coverage']}")

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        sns.violinplot(data=ad_bin200.obs, y='n_transcripts', ax=ax[0], cut=0)
        ax[0].axhline(18000, color='orange', ls='--')
        ax[0].set_title(f'Median MID: {self.bin200_median_transcript}')
        sns.ecdfplot(data=ad_bin200.obs, x='n_transcripts', ax=ax[1])
        ax[1].axvline(7000, color='orange', ls='--')
        ax[1].axhline(0.2, color='orange', ls='--')  
        ax[1].set_title(f'% Bins > Base MID: {self.bin200_transcript_coverage:.2f}%')
        sns.violinplot(data=ad_bin200.obs, y='n_genes', ax=ax[2], cut=0)
        plt.tight_layout()

        if save_fig is True:
            fig.savefig(os.path.join(self.save_path, 'Bin200_transcript_detection.pdf'))

# class StereoSeqSectionPair(StereoSeqSection):

# class XeniumSection(SpatialDataset):

# class XeniumSectionCollection(XeniumSection):
