## Class SpatialDataset to organize spatial datasets from variety of platforms
## includes subclasses for each platform to encompass data processing steps
## use to version data processing
## These classes can just store pointers to segmentation, mapping, etc and not copies of the data itself. This will allow the tools already
## built to still access those files without having to load the whole class which could get burdensome
## utilizes new Allen Services for data grabbing 
## spatial_config.yml to store datapaths and other configurable properties

version = 1

from sawg.celltype_mapping import ScrattchMapping, CellTypeMapping
from sawg.segmentation import run_segmentation, CellposeSegmentationMethod, CellposeSegmentationResult
from sawg.spot_table import SpotTable
from sawg.util import load_config
import anndata as ad
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
from pathlib import Path
from .optional_import import optional_import
widgets = optional_import('ipywidgets')
interact_manual = optional_import('ipywidgets.interact_manual')


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
            data_path = Path(config['merscope_save_path']).joinpath(str(barcode), file_name)
        if dataset_type == StereoSeqSection:
            data_path = Path(config['stereoseq_save_path']).joinpath(str(barcode), file_name)
        # other config paths would be added here as they come about
        if not Path.is_file(data_path):
            print(f'SpatialDataset {data_path} does not exist')
            return None
        with open(data_path, 'rb') as file:
            dataset = pkl.load(file)
            file.close()
            print(f'SpatialDataset {data_path} loaded...')
        return dataset
    
    def save_dataset(self, file_name: str='spatial_dataset'):
        file = open(self.save_path.joinpath(file_name), 'wb')
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
            fig.savefig(self.save_path.joinpath('correlation_to_bulk.pdf'))
        
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
            fig.savefig(self.save_path.joinpath('marker_gene_transcripts.pdf'))
        
class MERSCOPESection(SpatialDataset):

    pts_qc_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="QCMetadata")))
    pts_request_filt = pts_schema.MetadataFilterInput(type=pts_schema.DataTypeFilterInput(name=pts_schema.StringOperationFilterInput(eq="MerscopeImagingRequestMetadata")))

    def __init__(self, barcode):
        config = load_config()
        save_path = Path(config['merscope_save_path']).joinpath(str(barcode))
        if Path.is_file(save_path.joinpath('spatial_dataset')):
            print(f'SpatialDataset already exists and will be loaded. If you want to reprocess this dataset delete the file and start over')
            cached = SpatialDataset.load_from_barcode(barcode, MERSCOPESection)
            self.__dict__ = cached.__dict__
        
        else:
            SpatialDataset.__init__(self, barcode)
            self.save_path = save_path
            if not Path.exists(self.save_path):
                Path.mkdir(self.save_path)
            
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
        
    def get_section_data_paths(self):
        platform = sys.platform # don't like this but need to edit the file names to be read by Windows
        #dats
        dats = DatsClient()
        pts = PtsClient()
        macaque_dats_account = dats.get_account(name = self.config['merscope_lims_code'])
        
        merscope_expt = pts.get_process_by_id(self.merscope_expt_pts_id)
        for output in merscope_expt.outputs: 
            try:
                dats_collection = dats.get_collection_by_id(account_id = macaque_dats_account.id, collection_id = output.external_id)
                if dats_collection.type == 'File Bundle':
                    spec_data_collection = dats_collection
                    break
            except:
                pass

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
#         # should it replace self.anndata? keep segmented cbg with segmentation results, save out new anndata with cell statistics and qc columns as the useable self.anndata.
#         #        This can then be replaced when new segmentations are done and the mapping object will retain the copy of the anndata that was used for that mapping


#     def run_segmentation_on_section(self, method):
#         self.segmentation = CellposeSegmentationResult
        
#     def run_mapping_on_section(self, method):

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
            if self.segmentation_path is not None and Path.exists(self.segmentation_path):
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

            self.ad_files = {}
            self.mappings = []
            self.segmentations = []

            self.get_section_metadata()
            self.get_section_data_paths()

            self.save_dataset() 
        
    def get_section_data_paths(self):
        # hopefully this will get integrated into Allen Services but for now, hardcode paths
        
        self.detected_transcripts_file  = self.save_path.joinpath('gem_files', (self.barcode + '.tissue.gem'))
        self.bin_file =  self.save_path.joinpath('gem_files', (self.barcode + '.tissue.gef'))
        self.cellbin_file =  self.save_path.joinpath('gem_files', (self.barcode + '.tissue.gef'))
    
    def get_section_metadata(self):
        # hopefully this will get integrated into Allen Services but for now, hardcode 
        
        self.species = 'Macaque'
        
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
            fig.savefig( self.save_path.joinpath('gene_expression_qc.pdf'))


        self.qc_widget(metric='gene_expression_qc')
    
    def qc_spatial_corr_to_bulk(self, threshold=0.6, kwargs={}):
        self.spatial_corr_to_bulk(**kwargs)
        self.qc['corr_to_bulk'] = 'Pass' if self.corr_to_bulk >= threshold else 'Fail'
        print(f"corr_to_bulk: {self.qc['corr_to_bulk']}")
        
    def qc_bin200(self, median_transcript_thresh=18000, base_transcript_thresh=7000, save_fig=True):
        import stereo as st

        ad200_file = self.ad_files.get('bin200', None)
        if ad200_file is None:
            ad200_file = os.path.join(self.save_path, 'ad_sp_bin200.h5ad')
            self.ad_files['bin200'] = ad200_file

        if not os.path.isfile(ad200_file):
            meta = {'chip_name': self.barcode,
                    'binsize': '200',
                    'species': self.species,
                    'region': self.broad_region,
                    'method': 'StereoSeq',
                   }  
            bin200_data = st.io.read_gef(file_path=self.bin_file, bin_size=200)
            bin200_data.tl.raw_checkpoint()
            bin200_data.tl.cal_qc()
            ad_bin200 = st.io.stereo_to_anndata(bin200_data, flavor='scanpy', reindex=True, output= ad200_file)
            ad_bin200.uns.update(meta)
            ad_bin200.write_h5ad(ad200_file)
        else:
            ad_bin200 = ad.read_h5ad(ad200_file)

        self.bin200_median_transcript = ad_bin200.obs['total_counts'].median()
        self.bin200_transcript_coverage = len(ad_bin200.obs[ad_bin200.obs['total_counts']>base_transcript_thresh])/len(ad_bin200)*100

        self.qc['bin200_median_transcript'] = 'Pass' if self.bin200_median_transcript >= median_transcript_thresh else 'Fail'
        print(f"bin200_median_transcript: {self.qc['bin200_median_transcript']}")

        self.qc['bin200_transcript_coverage'] = 'Pass' if self.bin200_transcript_coverage >= 0.8 else 'Fail'
        print(f"bin200_transcript_coverage: {self.qc['bin200_transcript_coverage']}")

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        sns.violinplot(data=ad_bin200.obs, y='total_counts', ax=ax[0], cut=0)
        ax[0].axhline(18000, color='orange', ls='--')
        ax[0].set_title(f'Median MID: {self.bin200_median_transcript}')
        sns.ecdfplot(data=ad_bin200.obs, x='total_counts', ax=ax[1])
        ax[1].axvline(7000, color='orange', ls='--')
        ax[1].axhline(0.2, color='orange', ls='--')  
        ax[1].set_title(f'% Bins > Base MID: {self.bin200_transcript_coverage:.2f}%')
        sns.violinplot(data=ad_bin200.obs, y='n_genes_by_counts', ax=ax[2], cut=0)
        plt.tight_layout()

        if save_fig is True:
            fig.savefig( self.save_path.joinpath('Bin200_transcript_detection.pdf'))
    
    def run_mapping(self, binsize=50, taxonomy=None, training_genes='hvgs', kwargs={}):
        import stereo as st

        bin_key = 'bin' + str(binsize) if binsize != 'cell' else binsize
        ad_file = self.ad_files.get(bin_key, None)
        if ad_file is None:
            ad_file = Path.joinpath(self.save_path, 'ad_sp_' + bin_key + '.h5ad')
            self.ad_files[bin_key] = ad_file

        if not os.path.isfile(ad_file):
            meta = {
                'chip_name': self.barcode,
                'binsize': str(binsize),
                'species': self.species,
                'region': self.broad_region,
                'method': 'StereoSeq',
                }
            
            if binsize == 'cell':
                data = st.io.read_gef(file_path=self.cellbin_file, bin_type='cell_bins')
            else:
                data = st.io.read_gef(file_path=self.bin_file, bin_size=binsize)
            
            data.tl.raw_checkpoint()
            data.tl.cal_qc()
            ad_data = st.io.stereo_to_anndata(data, flavor='scanpy', reindex=True, output= ad_file)
            ad_data.uns.update(meta)
            ad_data.layers['logcounts'] = np.log1p(ad_data.X)
            ad_data.write_h5ad(ad_file)
        else:
            ad_data = ad.read_h5ad(ad_file)

        if taxonomy is None or taxonomy not in self.config['taxonomy_paths']:
            print(f"No taxonomy provided or taxonmomy not in listed config {self.config['taxonomy_paths'].keys()}. Skipping celltype mapping QC.")
            return
        taxonomy_path = self.config['taxonomy_paths'][taxonomy]
        if training_genes == 'hvgs':
            taxonomy_ad = ad.read_h5ad(Path.joinpath(taxonomy_path, 'AI_taxonomy.h5ad'), backed='r')
            training_genes = taxonomy_ad.var[taxonomy_ad.var['highly_variable_genes']==True].index.to_list()
            meta = {'training_genes': 'taxonomy highly variable genes'}
        else:
            meta = kwargs.get('meta', {})

        scrattch_map = ScrattchMapping(
            sp_data = ad_data,
            taxonomy_path = taxonomy_path,
        )
        
        ad_map_args = {'save_path': self.mapping_path.as_posix(), 
                    'ad_sp_layer': 'logcounts',
                    'training_genes': training_genes,
                    'meta': meta}
        if ad_map_args in kwargs.keys():
            ad_map_args.update(kwargs['ad_map_args'])

        hpc_args = {
            'job_path': self.config['hpc_outputs'] + '/scripts/',
            'output': self.config['hpc_outputs'] + '/%j.out',
            'error': self.config['hpc_outputs'] + '/%j.err',
        }
        if hpc_args in kwargs.keys():
            hpc_args.update(kwargs['hpc_args'])

        docker = kwargs.get('docker', None)

        scrattch_map.run_on_hpc(ad_map_args, hpc_args, docker=docker)
        self.mappings.append(os.path.basename(scrattch_map.run_directory)) 
        return scrattch_map

    def qc_mapping(self, mapping, threshold=0.1):
        if not isinstance(mapping, ScrattchMapping):
            mapping = ScrattchMapping.load_from_timestamp(directory=self.mapping_path, timestamp=str(mapping))

        mapping.load_scrattch_mapping_results()
        
# class StereoSeqSectionPair(StereoSeqSection):

# class XeniumSection(SpatialDataset):
#     def __init__(self, barcode):
#         super().__init__(barcode)   
#         self.save_path = os.path.join(self.config['xenium_save_path'], barcode)
#         if not os.path.exists(self.save_path):
#             os.mkdir(self.save_path)


# class XeniumSectionCollection(XeniumSection):