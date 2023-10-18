# cell_type_mapper pipeline description:
#   (A) Install dependencies & the cell-type-mapper package in a clean, python3.9 environment
#       - follow the instructions at:
#         https://github.com/AllenInstitute/cell_type_mapper/#installation
#   (B) Change the file i/o pathways to your project's
#   (C) From your env, run this .py script, which follows the 4 Steps in:
#       https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md
#         (1) (precomputed) Computing the average gene expression profile per cell type cluster
#         (2) (precomputed) Encoding marker genes 
#         (3) (runs in this script) "Validating" the unlabeled dataset
#         (4) (runs in this script) Mapping unlabeled data onto the reference taxonomy
#   (D) To add the mapping results into the .obs columns of your input h5ad file
#        & save a new h5ad, run the following from your env:
#         python -m cell_type_mapper.cli.transcribe_to_obs --result_path /ouput/file/path/extended_results.json --h5ad_path /path/to/input/h5ad/unlabeled_query_dataset.h5ad --new_h5ad_path /path/to/new_file.h5ad
#        - all mapping columns will with start with "CDM_"
#
# script by: Remi Mathieu
# additional comments & documentation by: Meghan Turner


import json
import tempfile
import os
import time

from cell_type_mapper.cli.validate_h5ad import (
    ValidateH5adRunner)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)


##### File I/O #####

# create a directory to store the output
mouse = 'mouse_'+'609882'
path_dir_res= os.path.join('/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/meghanturner/cell_type_mapper/',mouse)
# mouse = 'mouse_'+'669324'
# path_dir_res= os.path.join('/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/remimathieu/MERSCOPES/development/CDM',mouse)
os.makedirs(path_dir_res, exist_ok=True)

# create a directory to store the temporary files
path_tmp_res= os.path.join(path_dir_res,'scratch')
os.makedirs(path_tmp_res, exist_ok=True)

# extended_result_path
path_extended_result = os.path.join(path_dir_res,'extended_results.json')

# csv_result_path
path_csv_result = os.path.join(path_dir_res,'basic_results.csv')

# path to the h5ad file
h5ad_obj = 'atlas_brain_609882_AIT21.0_mouse.h5ad'
input_dir = os.path.join('//allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_609882/cirro_folder')
# h5ad_obj = mouse + '_combined_clusteredDF.h5ad'
# input_dir = os.path.join('/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/remimathieu/MERSCOPES/development/clustering/analysis',mouse)
path_h5ad = os.path.join(input_dir,h5ad_obj)


##### Run cell_type_mapper pipeline #####
if __name__ == "__main__":

    output_dir = path_dir_res

    # temporary file where validated path will be stored
    output_json = tempfile.mkstemp(suffix='.json')[1]
    
    # (3) Validate the unlabeled dataset
    start = time.time()
    
    config = {
        'h5ad_path': path_h5ad,
        'round_to_int': False,
        'layer': 'raw', # 'X' if raw counts are stored in X, 
                        # 'raw' if X stores log2CPM scaled counts & raw counts are stored in adata.layers['raw']
        'output_dir': output_dir,
        'output_json': output_json}
    
    validation_runner = ValidateH5adRunner(
        args=[],
        input_data=config)
    validation_runner.run()

    validated_path = json.load(open(output_json, 'rb'))['valid_h5ad_path']
    
    end = time.time()
    print('Validating took {} minutes'.format((end-start)/60))
    
    
    # (4) Map unlabeled (query) dataset onto reference taxonomy
    start = time.time()
        
    config = {
        'query_path': validated_path,
        'extended_result_path': path_extended_result,
        'csv_result_path': path_csv_result,
        'tmp_dir': path_tmp_res,
        # must drop "supertype" level, which is not used during hierarchical mapping
        'drop_level': 'CCN20230722_SUPT',
        'obsm_key' : 'CDM',
        # precomputed avg gene expression per cell type cluster in the taxonomy
        # from Step (1) of cell_type_mapper workflow
        # taxonomy used: CCN202307220 (aka likely AIT21.0?)
        'precomputed_stats': {
            'path': '/allen/aibs/technology/danielsf/knowledge_base/handoff_230821/precomputed_stats_ABC_revision_230821.h5'},
        # precomputed marker gene encoding from Step (2) of cell_type_mapper workflow
        'query_markers': {
            'serialized_lookup': '/allen/aibs/technology/danielsf/knowledge_base/handoff_230821/mouse_markers_230821.json'},
        'type_assignment': {
            'normalization': 'raw', # 'raw' if X is raw counts, 'log2CPM' if X is scaled counts
            'bootstrap_iteration': 100,
            'bootstrap_factor': 0.9,
            'n_processors': 32,
            'chunk_size': 10000,
            'rng_seed': 7812312}}
    
    runner = FromSpecifiedMarkersRunner(
        args = [],
        input_data=config)
    runner.run()
    
    end = time.time()
    print('Validating took {} hours'.format((end-start)/60/60))
