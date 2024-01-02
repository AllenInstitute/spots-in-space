# cell_type_mapper Workflow

This a workflow for running the cell_type_mapping package to map adult & developing whole mouse brain datasets to the latest Allen Institute whole mouse brain (WMB) taxonomy (release date: 08302023, taxonomy: CCN202307220).

Repository: https://github.com/AllenInstitute/cell_type_mapper/

Version: 1.0.0 (Oct 10, 2023)


### A) Install dependencies & the cell-type-mapper package in a clean 'cell_type_mapper' environment
- Follow the instructions at: https://github.com/AllenInstitute/cell_type_mapper/#installation
    - Create & activate a clean python=3.9 environment.
    - Download, clone, or fork the cell_type_mapper repository.
    - Run ```pip install -r requirements.txt``` to install the required packages into your environment.
    - Run ```pip install -e .``` from the root directory of this repository to install this package itself.


### B) Change the file i/o pathways in run_cell_type_mapper_pipeline.py script to match your project's pathways
Including:
- The output directory
- The input directory and path to your unlabeled dataset (.h5ad file)
- File paths related to Steps 1 & 2 (see C)


### C) From your 'cell_type_mapper' environment, run the run_cell_type_mapper_pipeline.py script

This script follows the 4 steps described in: https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md

#### 1. [Computing the average gene expression profile per cell type cluster](https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md#1-computing-the-average-gene-expression-profile-per-cell-type-cluster)
- This has been precomputed for the latest version of the Whole Mouse Brain taxonomy, CCN202307220, and can be found at ```//allen/aibs/technology/danielsf/knowledge_base/handoff_230821/precomputed_stats_ABC_revision_230821.h5```
- If you need to compute this step from a different, please refer to the [package's detailed workflow instructions](https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md#1-computing-the-average-gene-expression-profile-per-cell-type-cluster)

#### 2. [Encoding marker genes](https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md#2-encoding-marker-genes)
- This has been precomputed for the latest version of the Whole Mouse Brain taxonomy, CCN202307220, and can be found at ```//allen/aibs/technology/danielsf/knowledge_base/handoff_230821/mouse_markers_230821.json```
- If you need to compute this step from a different, please refer to the [package's detailed workflow instructions](https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md#2-encoding-marker-genes)

#### 3. ["Validating" the unlabeled dataset](https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md#3-validating-the-unlabeled-dataset)

This is executed in run_cell_type_mapper_pipeline.py.

#### 4. [Mapping unlabeled data onto the reference taxonomy](https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md#4-mapping-unlabeled-data-onto-the-reference-taxonomy)

This is executed in run_cell_type_mapper_pipeline.py.


### D) To add the mapping results into the .obs columns of your input h5ad file & save a new h5ad, run the following from your 'cell_type_mapper' environment:

```python -m cell_type_mapper.cli.transcribe_to_obs --result_path /ouput/file/path/extended_results.json --h5ad_path /path/to/input/h5ad/unlabeled_query_dataset.h5ad --new_h5ad_path /path/to/new_file.h5ad```

- Change result_path, h5ad_path, and new_h5ad_path
- All mapping columns in the new_h5ad .obs will with start with "CDM_"