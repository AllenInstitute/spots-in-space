<img src=images/logo1-1.png width="100"> 

# Spots-In-Space 

A Python based scalable pipeline to apply the **Cellpose** segmentation algorithm to large, subcellular resolution spatial transcriptomics datasets. This tool provides wrapper functions to tile large images, distribute the segmentation jobs on a SLURM cluster, and stitch the results back together. We also provide the option to create an mRNA density image from the spot table as a label for the cytosolic regions.


## ✨ Features
* **Tiling & Stitching:** Automatically splits large images into tiles for processing and reassembles the final segmentation masks.
* **mRNA Density Image:** Generates mRNA density image from spot table. 
* **SLURM Integration:** Distributes segmentation tasks across cluster nodes for high-throughput processing.
* **Comprehensive Outputs:** Generates standard cell-by-gene tables, metadata files, and cell outlines in GeoJSON format.
* **Built-in Visualization:** Includes plotting functions to easily visualize cell boundaries overlaid on the original image data.

## Installation

Create a conda environment in which to install spots-in-space:
```bash
conda create -n sis python=3.12
conda activate sis
```

We recommend cloning this repo and installing via `pip`:
```bash
git clone https://github.com/AllenInstitute/spots-in-space.git
cd spots-in-space
pip install ".[cellpose]"
```
Should you be interested in a distribution of spots-in-space without cellpose, it can be installed as such:
```bash
git clone https://github.com/AllenInstitute/spots-in-space.git
cd spots-in-space
pip install .
```

## Outputs

This pipeline generates:
1. **Cell-by-Gene Table:** An h5ad file with a .X matrix with cells as rows and genes as columns, a .obs containing descriptive data for each segmented cell, a .var containing per gene information, and a .uns containing cell polygons and segmentation metadata.
2. **Transcript assignments:** A .npy file containing cell ids of transcripts in the order of the input transcripts file.
3. **Cell outlines:** A .geojson FeatureCollection file with geometry, id, and properties assignments.
4. **SegmentedSpotTable:** A .npz file containing the SegmentedSpotTable object. Loadable via sis.SegmentedSpotTable.load_npz().
5. **Regions:** A .json file containing the regions for each of the segmentation tiles.
6. **Metadata:** A .json file containing metadata about the segmentation.

## Docs
Documentation for SIS classes and functions can be found at https://spots-in-space.readthedocs.io/en/latest/

## Level of support

We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
