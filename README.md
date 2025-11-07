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
conda create -n sis python=3.10
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
1. **Cell-by-Gene Table:** A matrix with cells as rows and genes as columns
2. **Metadata:** A file containing descriptive data for each segmented cell
3. **Cell outlines:** In the form of a geojson file


## Level of support

We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.