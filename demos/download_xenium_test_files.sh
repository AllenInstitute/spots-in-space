#!/bin/bash

# Download transcript information
echo "Downloading xenium output zip..."
curl -O "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Brain_Coronal_FF/Xenium_Prime_Mouse_Brain_Coronal_FF_outs.zip"

# Unzip and delete the downloaded file
echo "Unzipping xenium output zip..."
unzip Xenium_Prime_Mouse_Brain_Coronal_FF_outs.zip -d xenium_test_data
rm Xenium_Prime_Mouse_Brain_Coronal_FF_outs.zip

# Removing excess files
echo "Removing excess files..."
cd xenium_test_data
rm *.zip *.gz analysis_summary.html cell_boundaries.parquet cells.parquet cell_feature_matrix.h5 experiment.xenium gene_panel.json nucleus_boundaries.parquet metrics_summary.csv
rm -r morphology_focus

echo "Download complete."