#!/bin/bash

BASE_URL="https://download.brainimagelibrary.org/aa/79/aa79b8ba5b3add56/638850/1199650932/merfish_output"

# Set up the test data dir
mkdir -p merscope_test_data/images/
cd merscope_test_data

# Download transcript information
echo "Downloading transcripts..."
curl -o detected_transcripts.csv "${BASE_URL}/cellpose_cyto2_nuclei/cellpose-detected_transcripts.csv"

# Download image information
cd images
echo "Downloading images..."
curl -O "${BASE_URL}/images/micron_to_mosaic_pixel_transform.csv"
for i in {0..6}; do
    FILE_NAME="mosaic_DAPI_z${i}.tif"
    curl -O "${BASE_URL}/images/${FILE_NAME}" || echo "Failed to download ${FILE_NAME}"
done

echo "Download complete."