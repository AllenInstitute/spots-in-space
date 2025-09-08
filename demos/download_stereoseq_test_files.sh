#!/bin/bash

# Set up the test data dir
mkdir -p stereoseq_test_data
cd stereoseq_test_data

# Download transcript information
echo "Downloading transcripts..."
curl -O "https://ftp.cngb.org/pub/stomics/STT0000124/Analysis/STSA0000877/STTS0001449/SS200000135TL_D1.tissue.gem.gz"
gunzip SS200000135TL_D1.tissue.gem.gz

# Download transcript information
echo "Downloading image..."
curl -O "https://ftp.cngb.org/pub/stomics/STT0000124/Analysis/STSA0000877/STTS0001449/SS200000135TL_D1_regist.tif"

echo "Download complete."

