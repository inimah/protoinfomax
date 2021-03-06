#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate protoinfomax_env
cd ../src

# Preparation stage -- extracting keywords
# note that here we provide example on extracting keywords from Sentiment dataset (AmazonDat)
python extract_sentiment.py -config ../config/config_sentiment -section test-run > ./log_kw_extraction_sentiment.txt

