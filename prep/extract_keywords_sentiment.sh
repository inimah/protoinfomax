#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src

# Preparation stage -- extracting keywords
python extract_sentiment.py -config ../config/config_sentiment -section test-run > ./log_kw_extraction_sentiment.txt

