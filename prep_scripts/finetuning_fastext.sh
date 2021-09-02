#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate protoinfomax_env
cd ../src

# Preparation stage -- finetuning word embeddings (FastText)
# note that here we provide example from Sentiment dataset (AmazonDat)
python train_fasttext.py -config ../config/config_sentiment -section test-run > ./log_finetuning_fasttext.txt

