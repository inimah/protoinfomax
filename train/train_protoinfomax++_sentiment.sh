#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src

# Training ProtoInfoMax++ on sentiment classification K=100
python train_imax_kw.py -config ../config/config_sentiment_kw -section test-run > ../train/log_train_imax_kw_sentiment.txt 

