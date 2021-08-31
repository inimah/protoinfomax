#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate protoinfomax_env
cd ../src

# Training O-Proto on sentiment classification K=100
python train_oproto.py -config ../config/config_sentiment -section test-run > ../train/log_train_oproto_sentiment.txt
