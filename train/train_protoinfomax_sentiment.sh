#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src



# Training ProtoInfoMax on sentiment classification K=100
python train_imax.py -config ../config/config_sentiment -section test-run > ../train/log_train_imax_sentiment.txt


