#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src

# Training Proto-net on sentiment classification K=100
python train_proto.py -config ../config/config_sentiment -section test-run > ../train/log_train_proto_sentiment.txt
