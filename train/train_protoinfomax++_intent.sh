#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src

# Training ProtoInfoMax++ on intent classification K=100
python train_imax_kw_intent.py -config ../config/config_intent_kw -section test-run > ../train/log_train_imaxkw_intent.txt
