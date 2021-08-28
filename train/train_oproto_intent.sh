#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src

# Training OProto on intent classification K=100
python train_oproto_intent.py -config ../config/config_intent -section test-run > ../train/log_train_oproto_intent.txt

