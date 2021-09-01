#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate protoinfomax_env
cd ../src

# Training ProtoNet on intent classification K=100
python train_proto_intent.py -config ../config/config_intent -section test-run > ../train/log_train_proto_intent.txt

