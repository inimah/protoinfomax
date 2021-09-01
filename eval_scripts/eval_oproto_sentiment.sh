#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate protoinfomax_env
cd ../src

# Evaluate O-Proto on sentiment classification K=100
python eval_oproto.py -config ../config/config_sentiment -section test-run > ../eval/eval_oproto_sentiment.txt
