#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src



# Evaluate ProtoInfoMax on sentiment classification K=100
python eval_imax.py -config ../config/config_sentiment -section test-run > ../eval/eval_imax_sentiment.txt


