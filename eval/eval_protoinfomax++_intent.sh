#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV
cd ../src

# Evaluate ProtoInfoMax++ on intent classification K=100
#python eval_imax_kw_intent.py -config ../config/config_intent_kw -section test-run > ../eval/eval_imaxkw_intent.txt


# Evaluate ProtoInfoMax on intent classification K=100 N=2
python eval_imax_kw_intent_n2.py -config ../config/config_intent_kw_n2 -section test-run > ../eval/eval_imaxkw_intent_n2.txt