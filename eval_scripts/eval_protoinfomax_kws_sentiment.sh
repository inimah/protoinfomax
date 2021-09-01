#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate protoinfomax_env
cd ../src

# Evaluate ProtoInfoMax++ on sentiment classification K=100
python eval_imax_kw.py -config ../config/config_sentiment_kw -section test-run > ../eval/eval_imax_kw_sentiment.txt 

