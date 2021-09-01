#!/bin/bash
source HOME_DIR_INSTALLATION/miniconda3/etc/profile.d/conda.sh
conda activate protoinfomax_env
cd ../src

# Evaluate ProtoNet on intent classification K=100 N=1
#python eval_proto_intent.py -config ../config/config_intent -section test-run > ../eval/eval_proto_intent.txt

# Evaluate ProtoNet on intent classification K=100 N=2
python eval_proto_intent_n2.py -config ../config/config_intent_n2 -section test-run > ../eval/eval_proto_intent_n2.txt

