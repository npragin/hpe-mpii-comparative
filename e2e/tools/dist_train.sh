#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher none ${@:3}
python $(dirname "$0")/train.py $CONFIG ${@:3}

# bash ./tools/dist_train.sh ./configs/petr/petr_r50_16x2_100e_mpii.py 2

# Color jtter sat 0.2 contro 0.2 0.2, brght 0.05 
# Normal to mage net.
# 256

# Tran val splt. 7/8 1/8

# Normal and resz

# Object eypont sm
