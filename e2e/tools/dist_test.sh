#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT ${@:3}

# CUDA_VISIBLE_DEVICES=1  bash tools/dist_test.sh configs/petr/petr_r50_16x2_100e_coco.py checkpoint/petr_r50_16x2_100e_coco.pth 1 --show-dir  /output  --eval keypoints
