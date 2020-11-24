#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-38500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python6 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/vg_test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
