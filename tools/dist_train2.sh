#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=11003\
    $(dirname "$0")/train_with_best_save.py $CONFIG --seed 7 --launcher pytorch ${@:3}