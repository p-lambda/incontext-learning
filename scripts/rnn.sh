#!/bin/bash
set -x

source consts.sh

DATA_NAME=$1
DATA_DIR=$ROOT/data/$DATA_NAME
mkdir -p $DATA_DIR

MODEL=$2
MODEL_SUFFIX=""
SEED=$3
MODEL_DIR=$ROOT/rnn_output/${DATA_NAME}_SEED${SEED}
mkdir -p $MODEL_DIR


FULL_MODEL_NAME=${MODEL}${MODEL_SUFFIX}
python rnn/main.py \
        --cuda \
        --data $DATA_DIR \
        --epochs 10 \
        --save-interval 10 \
        --lr 1e-3 \
        --batch_size 8 \
        --bptt 1024 \
        --seed $SEED \
        --nlayers 6 \
        --nhid 768 \
        --clip 1.0 \
        --model $MODEL \
        --save ${MODEL_DIR}/${FULL_MODEL_NAME}_checkpoint.pt

python rnn/generate.py \
    --data ${DATA_DIR} \
    --checkpoint ${MODEL_DIR}/${FULL_MODEL_NAME}_checkpoint.pt \
    --cuda \
    --results_name ${FULL_MODEL_NAME}_in_context_results.tsv
