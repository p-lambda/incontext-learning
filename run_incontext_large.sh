#!/bin/bash
set -x
source consts.sh

DATA_NAME=$1
DATA_DIR=$ROOT/data/$DATA_NAME
OUTPUT_DIR=$ROOT/outputs_large
CHECKPOINT=${2-6000}
SEED=$3

MODEL_DIR=$OUTPUT_DIR/${DATA_NAME}_SEED${SEED}_pretrain/checkpoint-${CHECKPOINT}

python run_clm.py \
    --model_name_or_path $MODEL_DIR \
    --tokenizer_name ${DATA_DIR}/tokenizer.json \
    --custom_tokenizer \
    --small_model \
    --output_dir $MODEL_DIR \
    --eval_incontext \
    --logging_steps 100 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --train_file $DATA_DIR/id_prompts.json \
    --validation_file $DATA_DIR/id_prompts.json \
    --block_size 1024 \
    --learning_rate 8e-4 \
    --num_train_epochs 3 \
    --warmup_steps 1000 \
    --fp16 \
    --custom_num_layers 16
