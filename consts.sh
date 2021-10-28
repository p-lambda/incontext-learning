#!/bin/bash

ROOT="."
mkdir -p $ROOT
source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate eix-hf

CACHE=cache
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE

LOGDIR=logs
mkdir -p $LOGDIR
