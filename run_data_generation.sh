#!/bin/bash

source consts.sh

TRANS_TEMP=$1
START_TEMP=$2
N_SYMBOLS=$3
N_VALUES=$4
N_SLOTS=$5
VIC=$6
N_HMMS=$7
OTHER_ARGS=$8

python generate_data.py \
    --transition_temp $TRANS_TEMP \
    --start_temp $START_TEMP \
    --n_symbols $N_SYMBOLS \
    --n_values $N_VALUES \
    --n_slots $N_SLOTS \
    --value_identity_coeff $VIC \
    --n_hmms $N_HMMS \
    --root $ROOT \
    $OTHER_ARGS
