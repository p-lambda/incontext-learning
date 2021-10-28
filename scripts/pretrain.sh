#!/bin/bash

source consts.sh

N_VALUES=${1-10}
N_SLOTS=${2-10}
N_SYMBOLS=${3-10}
TRANS_TEMP=${4-0.1}
START_TEMP=${5-10.0}
VIC=$6
N_HMMS=$7
SEED=$8
DATA_NAME=GINC_trans${TRANS_TEMP}_start${START_TEMP}_nsymbols${N_SYMBOLS}_nvalues${N_VALUES}_nslots${N_SLOTS}_vic${VIC}_nhmms${N_HMMS}

bash run_pretrain.sh $DATA_NAME $SEED
bash run_incontext.sh $DATA_NAME 6000 $SEED

