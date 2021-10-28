#!/bin/bash

source consts.sh


# SMALL, MEDIUM, LARGE

N_VALUES=10
N_SLOTS=10
N_SYMBOLS=50
TRANS_TEMP=0.1
START_TEMP=10.0
VIC=0.9
N_HMMS=10

for N_SYMBOLS in 50 # 100 150
do
jid_gen=$(sbatch \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/SMALL_nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_generate.log \
    scripts/generate.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS)
echo -n "${jid_gen} "

for SEED in 1111 # 1112 1113 1114 1115
do
jid=$(sbatch \
    --dependency=afterany:${jid_gen} \
    --gres=gpu:1 \
    --parsable \
    --output $LOGDIR/SMALL_nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_SEED${SEED}.log \
    scripts/pretrain_small.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS $SEED $LR)
echo -n "${jid} "
done
done

for N_SYMBOLS in 50 100 150
do
jid_gen=$(sbatch \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/MEDIUM_nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_generate.log \
    scripts/generate.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS)
echo -n "${jid_gen} "

for SEED in 1111 1112 1113 1114 1115
do
jid=$(sbatch \
    --dependency=afterany:${jid_gen} \
    --gres=gpu:1 \
    --parsable \
    --output $LOGDIR/MEDIUM_nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_SEED${SEED}.log \
    scripts/pretrain.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS $SEED $LR)
echo -n "${jid} "
done
done

for N_SYMBOLS in 50 100 150
do
jid_gen=$(sbatch \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/LARGE_nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_generate.log \
    scripts/generate.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS)
echo -n "${jid_gen} "
for SEED in 1111 1112 1113 1114 1115
do
jid=$(sbatch \
    --dependency=afterany:${jid_gen} \
    --parsable \
    --gres=gpu:2 \
    --output $LOGDIR/LARGE_nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_SEED${SEED}.log \
    scripts/pretrain_large.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS $SEED)
echo -n "${jid} "
done
done

# ABLATIONS: RANDOM DATA

N_VALUES=10
N_SLOTS=10
N_SYMBOLS=50
TRANS_TEMP=0.1
START_TEMP=10.0
VIC=0.9
N_HMMS=10
for N_SYMBOLS in 50 100 150
do
jid1=$(sbatch \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_random_generate.log \
    scripts/random_generate.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS)
echo -n "${jid1} "
for SEED in 1111 1112 1113 1114 1115
do
jid=$(sbatch \
    --parsable \
    --dependency=afterany:${jid1} \
    --gres=gpu:1 \
    --output $LOGDIR/nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_SEED${SEED}_random.log \
    scripts/random.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS $SEED)
echo -n "${jid} "
done
done

# ABLATIONS: no prior on properties (slots)

N_VALUES=10
N_SLOTS=10
N_SYMBOLS=50
TRANS_TEMP=0.1
START_TEMP=10.0
VIC=0.9
N_HMMS=10
for N_SYMBOLS in 50 100 150
do
jid1=$(sbatch \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_noslot_generate.log \
    scripts/noslot_generate.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS)
echo -n "${jid1} "
for SEED in 1111 1112 1113 1114 1115
do
jid=$(sbatch \
    --parsable \
    --dependency=afterany:${jid1} \
    --gres=gpu:1 \
    --output $LOGDIR/nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_SEED${SEED}_noslotprior.log \
    scripts/noslotprior.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS $SEED)
echo -n "${jid} "
done
done

N_VALUES=10
N_SLOTS=10
N_SYMBOLS=50
TRANS_TEMP=0.1
START_TEMP=10.0
VIC=0.0 # this is the main change - don't mix identity with transition matrix
N_HMMS=10
for N_SYMBOLS in 50 100 150
do
jid1=$(sbatch \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_generate.log \
    scripts/generate.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS)
echo -n "${jid1} "
for SEED in 1111 1112 1113 1114 1115
do
jid=$(sbatch \
    --dependency=afterany:${jid1} \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_SEED${SEED}.log \
    scripts/fastchangingc.sh $N_VALUES $N_SLOTS $N_SYMBOLS $TRANS_TEMP $START_TEMP $VIC $N_HMMS $SEED)
echo -n "${jid} "
done
done

# RNN experiments
N_VALUES=10
N_SLOTS=10
N_SYMBOLS=50
VIC=0.9
TRANS_TEMP=0.1
START_TEMP=10.0
N_HMMS=10

for N_SYMBOLS in 50 100 150
do
for SEED in 1111 1112 1113 1114 1115
do
DATA_NAME=GINC_trans${TRANS_TEMP}_start${START_TEMP}_nsymbols${N_SYMBOLS}_nvalues${N_VALUES}_nslots${N_SLOTS}_vic${VIC}_nhmms${N_HMMS}
jid=$(sbatch \
    --dependency=afterany:${jid_gen} \
    --parsable \
    --gres=gpu:1 \
    --output $LOGDIR/RNN_nvalues${N_VALUES}_nslots${N_SLOTS}_nsymbols${N_SYMBOLS}_transtemp${TRANS_TEMP}_starttemp${START_TEMP}_VIC${VIC}_nhmms${N_HMMS}_SEED${SEED}.log \
    scripts/rnn.sh $DATA_NAME LSTM $SEED)
echo -n "${jid} "
done
done
