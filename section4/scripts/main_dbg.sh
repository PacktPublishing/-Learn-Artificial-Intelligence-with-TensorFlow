#!/bin/bash

throttleSecs=600
stepsPerCkpt=500
evalSteps=700
trainSteps=10000

B=128
V=80000
T=50
E=300

dataName="amazon_reviews"
dataDir="data/${dataName}"
processedDataDir="processed_data/${dataName}"

lr="1e-3"
modelDir="model_dir/tfdb_experiment/${dataName}_V${V}_LR${lr}"

../main.py \
    --data_dir=${dataDir} \
    --processed_data_dir=${processedDataDir} \
    --model_dir=${modelDir} \
    --steps_per_ckpt=${stepsPerCkpt} \
    --train_steps=${trainSteps} \
    --eval_steps=${evalSteps} \
    --dropout_prob=0.5 \
    --learning_rate=${lr} \
    --throttle_secs=${throttleSecs} \
    -V $V -T $T -E $E -B $B
