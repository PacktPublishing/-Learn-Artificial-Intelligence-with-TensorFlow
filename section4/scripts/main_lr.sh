#!/bin/bash

throttleSecs=60
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

# lrArr=("1e-3" "1e-4")
lrArr=("1e-2" "5e-2" "5e-4")
for lr in "${lrArr[@]}"; do
    modelDir="model_dir/lr_experiment/${dataName}_V${V}_LR${lr}"

    ./main.py \
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
done
