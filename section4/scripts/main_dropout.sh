#!/bin/bash

throttleSecs=60
stepsPerCkpt=600
trainSteps=15000
evalSteps=700

decay="0.0"
B=128
V=80000
T=50
E=300

dataName="amazon_reviews"
dataDir="data/${dataName}"
processedDataDir="processed_data/${dataName}"

dropoutArr=("0.0" "0.2" "0.5" "0.8")
for drop in "${dropoutArr[@]}"; do
    modelDir="model_dir/dropout_experiment/${dataName}_V${V}_Drop${drop:2}_L2${decay:2}"

    ./main.py \
        --data_dir=${dataDir} \
        --processed_data_dir=${processedDataDir} \
        --model_dir=${modelDir} \
        --steps_per_ckpt=${stepsPerCkpt} \
        --train_steps=${trainSteps} \
        --eval_steps=${evalSteps} \
        --l2_decay=${decay} \
        --dropout_prob=${drop} \
        --throttle_secs=${throttleSecs} \
        -V $V -T $T -E $E -B $B
done
