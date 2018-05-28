#!/bin/bash

throttleSecs=120
stepsPerCkpt=100  # 500
evalSteps=10 # 700
trainSteps=100

B=256
V=100000
T=80
E=300

dataName="amazon_reviews"
dataDir="data/${dataName}"
processedDataDir="processed_data/${dataName}"

lr="1e-3"
modelDir="model_dir/temp/${dataName}_E${E}_T${T}_V${V}"

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
