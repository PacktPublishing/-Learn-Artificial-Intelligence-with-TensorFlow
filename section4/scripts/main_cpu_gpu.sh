#!/bin/bash

throttleSecs=60
stepsPerCkpt=500
trainSteps=1000
evalSteps=700

decay="0.0"
# B=128
V=80000
T=50
E=300

dataName="amazon_reviews"
dataDir="data/${dataName}"
processedDataDir="processed_data/${dataName}"

batchArr=(8 32 128)
deviceArr=("gpu" "cpu")
for batch in "${batchArr[@]}"; do
    for device in "${deviceArr[@]}"; do
        modelDir="model_dir/cpu_gpu_experiment/${dataName}_B${batch}_${device}"

        ./main.py \
            --data_dir=${dataDir} \
            --processed_data_dir=${processedDataDir} \
            --model_dir=${modelDir} \
            --steps_per_ckpt=${stepsPerCkpt} \
            --train_steps=${trainSteps} \
            --eval_steps=${evalSteps} \
            --l2_decay=${decay} \
            --throttle_secs=${throttleSecs} \
            --device=${device} \
            -V $V -T $T -E $E -B ${batch}
    done
done
