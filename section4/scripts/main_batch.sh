#!/bin/bash

throttleSecs=60
stepsPerCkpt=600
trainSteps=15000
evalSteps=700

decay="0.0"
drop="0.8"
V=80000
T=50
E=300

dataName="amazon_reviews"
dataDir="data/${dataName}"
processedDataDir="processed_data/${dataName}"

learnArr=("1e-3")
batchArr=(16 64 128)

for learn in "${learnArr[@]}"; do
    for batch in "${batchArr[@]}"; do
        modelDir="model_dir/batch_experiment/learn${learn}/${dataName}_V${V}_Drop${drop:2}_B${batch}"

        echo "====================================================="
        echo "BATCH=${batch}    LEARN=${learn}"
        echo "====================================================="

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
            --learning_rate=${learn} \
            -V $V -T $T -E $E -B $batch
    done
done
