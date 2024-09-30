#!/bin/bash
DATASET="NS-Instruments"
METHOD=$1

if [ "$METHOD" != "zeroshot" ] && [ "$METHOD" != "coop" ] && [ "$METHOD" != "cocoop" ] && [ "$METHOD" != "palm" ]; then
    echo "Invalid METHOD=$METHOD . Please choose one of the following: ['zeroshot', 'coop', 'cocoop', 'palm']"
    exit 1
fi

echo "Running METHOD=$METHOD on DATASET=$DATASET"

DATASET_ROOT="<SET_PATH_TO_DATASET_ROOT_DIRECTORY_HERE>/Audio-Datasets/$DATASET"


if [ "$METHOD" = "coop" ] || [ "$METHOD" = "cocoop" ]; then
    CTX_DIM=512
else
    CTX_DIM=1024
fi


if [ "$METHOD" = "zeroshot" ]; then
    SEEDS=0
else
    SEEDS="0 1 2"
fi






for SEED in $SEEDS
    do
        python main.py \
            --model_name $METHOD \
            --dataset_root $DATASET_ROOT \
            --n_epochs 50 \
            --freq_test_model 10 \
            --ctx_dim $CTX_DIM \
            --batch_size 16 \
            --lr 0.05 \
            --seed $SEED \
            --exp_name "$DATASET" \
            --num_shots 16 \
            --do_logging
    done









