#!/bin/bash
DATASET="Beijing-Opera"
METHOD=$1

if [ "$METHOD" != "zeroshot" ] && [ "$METHOD" != "coop" ] && [ "$METHOD" != "cocoop" ] && [ "$METHOD" != "palm" ]; then
    echo "Invalid METHOD=$METHOD . Please choose one of the following: ['zeroshot', 'coop', 'cocoop', 'palm']"
    exit 1
fi

echo "Running METHOD=$METHOD on DATASET=$DATASET"

DATASET_ROOT="<SET_PATH_TO_DATASET_ROOT_DIRECTORY_HERE>/Audio-Datasets/$DATASET"

if [ -d "$DATASET_ROOT" ]; then
    echo "Dataset path exists: $DATASET_ROOT"
else
    echo "Dataset path does not exist. Please set the correct path to the dataset root directory in variable DATASET_ROOT"
fi

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


for FOLD in 1 2 3 4 5 
    do
        for SEED in $SEEDS
            do
                echo "Running Fold-$FOLD with SEED=$SEED"
                if [ -f "$DATASET_ROOT/train.csv" ]; then rm -rf "$DATASET_ROOT/train.csv"; fi
                if [ -f "$DATASET_ROOT/test.csv" ]; then rm -rf "$DATASET_ROOT/test.csv"; fi
                cp "$DATASET_ROOT/csv_files/train_$FOLD.csv" "$DATASET_ROOT/train.csv"
                cp "$DATASET_ROOT/csv_files/test_$FOLD.csv" "$DATASET_ROOT/test.csv"

                python main.py \
                    --model_name $METHOD \
                    --dataset_root $DATASET_ROOT \
                    --n_epochs 50 \
                    --freq_test_model 10 \
                    --ctx_dim $CTX_DIM \
                    --batch_size 16 \
                    --lr 0.05 \
                    --seed $SEED \
                    --exp_name "$DATASET-FOLD$FOLD" \
                    --num_shots 16 \
                    --do_logging 
            done
    done