#!/bin/bash
clear

# $1: The GPU index for training, eg. 0
# $2: The dataset for training, eg. shrec13

# Example command: ./train.sh 0 shrec13

if (($# == 0)); then
   echo "Default: Training on shrec13 with gpu index 0"
   gpu_index=0
   dataset=shrec13
elif (($# == 1)); then
    echo "Training on shrec13 with specified gpu index"
    gpu_index=$1          
    dataset=shrec13
elif (($# == 2)); then
    echo "Training with specified dataset and gpu index"
    gpu_index=$1
    dataset=$2
fi

CUDA_VISIBLE_DEVICES=${gpu_index} python main.py \
    --phase "train" \
    --batch_size 30 \
    --margin 20 \
    --learning_rate "0.001" \
    --momentum "0.9" \
    --sketch_train_list ./dataset/${dataset}_trainList.txt \
    --sketch_test_list ./dataset/${dataset}_testList.txt \
    --shape_list ./dataset/${dataset}_shapeList.txt \
    --ckpt_dir ./checkpoint/${dataset} \
    --logdir ./logs/${dataset} \
    --num_views_shape 12 \
    --inputFeaSize 4096 \
    --outputFeaSize 256 \
    --maxiter 1000000
