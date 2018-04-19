#!/bin/bash
clear
# set the dataset for training
dataset=shrec13

CUDA_VISIBLE_DEVICES=$1 python main.py \
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
