#!/bin/bash
clear

lossType="weightedFeaContrastiveLoss"

CUDA_VISIBLE_DEVICES=$1 py_gxdai main.py \
    --phase $2 \
    --batch_size 30 \
    --margin 20 \
    --learning_rate "0.001" \
    --momentum "0.9" \
    --lamb 5 \
    --sketch_train_list ./trainList.txt \
    --sketch_test_list ./testList.txt \
    --shape_list ./shapeList.txt \
    --ckpt_dir ./checkpoint/${lossType} \
    --logdir ./logs/${lossType} \
    --lossType $lossType \
    --activationType relu \
    --normFlag 0 \
    --num_views_sketch 1 \
    --num_views_shape 12 \
    --inputFeaSize 4096 \
    --outputFeaSize 256 \
    --maxiter 1000000 \
    --weightFile "checkpoint/weightedFeaContrastiveLoss/model-15000"
