#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING="/aidata/pengfei/datasets/dtu/"

python train.py --dataset dtu_yao --batch_size 4 --epochs 8 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 9 9 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--lr 0.001 \
--loadckpt ./checkpoints/model_000007.ckpt \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/val.txt \
--logdir ./checkpoints $@
