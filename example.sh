#!/bin/bash

python train.py \
--train_bs=128 \
--max_epoch=50 \
--eps=0.5 \
--eval_steps=1 \
--max_iter=100000

