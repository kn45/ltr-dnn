#!/bin/bash

python train.py \
--train_bs=128 \
--max_epoch=25 \
--eps=0.2 \
--eval_steps=1000 \
--max_iter=1200000

#--max_iter=2400000
