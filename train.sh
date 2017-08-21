#!/bin/bash

python -u train.py \
--train_bs=128 \
--max_epoch=25 \
--eps=0.2 \
--eval_steps=20 \
--max_iter=1000 \
--train_file=./data_train_example.tsv \
--test_file=./data_test_example.tsv \
--valid_file=./data_test_example.tsv

