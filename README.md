# LTR-DNN

A tensorflow implementation of LTR DNN semantic similarity model.  

Tensorflow v1.2 on CPU is used for testing.

## Example:

```shell
# train
python -u train.py \
--train_bs=128 \
--max_epoch=25 \
--eps=0.2 \
--eval_steps=20 \
--max_iter=1000 \
--embedding_file=./data/words_embedding \
--train_file=./data/data_train_example.tsv \
--test_file=./data/data_test_example.tsv \
--valid_file=./data/data_test_example.tsv

# predict
python -u predict.py \
--predict_file=./data/data_predict_example.tsv
```


## Reference:

- It's an LTR improvement of [DSSM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)


