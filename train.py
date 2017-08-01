#!/usr/bin/env python

import numpy as np
import sys
#sys.path.append('../../MLFlow/utils')
import tensorflow as tf
from ltr_dnn import LTRDNN
import dataproc

def inp_fn(data):
    q = []
    pt = []
    nt = []
    for inst in data:
        flds = inst.split('\t')
        if len(flds) < 3:
            continue
        query = map(int, flds[0].split(" "))
        pos_title = map(int, flds[1].split(" "))
        neg_title = map(int, flds[2].split(" "))
        q.append(dataproc.zero_padding(query, SEQ_LEN))
        pt.append(dataproc.zero_padding(pos_query, SEQ_LEN))
        nt.append(dataproc.zero_padding(neg_query), SEQ_LEN))
    return np.array(q), np.array(pt), np.array(nt)

train_file = './rt-polarity.shuf.train'
test_file = './rt-polarity.shuf.test'
freader = dataproc.BatchReader(train_file)
with open(test_file) as f:
    test_data = [x.rstrip('\n').split("\t") for x in f.readlines()]
test_q, test_pt, test_nt = inp_fn(test_data)

mdl = TextRNNClassifier(
    vocab_size = 1532783
    emb_dim=256,
    repr_dim=256,
    seq_len=300,
    combiner='sum',
    lr=1e-3,
    init_emb=None
    )

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
metrics = ['loss', 'auc']
niter = 0
mdl_ckpt_dir = './model_ckpt/model.ckpt'
while niter < 500:
    niter += 1
    batch_data = freader.get_batch(128)
    if not batch_data:
        break
    train_q, train_pt, train_nt = inp_fn(batch_data)
    mdl.train_step(sess, train_q, train_pt, train_nt)
    train_eval = mdl.eval_step(sess, train_q, train_pt, train_nt, metrics)
    test_eval = mdl.eval_step(sess, test_q, test_pt, test_nt, metrics) \
        if niter % 20 == 0 else 'SKIP'
    print niter, 'train:', train_eval, 'test:', test_eval
save_path = mdl.saver.save(sess, mdl_ckpt_dir, global_step=mdl.global_step)
print "model saved:", save_path
sess.close()
