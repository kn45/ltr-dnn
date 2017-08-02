#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../MLFlow/utils')
import tensorflow as tf
from ltrdnn import LTRDNN
import dataproc


def inp_fn(data):
    q_indices = []
    pt_indices = []
    nt_indices = []
    q_values = []
    pt_values = []
    nt_values = []
    batch_size = len(data)
    for i, inst in enumerate(data):
        flds = inst.split('\t')
        query = map(int, flds[0].split(' '))
        pos_title = map(int, flds[1].split(' '))
        neg_title = map(int, flds[2].split(' '))
        for j, word_id in enumerate(query):
            q_indices.append([i, j])
            q_values.append(word_id)
        for j, word_id in enumerate(pos_title):
            pt_indices.append([i, j])
            pt_values.append(word_id)
        for j, word_id in enumerate(neg_title):
            nt_indices.append([i, j])
            nt_values.append(word_id)
    return (q_indices, q_values, [batch_size, FLAGS.seq_len]), \
           (pt_indices, pt_values, [batch_size, FLAGS.seq_len]), \
           (nt_indices, nt_values, [batch_size, FLAGS.seq_len])

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('seq_len', 30, 'max seqence length')
flags.DEFINE_integer('train_bs', 16, 'train batch size')
flags.DEFINE_integer('max_epoch', 1, 'max epoch')
flags.DEFINE_integer('max_iter', 100, 'max iteration')


train_file = './data_train_example.tsv'
test_file = './data_train_example.tsv'
train_freader = dataproc.BatchReader(train_file, max_epoch=FLAGS.max_iter)
with open(test_file) as f:
    test_data = [x.rstrip('\n') for x in f.readlines()][0: 10]
test_q, test_pt, test_nt = inp_fn(test_data)

mdl = LTRDNN(
    vocab_size=1532783,
    emb_dim=256,
    repr_dim=256,
    seq_len=FLAGS.seq_len,
    combiner='sum',
    lr=1e-3,
    eps=0.5)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
metrics = ['loss']
mdl_ckpt_dir = './model_ckpt/model.ckpt'
for niter in xrange(FLAGS.max_iter):
    batch_data = train_freader.get_batch(FLAGS.train_bs)
    if not batch_data:
        break
    train_q, train_pt, train_nt = inp_fn(batch_data)
    mdl.train_step(sess, train_q, train_pt, train_nt)
    train_eval = mdl.eval_step(sess, train_q, train_pt, train_nt, metrics)
    test_eval = mdl.eval_step(sess, test_q, test_pt, test_nt, metrics) \
        if niter % 20 == 0 else 'SKIP'
    print niter, 'train:', train_eval, 'test:', test_eval
save_path = mdl.saver.save(sess, mdl_ckpt_dir, global_step=mdl.global_step)
print 'model saved:', save_path
sess.close()
