#!/usr/bin/env python

import numpy as np
import sys
import tensorflow as tf
from ltrdnn import LTRDNN
sys.path.append('../MLFlow/utils')
import dataproc


flags = tf.flags
FLAGS = flags.FLAGS
# model related:
flags.DEFINE_integer('vocab_size', 1532783, 'vocabulary size')
flags.DEFINE_integer('emb_dim', 256, 'embedding dimension')
flags.DEFINE_integer('repr_dim', 256, 'sentence representing dimension')
flags.DEFINE_string('combiner', 'sum', 'how to combine words in a sentence')
# training related:
flags.DEFINE_integer('train_bs', 128, 'train batch size')
flags.DEFINE_integer('max_epoch', 1, 'max epoch')
flags.DEFINE_integer('max_iter', 100, 'max iteration')
flags.DEFINE_float('eps', 1.0, 'zero-loss threshold epsilon in hinge loss')
flags.DEFINE_integer('eval_steps', 20, 'every how many steps to evaluate')


def inp_fn(data):
    q_indices = []
    pt_indices = []
    nt_indices = []
    q_values = []
    pt_values = []
    nt_values = []
    batch_size = len(data)
    seq_len = 0
    for i, inst in enumerate(data):
        flds = inst.split('\t')
        query = map(int, flds[0].split(' '))
        pos_title = map(int, flds[1].split(' '))
        neg_title = map(int, flds[2].split(' '))
        seq_len = max(seq_len, len(query), len(pos_title), len(neg_title))
        for j, word_id in enumerate(query):
            q_indices.append([i, j])
            q_values.append(word_id)
        for j, word_id in enumerate(pos_title):
            pt_indices.append([i, j])
            pt_values.append(word_id)
        for j, word_id in enumerate(neg_title):
            nt_indices.append([i, j])
            nt_values.append(word_id)
    return (q_indices, q_values, [batch_size, seq_len]), \
           (pt_indices, pt_values, [batch_size, seq_len]), \
           (nt_indices, nt_values, [batch_size, seq_len])


train_file = './data_train_example.tsv'
test_file = './data_train_example.tsv'
train_freader = dataproc.BatchReader(train_file, max_epoch=FLAGS.max_iter)
with open(test_file) as f:
    test_data = [x.rstrip('\n') for x in f.readlines()][0: 10]
test_q, test_pt, test_nt = inp_fn(test_data)

mdl = LTRDNN(
    vocab_size=FLAGS.vocab_size,
    emb_dim=FLAGS.emb_dim,
    repr_dim=FLAGS.repr_dim,
    combiner=FLAGS.combiner,
    eps=FLAGS.eps)

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
        if niter % FLAGS.eval_steps == 0 else 'SKIP'
    pred_diff = mdl.predict_diff(sess, train_q, train_pt, train_nt)
    pred_qt = mdl.predict_sim_qt(sess, train_q, train_pt)
    pred_qq = mdl.predict_sim_qq(sess, train_q, train_q)
    print niter, 'train_loss:', train_eval, 'test_loss:', test_eval, \
        'diff 0/1:', pred_diff, 'sim_qt:', pred_qt, 'sim_qq:', pred_qq
save_path = mdl.saver.save(sess, mdl_ckpt_dir, global_step=mdl.global_step)
print 'model saved:', save_path
sess.close()
