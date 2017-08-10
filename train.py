#!/usr/bin/env python

import dataproc
import numpy as np
import sys
import random
import tensorflow as tf
from ltrdnn import LTRDNN


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
flags.DEFINE_integer('max_iter', 1000, 'max iteration')
flags.DEFINE_float('eps', 1.0, 'zero-loss threshold epsilon in hinge loss')
flags.DEFINE_integer('eval_steps', 20, 'every how many steps to evaluate')


def ramdon_choose(l):
    rand_obj = random.sample(l, 1)[0]
    return rand_obj


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
        pos_title_num = int(flds[1])
        pos_titles = flds[2:2+pos_title_num]
        neg_title_num = int(flds[2+pos_title_num])
        neg_titles = flds[2+pos_title_num+1:]

        pos_title = ramdon_choose(pos_titles)
        pos_title = map(int, pos_title.split(' '))
        neg_title = ramdon_choose(neg_titles)
        neg_title = map(int, neg_title.split(' '))

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


def eval_fn(inst):
    get_len = lambda lst: max([len(x) for x in lst])
    flds = inst.split('\t')
    queries = flds[0:1]
    pos_title_num = int(flds[1])
    pos_titles = flds[2:2+pos_title_num]
    neg_title_num = int(flds[2+pos_title_num])
    neg_titles = flds[2+pos_title_num+1:]

    queries = [map(int, x.split(' ')) for x in queries]
    pos_titles = [map(int, x.split(' ')) for x in pos_titles]
    neg_titles = [map(int, x.split(' ')) for x in neg_titles]
    seq_len = max(get_len(queries), get_len(pos_titles), get_len(neg_titles))

    qry_out = []
    for query in queries:
        qry_idx = []
        qry_val = []
        for j, word_id in enumerate(query):
            qry_idx.append([0, j])
            qry_val.append(word_id)
        qry_out.append((qry_idx, qry_val, [1, seq_len]))
    pos_out = []
    for pos_title in pos_titles:
        pos_idx = []
        pos_val = []
        for j, word_id in enumerate(pos_title):
            pos_idx.append([0, j])
            pos_val.append(word_id)
        pos_out.append((pos_idx, pos_val, [1, seq_len]))
    neg_out = []
    for neg_title in neg_titles:
        neg_idx = []
        neg_val = []
        for j, word_id in enumerate(neg_title):
            neg_idx.append([0, j])
            neg_val.append(word_id)
        neg_out.append((neg_idx, neg_val, [1, seq_len]))
    return qry_out, pos_out, neg_out


# train_file = './3.train.negtive_sampled.ids'
# test_file = './3.train.negtive_sampled.ids.pair'
# test_file = './data/3.test.negtive_sampled.ids'
train_file = './data_train_example.tsv'
valid_file = './data_test_example.tsv'
test_file = './data_test_example.tsv'
train_freader = dataproc.BatchReader(train_file, max_epoch=FLAGS.max_epoch)
with open(valid_file) as f:
    valid_data = [x.rstrip('\n') for x in f.readlines()]
valid_q, valid_pt, valid_nt = inp_fn(valid_data)

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
    valid_eval = mdl.eval_step(sess, valid_q, valid_pt, valid_nt, metrics) \
        if niter % FLAGS.eval_steps == 0 else 'SKIP'
    print niter, 'train_loss:', train_eval, 'valid_loss:', valid_eval

feval = open(test_file)
acc = mdl.pairwise_accuracy(sess, feval, eval_fn)
print 'pairwise accuracy:', acc

save_path = mdl.saver.save(sess, mdl_ckpt_dir, global_step=mdl.global_step)
print 'model saved:', save_path

sess.close()
feval.close()
