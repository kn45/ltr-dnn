#!/usr/bin/env python

import dataproc
import itertools
import numpy as np
import random
import time
import sys
import tensorflow as tf
from collections import defaultdict
from ltrdnn import LTRDNN
import utils


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


def inp_fn(data):
    def _random_choose(l): return random.sample(l, 1)[0]
    sp_feed = defaultdict(list)
    batch_size = len(data)
    seq_len = 0
    for i, inst in enumerate(data):
        flds = inst.split('\t')
        query = map(int, flds[0].split(' '))
        pos_title_num = int(flds[1])
        pos_titles = flds[2:2+pos_title_num]
        neg_title_num = int(flds[2+pos_title_num])
        neg_titles = flds[2+pos_title_num+1:]

        pos_title = _random_choose(pos_titles)
        pos_title = map(int, pos_title.split(' '))
        neg_title = _random_choose(neg_titles)
        neg_title = map(int, neg_title.split(' '))

        seq_len = max(seq_len, len(query), len(pos_title), len(neg_title))
        for j, word_id in enumerate(query):
            sp_feed['qry_idx'].append([i, j])
            sp_feed['qry_val'].append(word_id)
        for j, word_id in enumerate(pos_title):
            sp_feed['pos_idx'].append([i, j])
            sp_feed['pos_val'].append(word_id)
        for j, word_id in enumerate(neg_title):
            sp_feed['neg_idx'].append([i, j])
            sp_feed['neg_val'].append(word_id)
    return (sp_feed['qry_idx'], sp_feed['qry_val'], [batch_size, seq_len]), \
           (sp_feed['pos_idx'], sp_feed['pos_val'], [batch_size, seq_len]), \
           (sp_feed['neg_idx'], sp_feed['neg_val'], [batch_size, seq_len])


def eval_fn(inst):
    def _max_len(lst): return max([len(x) for x in lst])
    flds = inst.split('\t')
    qrys = flds[0:1]
    pos_num = int(flds[1])
    poss = flds[2:2+pos_num]
    neg_num = int(flds[2+pos_num])
    negs = flds[2+pos_num+1:]
    qrys = [map(int, x.split(' ')) for x in qrys]
    poss = [map(int, x.split(' ')) for x in poss]
    negs = [map(int, x.split(' ')) for x in negs]
    seq_len = max(_max_len(qrys), _max_len(poss), _max_len(negs))
    batch_size = len(qrys) * len(poss) * len(negs)

    sp_feed = defaultdict(list)
    for i, (qry, pos, neg) in enumerate(itertools.product(qrys, poss, negs)):
        for j, word_id in enumerate(qry):
            sp_feed['qry_idx'].append([i, j])
            sp_feed['qry_val'].append(word_id)
        for j, word_id in enumerate(pos):
            sp_feed['pos_idx'].append([i, j])
            sp_feed['pos_val'].append(word_id)
        for j, word_id in enumerate(neg):
            sp_feed['neg_idx'].append([i, j])
            sp_feed['neg_val'].append(word_id)
    return (sp_feed['qry_idx'], sp_feed['qry_val'], [batch_size, seq_len]), \
           (sp_feed['pos_idx'], sp_feed['pos_val'], [batch_size, seq_len]), \
           (sp_feed['neg_idx'], sp_feed['neg_val'], [batch_size, seq_len])


print 'environment done...'
sys.stdout.flush()
train_file = '../data/3.train.negtive_sampled.ids'
test_file = '../data/3.test.negtive_sampled.ids'
valid_file = '../data/3.test.negtive_sampled.ids'
init_model = '../split_model/'
#train_file = '/mnt/yardcephfs/mmyard/g_wxg_fd_search/maricoliao/tensorflow/ltr-dnn/data/3.train.negtive_sampled.ids'
#test_file = '/mnt/yardcephfs/mmyard/g_wxg_fd_search/maricoliao/tensorflow/ltr-dnn/data/3.test.negtive_sampled.ids'
#valid_file = '/mnt/yardcephfs/mmyard/g_wxg_fd_search/maricoliao/tensorflow/ltr-dnn/data/3.test.negtive_sampled.ids'
train_freader = dataproc.BatchReader(train_file, max_epoch=FLAGS.max_epoch)
with open(valid_file) as f:
    valid_data = [x.rstrip('\n') for x in f.readlines()]
valid_q, valid_pt, valid_nt = inp_fn(valid_data)

init_model = utils.load_model(init_model)
print 'load init model done...'

mdl = LTRDNN(
    vocab_size=FLAGS.vocab_size,
    emb_dim=FLAGS.emb_dim,
    repr_dim=FLAGS.repr_dim,
    combiner=FLAGS.combiner,
    eps=FLAGS.eps,
    init_emb=init_model.emb)
print 'init mdl model done...'
print init_model.emb.shape
init_model.emb = None
print type(init_model.emb)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
metrics = ['loss']
mdl_ckpt_dir = './model_ckpt/model.ckpt'
print 'train begin...'
sys.stdout.flush()
for niter in xrange(FLAGS.max_iter):
    batch_data = train_freader.get_batch(FLAGS.train_bs)
    if not batch_data:
        break
    train_q, train_pt, train_nt = inp_fn(batch_data)
    mdl.train_step(sess, train_q, train_pt, train_nt)
    train_eval = mdl.eval_step(sess, train_q, train_pt, train_nt, metrics)
    valid_eval = mdl.eval_step(sess, valid_q, valid_pt, valid_nt, metrics) \
        if niter % FLAGS.eval_steps == 0 else 'SKIP'
    if niter % FLAGS.eval_steps == 0:
        now_time = time.strftime('%Y%m%d_%H:%M:%S',time.localtime(time.time()))
        print  now_time, niter, 'train_loss:', train_eval, 'valid_loss:', valid_eval
        sys.stdout.flush()

save_path = mdl.saver.save(sess, mdl_ckpt_dir, global_step=mdl.global_step)
print 'model saved:', save_path

feval = open(test_file)
acc = mdl.pairwise_accuracy(sess, feval, eval_fn)
print 'pairwise accuracy:', acc

sess.close()
feval.close()
