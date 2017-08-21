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
    batch_size = len(poss) + len(negs)
    
    # all titles
    titles = poss + negs

    sp_feed = defaultdict(list)
    for i, (qry, titles) in enumerate(itertools.product(qrys, titles)):
        for j, word_id in enumerate(qry):
            sp_feed['qry_idx'].append([i, j])
            sp_feed['qry_val'].append(word_id)
        for j, word_id in enumerate(titles):
            sp_feed['pos_idx'].append([i, j])
            sp_feed['pos_val'].append(word_id)
    return (sp_feed['qry_idx'], sp_feed['qry_val'], [batch_size, seq_len]), \
           (sp_feed['pos_idx'], sp_feed['pos_val'], [batch_size, seq_len]), \


#test_file = '../data/3.test.negtive_sampled.ids'

mdl = LTRDNN(
    vocab_size=FLAGS.vocab_size,
    emb_dim=FLAGS.emb_dim,
    repr_dim=FLAGS.repr_dim,
    combiner=FLAGS.combiner,
    eps=FLAGS.eps)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
saver = tf.train.Saver()
mdl_ckpt_dir = './model_ckpt/'
ckpt = tf.train.get_checkpoint_state(mdl_ckpt_dir)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
saver.restore(sess, ckpt.model_checkpoint_path)

print 'environment done...'
for line in sys.stdin:
    line = line.rstrip("\r\n")
    #inst = line.split("\t")
    test_q, test_pt = eval_fn(line)
    #test_eval = mdl.predict_sim_qq(sess, test_q, test_pt)
    test_eval = mdl.predict_sim_qt(sess, test_q, test_pt)
    print test_eval

#acc = mdl.pairwise_accuracy(sess, feval, eval_fn)
#print 'pairwise accuracy:', acc

sess.close()
