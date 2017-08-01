import random
import numpy as np
import sys
import tensorflow as tf


class LTRDNN(object):
    """LTR-DNN model
    """
    def __init__(self, vocab_size, emb_dim=256, repr_dim=256,
                 seq_len=50, combiner='sum', lr=1e-3, eps=1.0,
                 init_emb=None):
        """Construct network.
        """
        if combiner not in ['sum', 'mean']:
            raise Exception('invalid combiner')

        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.eps = eps

        # prepare placeholder for query, pos, neg
        # https://www.tensorflow.org/api_docs/python/tf/sparse_placeholder
        # input is a batch_size*seq_len sparse tensor
        self.inp_qry = tf.sparse_placeholder(tf.int64, 'input_qry')
        self.inp_pos = tf.sparse_placeholder(tf.int64, 'input_pos')
        self.inp_neg = tf.sparse_placeholder(tf.int64, 'input_neg')

        # embedding from pretrained one or random one
        embedding = \
            tf.Variable(
                tf.convert_to_tensor(init_emb, dtype=tf.float32),
                name='emb_mat') if init_emb else \
            tf.Variable(
                tf.random_uniform([vocab_size, emb_dim], -0.2, 0.2),
                name='emb_mat')
        # #shape of emb_qry: batch_size * emb_dim
        emb_qry = tf.nn.embedding_lookup_sparse(
            embedding, self.inp_qry, combiner=combiner)
        emb_pos = tf.nn.embedding_lookup_sparse(
            embedding, self.inp_pos, combiner=combiner)
        emb_neg = tf.nn.embedding_lookup_sparse(
            embedding, self.inp_neg, combiner=combiner)

        # construct fc layer to get repr of sentence
        with tf.name_scope('query fc'):
            w = tf.get_variable(
                'W', shape=[emb_dim, repr_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[repr_dim]), name='b')
            # #shape of repr_qry: batch_size * repr_dim
            self.repr_qry = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_qry, w, b), name='repr_query')
        with tf.name_scope('title fc'):
            w = tf.get_variable(
                'W', shape=[emb_dim, repr_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[repr_dim]), name='b')
            self.repr_pos = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_pos, w, b), name='repr_title+')
            self.repr_neg = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_neg, w, b), name='repr_title-')

        # similarity between q&p, q&n, p&n
        # tf.losses.cosine_distance is not good to use here
        # #shape of norm_qry: batch_size * repr_dim
        self.norm_qry = tf.nn.l2_normalize(self.repr_qry, dim=1)
        self.norm_pos = tf.nn.l2_normalize(self.repr_pos, dim=1)
        self.norm_neg = tf.nn.l2_normalize(self.repr_neg, dim=1)
        # #shape of simi_pos: batch_size * 1
        self.sim_qp = tf.reduce_sum(
            tf.multiply(self.norm_qry, self.norm_pos), axis=1)
        self.sim_qn = tf.reduce_sum(
            tf.multiply(self.norm_qry, self.norm_neg), axis=1)
        self.sim_pn = tf.reduce_sum(
            tf.multiply(self.norm_pos, self.norm_neg), axis=1)

        # calculate hinge loss
        self.sim_diff = tf.substract(self.sim_qp, self.sim_qn)
        self.labels = tf.ones(shape=tf.shape(self.sim_pos))
        # modified hinge_loss = (1 / batch_size) * max(0, eps - sim_diff)
        self.loss = tf.losses.hinge_loss(
            labels=self.labels,
            logits=self.sim_diff / self.eps,
            reduction=tf.losses.Reduction.MEAN
            ) * self.eps
        self.total_loss = self.loss  # add reg-loss

        # optimizer
        self.opt = tf.train.AdamOptimizer(lr).minimize(
            self.total_loss, global_step=self.global_step)

        # @TODO: Add some metrics.
        # @TODO: Add regularization. Dropout, l2-reg, etc.

        # saver and loader
        self.saver = tf.train.Saver()

    def train_step(self, sess, inp_batch_q, inp_batch_p, inp_batch_n):
        input_dict = {
            self.inp_q: inp_batch_q,
            self.inp_p: inp_batch_p,
            self.inp_n: inp_batch_n}
        sess.run(self.opt, feed_dict=input_dict)

    def eval_step(self, sess, dev_qry, dev_pos, dev_neg, metrics=None):
        if not metrics:
            metrics = ['loss']
        eval_dict = {
            self.inp_qry: dev_qry,
            self.inp_pos: dev_pos,
            self.inp_neg: dev_neg}
        eval_res = []
        for metric in metrics:
            if metric == 'loss':
                eval_res.append(sess.run(self.loss, feed_dict=eval_dict))
        return eval_res

    def predict_sim_qt(self, sess, inp_query, inp_title):
        pred_dict = {
            self.inp_qry: inp_query,
            self.inp_pos: inp_title}
        return sess.run(self.sim_qp, feed_dict=pred_dict)

    def predict_sim_qq(self, sess, inp_title1, inp_title2):
        pred_dict = {
            self.inp_pos: inp_title1,
            self.inp_neg: inp_title2}
        return sess.run(self.sim_pn, feed_dict=pred_dict)
