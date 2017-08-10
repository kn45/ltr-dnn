import itertools
import numpy as np
import sys
import tensorflow as tf


class LTRDNN(object):
    """LTR-DNN model
    """
    def __init__(self, vocab_size, emb_dim=256, repr_dim=256,
                 combiner='sum', lr=1e-3, eps=1.0,
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
        self.inp_qry = tf.sparse_placeholder(dtype=tf.int64, name='input_qry')
        self.inp_pos = tf.sparse_placeholder(dtype=tf.int64, name='input_pos')
        self.inp_neg = tf.sparse_placeholder(dtype=tf.int64, name='input_neg')
        # use only when predicting sim-qq
        self.inp_prd = tf.sparse_placeholder(dtype=tf.int64, name='input_prd')

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
            embedding, self.inp_qry, sp_weights=None, combiner=combiner)
        emb_pos = tf.nn.embedding_lookup_sparse(
            embedding, self.inp_pos, sp_weights=None, combiner=combiner)
        emb_neg = tf.nn.embedding_lookup_sparse(
            embedding, self.inp_neg, sp_weights=None, combiner=combiner)
        emb_prd = tf.nn.embedding_lookup_sparse(
            embedding, self.inp_prd, sp_weights=None, combiner=combiner)

        # construct fc layer to get repr of sentence
        with tf.name_scope('query-fc'):
            w = tf.get_variable(
                'q-fc-W', shape=[emb_dim, repr_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[repr_dim]), name='b')
            # #shape of repr_qry: batch_size * repr_dim
            self.repr_qry = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_qry, w, b), name='repr_query')
            self.repr_prd = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_prd, w, b), name='repr_predq')
        with tf.name_scope('title-fc'):
            w = tf.get_variable(
                't-fc-W', shape=[emb_dim, repr_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[repr_dim]), name='b')
            self.repr_pos = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_pos, w, b), name='repr_title_pos')
            self.repr_neg = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_neg, w, b), name='repr_title_neg')

        # cosine similarity between q&p, q&n, q&q
        # #shape of norm_qry: batch_size * repr_dim
        self.norm_qry = tf.nn.l2_normalize(self.repr_qry, dim=1)
        self.norm_pos = tf.nn.l2_normalize(self.repr_pos, dim=1)
        self.norm_neg = tf.nn.l2_normalize(self.repr_neg, dim=1)
        self.norm_prd = tf.nn.l2_normalize(self.repr_prd, dim=1)
        # #shape of sim_qp: batch_size * 1
        self.sim_qp = tf.reduce_sum(
            tf.multiply(self.norm_qry, self.norm_pos), axis=1)
        self.sim_qn = tf.reduce_sum(
            tf.multiply(self.norm_qry, self.norm_neg), axis=1)
        self.sim_qq = tf.reduce_sum(
            tf.multiply(self.norm_qry, self.norm_prd), axis=1)

        # calculate hinge loss
        self.sim_diff = tf.subtract(self.sim_qp, self.sim_qn)
        self.labels = tf.ones(shape=tf.shape(self.sim_diff))
        # modified hinge_loss = (1 / batch_size) * max(0, eps - sim_diff)
        self.loss = tf.losses.hinge_loss(
            labels=self.labels,
            logits=self.sim_diff / self.eps,
            reduction=tf.losses.Reduction.MEAN
            ) * self.eps
        self.total_loss = self.loss  # add reg-loss

        # optimizer
        # kindly notice the efficiecy problem of Adam with sparse op:
        # https://github.com/tensorflow/tensorflow/issues/6460
        # self.opt = tf.contrib.opt.LazyAdamOptimizer(lr).minimize(
        #     self.total_loss, global_step=self.global_step)
        self.opt = tf.train.AdadeltaOptimizer(lr).minimize(
            self.total_loss, global_step=self.global_step)

        # prediction
        # pred = 1 if sim_pos >= sim_neg else 0
        self.preds = tf.sign(tf.sign(self.sim_diff) + 1.)

        # @TODO: Add some metrics.
        # @TODO: Add regularization like dropout, l2-reg, etc.

        # accumulated accuracy
        # re-initialize local variables to conduct a new evaluation
        self.acc, self.update_acc = tf.contrib.metrics.streaming_accuracy(
            labels=self.labels, predictions=self.preds)

        # saver and loader
        # drop local variables of optimizer
        self.saver = tf.train.Saver(tf.trainable_variables())

    def accumulate_accuracy(self, sess, inp_q, inp_p, inp_n):
        """update accuracy by inputs and staged value.
        @return: newly-updated accuracy.
        """
        input_dict = {
            self.inp_qry: inp_q,
            self.inp_pos: inp_p,
            self.inp_neg: inp_n}
        sess.run(self.update_acc, feed_dict=input_dict)
        return sess.run(self.acc)

    def train_step(self, sess, inp_batch_q, inp_batch_p, inp_batch_n):
        input_dict = {
            self.inp_qry: inp_batch_q,
            self.inp_pos: inp_batch_p,
            self.inp_neg: inp_batch_n}
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

    def predict_diff(self, sess, inp_qry, inp_pos, inp_neg):
        """predict which title is more similar to query.
        @return: 1.0/0.0 if first/second title is more similar.
        """
        pred_dict = {
            self.inp_qry: inp_qry,
            self.inp_pos: inp_pos,
            self.inp_neg: inp_neg}
        return sess.run(self.preds, feed_dict=pred_dict)

    def predict_sim_qt(self, sess, inp_query, inp_title):
        """predict similarity of query and title.
        @return: cosine similarity. value range [-1, 1].
        """
        pred_dict = {
            self.inp_qry: inp_query,
            self.inp_pos: inp_title}
        return sess.run(self.sim_qp, feed_dict=pred_dict)

    def predict_sim_qq(self, sess, inp_query1, inp_query2):
        """predict similarity of two queries.
        @return: cosine similarity. value range [-1, 1].
        """
        pred_dict = {
            self.inp_qry: inp_query1,
            self.inp_prd: inp_query2}
        return sess.run(self.sim_qq, feed_dict=pred_dict)

    def pairwise_accuracy(self, sess, fiter, inp_fn):
        """evaluate the correct pairwise order ratio.
        @return: correct_pair/ total_pair
        @fiter:  an iterable to fetch instance (qry&pos&neg of each query)
        @inp_fn: a func extracting ([qry], [pos], [neg]) from instance
        """
        accuracy = None
        for n, inst in enumerate(fiter):
            qrys, poss, negs = inp_fn(inst)
            for qry, pos, neg in itertools.product(qrys, poss, negs):
                accuracy = self.accumulate_accuracy(sess, qry, pos, neg)
        return accuracy
