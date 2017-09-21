import numpy as np
import sys
import tensorflow as tf


class LTRDNN(object):
    """LTR-DNN model
    """
    def __init__(self, vocab_size, emb_dim=256, repr_dim=256,
                 combiner='sum', lr=1e-4, eps=1.0):
        """Construct network.
        """
        if combiner not in ['sum', 'mean']:
            raise Exception('invalid combiner')

        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.pretrained_emb = tf.placeholder(tf.float32, [vocab_size, emb_dim])
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
        self.embedding = tf.Variable(
            tf.random_uniform([vocab_size, emb_dim], -0.02, 0.02),
            name='emb_mat')
        # defined an assign embedding value op
        self.init_embedding = self.embedding.assign(self.pretrained_emb)

        # #shape of emb_qry: batch_size * emb_dim
        emb_qry = tf.nn.embedding_lookup_sparse(
            self.embedding, self.inp_qry, sp_weights=None, combiner=combiner)
        emb_pos = tf.nn.embedding_lookup_sparse(
            self.embedding, self.inp_pos, sp_weights=None, combiner=combiner)
        emb_neg = tf.nn.embedding_lookup_sparse(
            self.embedding, self.inp_neg, sp_weights=None, combiner=combiner)
        emb_prd = tf.nn.embedding_lookup_sparse(
            self.embedding, self.inp_prd, sp_weights=None, combiner=combiner)

        # construct fc layer to get repr of sentence
        w = tf.get_variable(
            'q-fc-W', shape=[emb_dim, repr_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[repr_dim]), name='b')
        with tf.name_scope('t_qry-vec'):
            # #shape of repr_qry: batch_size * repr_dim
            self.repr_qry = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_qry, w, b), name='repr_query')
            self.norm_qry = tf.nn.l2_normalize(self.repr_qry, dim=1)
        with tf.name_scope('t_prd-vec'):
            self.repr_prd = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_prd, w, b), name='repr_predq')
            self.norm_prd = tf.nn.l2_normalize(self.repr_prd, dim=1)

        w = tf.get_variable(
            't-fc-W', shape=[emb_dim, repr_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[repr_dim]), name='b')
        with tf.name_scope('q_pos-vec'):
            self.repr_pos = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_pos, w, b), name='repr_title_pos')
            self.norm_pos = tf.nn.l2_normalize(self.repr_pos, dim=1)
        with tf.name_scope('q_neg-vec'):
            self.repr_neg = tf.nn.softsign(
                tf.nn.xw_plus_b(emb_neg, w, b), name='repr_title_neg')
            self.norm_neg = tf.nn.l2_normalize(self.repr_neg, dim=1)

        # cosine similarity between q&p, q&n, q&q
        with tf.name_scope('sim-qp'):
            # #shape of sim_qp: batch_size * 1
            self.sim_qp = tf.reduce_sum(
                tf.multiply(self.norm_qry, self.norm_pos), axis=1)
        with tf.name_scope('sim-qn'):
            self.sim_qn = tf.reduce_sum(
                tf.multiply(self.norm_qry, self.norm_neg), axis=1)
        with tf.name_scope('sim-qq'):
            self.sim_qq = tf.reduce_sum(
                tf.multiply(self.norm_qry, self.norm_prd), axis=1)
        with tf.name_scope('diff_qp-qn'):
            self.sim_diff = tf.subtract(self.sim_qp, self.sim_qn)

        with tf.name_scope('label'):
            self.labels = tf.ones(shape=tf.shape(self.sim_diff))
        with tf.name_scope('pairwise-loss'):
            # calculate hinge loss
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
        self.opt = tf.contrib.opt.LazyAdamOptimizer(lr).minimize(
            self.total_loss, global_step=self.global_step)

        with tf.name_scope('pairwise-prediction'):
            # prediction
            # pred = 1 if sim_pos >= sim_neg else 0
            self.preds = tf.sign(tf.sign(self.sim_diff) + 1.)

        # @TODO: Add regularization like dropout, l2-reg, etc.

        with tf.name_scope('accuracy'):
            # accumulated accuracy
            # re-initialize local variables to conduct a new evaluation
            self.acc, self.update_acc = tf.contrib.metrics.streaming_accuracy(
                labels=self.labels, predictions=self.preds)

        # saver and loader
        # drop local variables of optimizer
        self.saver = tf.train.Saver(tf.trainable_variables())

    def train_step(self, sess, inp_batch_q, inp_batch_p, inp_batch_n):
        input_dict = {
            self.inp_qry: inp_batch_q,
            self.inp_pos: inp_batch_p,
            self.inp_neg: inp_batch_n,
            self.dropout_prob: 0.5}
        sess.run(self.opt, feed_dict=input_dict)

    def assign_embedding(self, sess, embedding=None):
        if embedding is None:
            raise Exception('embedding is None')
        input_dict = {self.pretrained_emb: embedding}
        sess.run(self.init_embedding, feed_dict=input_dict)

    def eval_step(self, sess, dev_qry, dev_pos, dev_neg, metrics=None):
        if not metrics:
            metrics = ['loss']
        eval_dict = {
            self.inp_qry: dev_qry,
            self.inp_pos: dev_pos,
            self.inp_neg: dev_neg,
            self.dropout_prob: 1.0}
        eval_res = []
        for metric in metrics:
            if metric == 'loss':
                eval_res.append(sess.run(self.loss, feed_dict=eval_dict))
        return eval_res

    def predict_sim(self, sess, query, title1, title2):
        """predict similarity between query&title1, query&title2, label
        @return: [sim_qt1, sim_qt2, 1.0/0.0]
        """
        eval_dict = {
            self.inp_qry: query,
            self.inp_pos: title1,
            self.inp_neg: title2,
            self.dropout_prob: 1.0}
        return sess.run([self.sim_qp, self.sim_qn, self.preds],
                        feed_dict=eval_dict)

    def predict_diff(self, sess, inp_qry, inp_pos, inp_neg):
        """predict which title is more similar to query.
        @return: 1.0/0.0 if first/second title is more similar.
        """
        pred_dict = {
            self.inp_qry: inp_qry,
            self.inp_pos: inp_pos,
            self.inp_neg: inp_neg,
            self.dropout_prob: 1.0}
        return sess.run(self.preds, feed_dict=pred_dict)

    def predict_sim_qt(self, sess, inp_query, inp_title):
        """predict similarity of query and title.
        @return: cosine similarity. value range [-1, 1].
        """
        pred_dict = {
            self.inp_qry: inp_query,
            self.inp_pos: inp_title,
            self.dropout_prob: 1.0}
        return sess.run(self.sim_qp, feed_dict=pred_dict)

    def predict_sim_qq(self, sess, inp_query1, inp_query2):
        """predict similarity of two queries.
        @return: cosine similarity. value range [-1, 1].
        """
        pred_dict = {
            self.inp_qry: inp_query1,
            self.inp_prd: inp_query2,
            self.dropout_prob: 1.0}
        return sess.run(self.sim_qq, feed_dict=pred_dict)

    def _accumulate_accuracy(self, sess, inp_q, inp_p, inp_n):
        """update accuracy by inputs and staged value.
        @return: newly-updated accuracy.
        """
        input_dict = {
            self.inp_qry: inp_q,
            self.inp_pos: inp_p,
            self.inp_neg: inp_n,
            self.dropout_prob: 1.0}
        sess.run(self.update_acc, feed_dict=input_dict)
        return sess.run(self.acc)

    def pairwise_accuracy(self, sess, fiter, inp_fn, verb=None):
        """evaluate the correct pairwise order ratio.
        @return: accuracy=(correct_pair/total_pair).
        @fiter : an iterable yielding instance (qry & pos & neg of each query).
        @inp_fn: a func extracting (qry, pos, neg) from instance, in which
                 qry, pos, neg are all batch-sentence that could be feed to
                 self.inp_X.
        @verb  : print progress hint every verb lines. None for no hint.
        """
        accuracy = None
        for nl, inst in enumerate(fiter):
            if verb and nl % verb == 0:  # print hint
                sys.stderr.write(str(verb) + ' lines in pairwise_acc.\n')
            qrys, poss, negs = inp_fn(inst)
            accuracy = self._accumulate_accuracy(sess, qrys, poss, negs)
        return accuracy
