import itertools
import numpy as np
import sys
import tensorflow as tf


class LTRDNN(object):
    def __init__(self):
        self.labels = tf.placeholder(tf.float32, [None, 1], 'a')
        self.preds = tf.placeholder(tf.float32, [None, 1], 'b')

        # accumulated accuracy
        # re-initialize local variables to conduct a new evaluation
        self.acc, self.update_acc = tf.contrib.metrics.streaming_accuracy(
            labels=self.labels, predictions=self.preds)

    def accumulate_accuracy(self, sess, l, p):
        """update accuracy by inputs and staged value.
        @return: newly-updated accuracy.
        """
        input_dict = {
            self.labels: l,
            self.preds: p}
        sess.run(self.update_acc, feed_dict=input_dict)
        return sess.run(self.acc)

    def pairwise_accuracy(self, sess, fiter, inp_fn):
        """evaluate the correct pairwise order ratio.
        @return: correct_pair/ total_pair
        @fiter:  an iterable to fetch instance (qry&pos&neg of each query)
        @inp_fn: a func extracting ([qry], [pos], [neg]) from instance
        """
        accuracy = None
        for inst in fiter:
            qrys, poss, negs = inp_fn(inst)
            for qry, pos, neg in itertools.product(qrys, poss, negs):
                accuracy = self.accumulate_accuracy(sess, qry, pos, neg)
        return accuracy

l1 = [[1.0], [1.0], [1.0], [1.0]]
p1 = [[1.0], [1.0], [1.0], [0.0]]

l2 = [[1.0], [1.0], [1.0], [1.0]]
p2 = [[1.0], [0.0], [0.0], [0.0]]


mdl = LTRDNN()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
print mdl.accumulate_accuracy(sess, l1, p1)
print mdl.accumulate_accuracy(sess, l2, p2)
sess.close()
