import tensorflow as tf

sess = tf.Session()
eps = 0.12
logits = tf.constant([0.1, 0.5])
labels = tf.constant([1., 1.])

loss = tf.losses.hinge_loss(
    labels=labels, logits=logits/eps,
    reduction=tf.losses.Reduction.MEAN) * eps
print sess.run(loss)  # ((0.12 - 0.1) + 0.) /2
sess.close()
