import tensorflow as tf
import numpy as np

#
# a = tf.placeholder(dtype=tf.int32, shape=[2, 3], name='a')
# b = a
# for i in range(5):
#     b = b + 1
# c = b * 2
# d = np.zeros((2, 3), dtype=np.int32)
# with tf.Session() as sess:
#     print(sess.run(c, feed_dict={a: d}))


a = tf.placeholder(dtype=tf.int32, shape=[2, 3], name='a')
for i in range(5):
    a = a + 1
c = a + 1
d = np.zeros((2, 3), dtype=np.int32)
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: d}))
