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


# a = tf.placeholder(dtype=tf.int32, shape=[2, 3], name='a')
# for i in range(5):
#     a = a + 1
# c = a + 1
# d = np.zeros((2, 3), dtype=np.int32)
# with tf.Session() as sess:
#     print(sess.run(c, feed_dict={a: d}))
# # 直接操作占位符不会出错，但是值不会改变。


# a = np.random.randint(10, size=(10, 100))
# def gen():
#     for i in range(a.shape[1]//10):
#         yield (a[:, 10*i:10*(i+1)], a[:, 10*i+1:10*(i+1)+1])
# for i in gen():
#     print(i)
# # 最后一个batch缺少一列，T=9。


# a = np.random.randint(10, size=(3, 4))
# b = np.array([range(i, 10+i) for i in range(10)])
# c = b[a]
# print(a)
# print('---------------------------')
# print(b)
# print('---------------------------')
# print(c)


state_size = 10
num_classes = 5
T = 20
embedding = np.random.randint(10, size=(5, 10))
w = tf.constant(embedding, dtype=tf.float32)
feed = np.random.randint(5, size=(20,))
input = tf.placeholder(dtype=tf.int32, shape=[20,])
one_hot = tf.one_hot(input, num_classes)
output1 = tf.matmul(one_hot, w)
output2 = tf.nn.embedding_lookup(params=w, ids=input)

with tf.Session() as sess:
    o1 = sess.run(output1, feed_dict={input: feed})
    print(o1)
    print('--------------------------------------------')
    o2 = sess.run(output2, feed_dict={input: feed})
    print(o2)
    print('--------------------------------------------')
    print(o1 == o2)
# embedding_lookup(params, ids)相当于numpy.array中的params[ids]操作
# 从网络的角度看，这里相当于x先转换为one_hot向量，再乘上权重embedding_matrix

