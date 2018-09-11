import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
import time

num_steps = 10
batch_size = 200
num_classes = 2
state_size = 16
learning_rate = 0.1

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

def gen_batch(raw_data, batch_size, num_step):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)
    batch_patition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_patition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_patition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_patition_length * i : batch_patition_length * (i+1)]
        data_y[i] = raw_y[batch_patition_length * i : batch_patition_length * (i+1)]
    epoch_size = batch_patition_length // num_steps
    for i in range(epoch_size):
        x = data_x[:, i * num_steps : (i+1) * num_steps]
        y = data_y[:, i * num_steps : (i+1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)

# 网络中的占位符
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
init_state = tf.placeholder(tf.float32, [batch_size, state_size], name='init_state')

# 将输入转为one_hot，再unstack成列表
x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1)

# 构建网络的rnn部分

# # 1.手动实现rnn正向传播：
# with tf.variable_scope('rnn_cell'):
#     w = tf.get_variable('w', [num_classes + state_size, state_size])
#     b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
#
# def rnn_cell(rnn_input, state):
#     with tf.variable_scope('rnn_cell', reuse=True):
#         w = tf.get_variable('w', [num_classes + state_size, state_size])
#         b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
#     return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), w) + b)
#
# state = init_state
# rnn_outputs = []
# for rnn_input in rnn_inputs:
#     state = rnn_cell(rnn_input, state)
#     rnn_outputs.append(state)
# final_state = rnn_outputs[-1]

# # 2.使用tensorflow自带static_rnn实现：
# cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)
# rnn_outputs, final_state = tf.nn.static_rnn(cell=cell, inputs=rnn_inputs, initial_state=init_state)

# 3.使用tensorflow自带dynamic_rnn实现：
cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)
rnn_outputs_, final_state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=init_state)
rnn_outputs = tf.unstack(rnn_outputs_, axis=1)

# 构建网络的输出部分
with tf.variable_scope('softmax'):
    w = tf.get_variable('w', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes])
logits = [tf.matmul(rnn_output, w) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# 计算loss
y_as_list = tf.unstack(y, num=num_steps, axis=1)
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for\
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)

# 方便后面训练网络
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

def train_rnn(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())# 初始化权重
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):# 循环n个epoch
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print('\nepoch', idx)
            for step, (X, Y) in enumerate(epoch):# 循环1个epoch中的epoch_size个小小序列
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses, total_loss, final_state, train_step],
                             feed_dict={x: X, y: Y, init_state: training_state})# 这里training_state循环使用
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print('第 {0} 步的平均损失 {1}'.format(step, training_loss/100))
                    training_losses.append(training_loss/100)
                    training_loss = 0
    return training_losses

start_time = time.time()
training_losses = train_rnn(num_epochs=20, num_steps=num_steps, state_size=state_size)
print('训练耗时： ', time.time()-start_time)
print(training_losses[0])
plt.plot(training_losses)
plt.show()


# 1. 数据流细节：
# 输入：1000,000个数据的序列
# 输出：1000,000个数据的序列
# 模型：通过输入的数据，预测输出的数据，t时刻的输出数据与t-3和t-8时刻的输入数据有关
# 将1000,000长的序列分为200个小序列 -> (200, 5000)
# 把这些小序列当作互不相关的T=5000的序列数据
# 设定截断长度为10，因此每一个小序列分为500份小小序列，每份长度为10，这些序列我们想要保留他们的关系，只是因为截断反向传播将其分开
# 由于小序列之间互不相关，因此可以放在一个batch里同时训练 -> batch_size=200
# 所以每次输入(200, 10)的数据正向和反向传播，得到这个小小序列的losses和outputs
# 取最后一个output(即final_state)，作为下一个小小序列的initial_state继续进行正向和反向传播
# 像这样传播和训练500次，所有的数据都遍历了一遍，称为一个epoch -> epoch_size=500


# 2. tensorflow_rnn总结：
# 1)tensorflow中实现的主要是rnn_cell和正向传播的方法，对应于tf.contrib.rnn.BasicRNNCell和
#   tf.nn.static_rnn/tf.nn.dynamic_rnn。
# 2)static_rnn的输入为[tensor(batch_size, num_classes) for t in range(T)]，输出为
#   [tensor(batch_size, state_size) for t in range(T)]，遍历列表即可计算losses。
#   dynamic_rnn的输入为tensor(batch_size, T, num_classes)，输出为tensor(batch_size, T,
#   state_size)，需要先用tf.unstack转换为[tensor(batch_size, state_size) for t in range(T)]形
#   的列表，然后遍历计算即可。
# 3)由于dynamic_rnn可以接受长度不等的序列输入，据说计算效率也更高，因此优先使用dynamic_rnn。
#   但是我统计训练耗时的结果是：5 epochs时，dynamic_rnn用时22.3s，static_rnn用时12.9s，手写rnn用时12.4s
#                          20 epochs时，dynamic_rnn用时88.8s，static_rnn用时49.1s
#   可见，这个模型中static_rnn是快于dynamic_rnn的。

























