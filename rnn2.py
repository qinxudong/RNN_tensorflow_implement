import numpy as np
import tensorflow as tf
import time
from LayerNormalizedLSTMCell import LayerNormalizedLSTMCell
import matplotlib.pyplot as plt


# file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name1 = 'tinyshakespeare.txt'
file_name2 = 'variousscripts.txt'
with open(file_name2, 'r') as f:
    raw_data = f.read()
    print('数据长度', len(raw_data))

vocab = set(raw_data)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
data = [vocab_to_idx[c] for c in raw_data]
del raw_data

num_steps = 80
batch_size = 64
state_size = 512
num_classes = vocab_size
learning_rate = 5e-4
keep_prob = 0.9
num_layers = 3

def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len*i:batch_len*(i+1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError()
    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

def gen_epochs(num_epochs, num_steps, batch_size):
    for i in range(num_epochs):
        yield ptb_iterator(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def build_basic_rnn_graph_with_list(
        state_size = state_size,
        num_classes = num_classes,
        batch_size = batch_size,
        num_steps = num_steps,
        learning_rate = learning_rate,
        keep_prob = keep_prob
):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')

    x_one_hot = tf.one_hot(x, num_classes)
    rnn_inputs = tf.unstack(x_one_hot, axis=1)
    # rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
    rnn_inputs = [tf.nn.dropout(rnn_input, keep_prob) for rnn_input in rnn_inputs]
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    # with tf.Session() as sess:
    #     print(sess.run(init_state))
    rnn_outputs, final_state = tf.nn.static_rnn(cell=cell, inputs=rnn_inputs, initial_state=init_state)
    rnn_outputs = [tf.nn.dropout(rnn_output, keep_prob) for rnn_output in rnn_outputs]
    with tf.variable_scope('softmax'):
        w = tf.get_variable('w', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, w) + b for rnn_output in rnn_outputs]
    y_as_list = tf.unstack(y, axis=1)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_as_list, logits=logits)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return dict(
        x=x,
        y=y,
        keep_prob=keep_prob,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step
    )

# g = build_basic_rnn_graph_with_list()
# train_rnn(g, 5)


def build_multilayer_lstm_graph_with_static_rnn(
        state_size=state_size,
        num_classes=num_classes,
        batch_size=batch_size,
        num_steps=num_steps,
        num_layers=num_layers,
        learning_rate=learning_rate,
        keep_prob=keep_prob
):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')

    # 在x输入RNN之前加入一个embedding层，将x_t转换为长度为state_size的向量
    embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)# [batch_size, num_steps, state_size]
    # embedding_lookup(params, ids)相当于numpy.array中的params[ids]操作
    # 从网络的角度看，这里相当于x先转换为one_hot向量，再乘上权重embedding_matrix
    rnn_inputs_as_list = tf.unstack(rnn_inputs, axis=1)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)# state_is_tuple=True时返回(c_state, m_state)的tuple
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.static_rnn(cell=cell, inputs=rnn_inputs_as_list, initial_state=init_state)

    with tf.variable_scope('softmax'):
        w = tf.get_variable('w', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    y_as_list = tf.unstack(y, axis=1)
    logits = [tf.matmul(rnn_output, w) + b for rnn_output in rnn_outputs]
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=y_as_list))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x=x,
                y=y,
                init_state=init_state,
                final_state=final_state,
                total_loss=total_loss,
                train_step=train_step)


def build_multilayer_lstm_graph_with_dynamic_rnn(
        state_size=state_size,
        num_classes=num_classes,
        batch_size=batch_size,
        num_steps=num_steps,
        num_layers=num_layers,
        learning_rate=learning_rate,
        keep_prob=keep_prob
):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')

    # 在x输入RNN之前加入一个embedding层，将x_t转换为长度为state_size的向量
    embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)# [batch_size, num_steps, state_size]
    # embedding_lookup(params, ids)相当于numpy.array中的params[ids]操作
    # 从网络的角度看，这里相当于x先转换为one_hot向量，再乘上权重embedding_matrix

    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)# state_is_tuple=True时返回(c_state, m_state)的tuple
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        w = tf.get_variable('w', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshape = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, w) + b
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=y_reshape))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x=x,
                y=y,
                init_state=init_state,
                final_state=final_state,
                total_loss=total_loss,
                train_step=train_step)

# start_time = time.time()
# g = build_multilayer_lstm_graph_with_dynamic_rnn()
# # g = build_multilayer_lstm_graph_with_static_rnn()
# print('构建图耗时： ', time.time() - start_time)
# start_time = time.time()
# train_rnn(g, 3)
# print('训练耗时： ', time.time() - start_time)


def build_multilayer_lstm_graph_with_scan(
        state_size=state_size,
        num_classes=num_classes,
        batch_size=batch_size,
        num_steps=num_steps,
        num_layers=num_layers,
        learning_rate=learning_rate,
        keep_prob=keep_prob
):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')

    embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # LSTMCell: (state_is_tuple=True情况下)
    #           输入输出：cell(input, state) -> (output, new_state)
    #           input为一个时间步长的input， state为c_state和m_state组成的tuple
    #           input.shape==(batch_size, input_size),
    #           state.shape==((batch_size, state_size), (batch_size, state_size))
    #           output为一个时间步长的output，new_state为新的c_state和m_state组成的tuple
    #           output.shape==(batch_size, state_size),
    #           new_state.shape==((batch_size, state_size), (batch_size, state_size))
    #           因此cell总的输出为((batch_size, state_size), ((batch_size, state_size), (batch_size, state_size)))
    # MultiRNNCell: (cell=LSTMCell, state_is_tuple=True情况下)
    #           输入输出：cell(input, state) -> (output, new_state)
    #           input为第一层一个时间步长的input， state为每一层 c_state和m_state组成的tuple 组成的tuple
    #           input.shape==(batch_size, input_size),
    #           state.shape==(layer0((batch_size, state_size0), (batch_size, state_size0)), layer1((batch_size, state_size1), (batch_size, state_size1)), ...)
    #           output为最后一层一个时间步长的output，new_state为每一层 新的c_state和m_state组成的tuple 组成的tuple
    #           output.shape==(batch_size, state_size_of_last_layer),
    #           new_state.shape==(layer0((batch_size, state_size0), (batch_size, state_size0)), layer1((batch_size, state_size1), (batch_size, state_size1)), ...)
    #           因此cell总的输出为((batch_size, state_size_of_last_layer),
    #           (layer0((batch_size, state_size0), (batch_size, state_size0)), layer1((batch_size, state_size1), (batch_size, state_size1)), ...))

    def fn(accum, elem):
        return cell(elem, accum[1])
    rnn_outputs, final_states = tf.scan(fn=fn, elems=tf.transpose(rnn_inputs, [1, 0, 2]),
                                        initializer=(tf.zeros([batch_size, state_size]), init_state))
    # tf.scan的效果是，先用initializer作为accum的初始值，然后从elems中取elems[0]作为这一轮的elem带入fn，输出的结果作为新的accum跟
    # 下一个elem一起带入fn，如此循环直到elems[-1]。
    # 这里tf.scan中的fn的参数accum, return值，与initializer值必须为统一形状。
    # 解释：fn的return值会被用作下一轮的accum，而initializer会被当作第一个accum，因此必须统一形状。
    # tf.scan的输出会将迭代次数T加到tensor的第一个维度上，例如这里输出的tuple(rnn_outputs, final_states)：
    #       rnn_outputs: tensor(T, batch_size, state_size_of_last_layer)
    #       final_states: tuple(LSTMStateTuple_oflayer0((T, batch_size, state_size0), (T, batch_size, state_size0)),
    #                           LSTMStateTuple_oflayer1((T, batch_size, state_size1), (T, batch_size, state_size1)),
    #                           LSTMStateTuple_oflayer2((T, batch_size, state_size2), (T, batch_size, state_size2)))

    final_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
        tf.squeeze(tf.slice(c, [num_steps - 1, 0, 0], [1, batch_size, state_size])),
        tf.squeeze(tf.slice(h, [num_steps - 1, 0, 0], [1, batch_size, state_size]))) for c, h in final_states])

    with tf.variable_scope('softmax'):
        w = tf.get_variable('w', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshape = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, w) + b
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=y_reshape))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x=x,
                y=y,
                init_state=init_state,
                final_state=final_state,
                total_loss=total_loss,
                train_step=train_step)

# start_time = time.time()
# # g = build_multilayer_lstm_graph_with_scan()
# g = build_multilayer_lstm_graph_with_dynamic_rnn()
# # g = build_multilayer_lstm_graph_with_static_rnn()
# print('构建图耗时： ', time.time() - start_time)
# start_time = time.time()
# train_rnn(g, 20)
# print('训练耗时： ', time.time() - start_time)


def build_multilayer_lnlstm_graph_with_dynamic_rnn(
        cell_type=None,
        state_size=state_size,
        num_classes=num_classes,
        batch_size=batch_size,
        num_steps=num_steps,
        num_layers=num_layers,
        learning_rate=learning_rate,
        keep_probs=[1.0, 1.0]
):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    if cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    elif cell_type == 'BasicRNN':
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_probs[0])
    if num_layers != 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_probs[1])

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    with tf.variable_scope('softmax'):
        w = tf.get_variable('w', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, w) + b

    predictions = tf.nn.softmax(logits)
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        preds=predictions,
        saver=tf.train.Saver()
    )


def train_rnn(g, num_epochs, num_steps=num_steps, batch_size=batch_size, verbose=True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []

        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                feed_dict = {g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run(
                    [g['total_loss'], g['final_state'], g['train_step']],
                    feed_dict=feed_dict)
                training_loss += training_loss_

            if verbose:
                print('epoch: {0}的平均损失值： {1}'.format(idx, training_loss/steps))
            training_losses.append(training_loss/steps)
        if isinstance(save, str):
            g['saver'].save(sess, save)
    return training_losses


def generate_characters(g, save, num_chars, prompt='A', pick_top_chars=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, save)
        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]
        for i in range(num_chars):
            if state is not None:
                feed_dict = {g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict = {g['x']: [[current_char]]}
            _, preds, state= sess.run([g['init_state'], g['preds'], g['final_state']], feed_dict=feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)
        chars = map(lambda x: idx_to_vocab[x], chars)
        result = ''.join(chars)
        print(result)
        return result


# 实验0：生成文本
num_epochs = 30
num_layers = 1
cell_type = 'LN_LSTM'
# 构建图
g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type=cell_type, num_layers=num_layers,
                                                   learning_rate=learning_rate, keep_probs=[0.9, 0.9])
# 训练
start_time = time.time()
losses = train_rnn(g=g, num_epochs=num_epochs,
                   save='saves/{0}_{1}layers_{2}epochs'.format(cell_type, num_layers, num_epochs))
print("训练耗时：", time.time()-start_time)
print('last loss: ', losses[-1])
# 生成文本
g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type=cell_type, num_layers=num_layers, num_steps=1,
                                                   batch_size=1, learning_rate=learning_rate, keep_probs=[0.9, 0.9])
text = generate_characters(g=g, save='saves/{0}_{1}layers_{2}epochs'.format(cell_type, num_layers, num_epochs),
                           num_chars=750, prompt='A', pick_top_chars=1)
file_name_output = 'output_{0}_{1}layers_{2}epochs.txt'.format(cell_type, num_layers, num_epochs)
with open(file_name_output, 'w') as f:
    f.write(text)
print(text)
# # 实验结果：生成文本毫无逻辑。


#
# # 实验1：对比static, dynamic, scan()的训练速度和效果
# num_epochs = 1
# num_layers = 3
#
# g = build_multilayer_lstm_graph_with_scan(num_layers=num_layers, keep_prob=0.9, learning_rate=learning_rate)
# start_time = time.time()
# losses_scan = train_rnn(g=g, num_epochs=num_epochs)
# duration_scan = time.time() - start_time
# print("scan训练耗时：", duration_scan)
#
# g = build_multilayer_lstm_graph_with_dynamic_rnn(num_layers=num_layers, keep_prob=0.9, learning_rate=learning_rate)
# start_time = time.time()
# losses_dynamic = train_rnn(g=g, num_epochs=num_epochs)
# duration_dynamic = time.time() - start_time
# print("dynamic训练耗时：", duration_dynamic)
#
# g = build_multilayer_lstm_graph_with_static_rnn(num_layers=num_layers, keep_prob=0.9, learning_rate=learning_rate,
#                                                 batch_size=batch_size, state_size=100)
# start_time = time.time()
# losses_static = train_rnn(g=g, num_epochs=num_epochs, batch_size=batch_size)
# duration_static = time.time() - start_time
# print("static训练耗时：", duration_static)
#
# plt.title('loss curves of static, dynamic, scan RNN')
# plt.plot(losses_scan, color='green', label='losses_scan({}s)'.format(duration_scan))
# plt.plot(losses_dynamic, color='red', label='losses_dynamic({}s)'.format(duration_dynamic))
# plt.plot(losses_static, color='blue', label='losses_static({}s)'.format(duration_static))
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('构建计算图方式对比_{0}层{1}epochs.png'.format(num_layers, num_epochs))
# plt.show()
# # # 实验结果：lstm_with_scan在1个epoch之后loss几乎不再下降，lstm_with_static_rnn在state_size==512,num_layers==3时内存溢出，
# # #         改为state_size==100后正常训练。也就是能够正常使用的只有lstm_with_dynamic_rnn。
#
#
# # 实验2：分析dropout, layer normalization的训练速度和效果
# num_epochs=1
# num_layers=3
# keep_probs = [0.9, 0.9]
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(num_layers=num_layers, learning_rate=learning_rate)
# start_time = time.time()
# losses_basic = train_rnn(g=g, num_epochs=num_epochs)
# duration_basic = time.time()-start_time
# print("BasicRNN训练耗时：", duration_basic)
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type='LSTM', num_layers=num_layers,
#                                                    learning_rate=learning_rate)
# start_time = time.time()
# losses_lstm = train_rnn(g=g, num_epochs=num_epochs)
# duration_lstm = time.time()-start_time
# print("LSTM训练耗时：", duration_lstm)
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type='LN_LSTM', num_layers=num_layers,
#                                                    learning_rate=learning_rate)
# start_time = time.time()
# losses_lnlstm = train_rnn(g=g, num_epochs=num_epochs)
# duration_lnlstm = time.time()-start_time
# print("LN_LSTM训练耗时：", duration_lnlstm)
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(num_layers=num_layers,
#                                                    learning_rate=learning_rate, keep_probs=keep_probs)
# start_time = time.time()
# losses_dropoutlstm = train_rnn(g=g, num_epochs=num_epochs)
# duration_dropoutlstm = time.time()-start_time
# print("Dropout_LSTM训练耗时：", duration_dropoutlstm)
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type='LN_LSTM', num_layers=num_layers,
#                                                    learning_rate=learning_rate, keep_probs=keep_probs)
# start_time = time.time()
# losses_dropoutlnlstm = train_rnn(g=g, num_epochs=num_epochs)
# duration_dropoutlnlstm = time.time()-start_time
# print("Dropout_LN_LSTM训练耗时：", duration_dropoutlnlstm)
#
#
# plt.title('loss curves of basicRNN, LSTM, LN_LSTM, Dropout_LSTM, Dropout_LN_LSTM')
# plt.plot(losses_basic, color='green', label='losses_basic({}s)'.format(duration_basic))
# plt.plot(losses_lstm, color='red', label='losses_lstm({}s)'.format(duration_lstm))
# plt.plot(losses_lnlstm, color='blue', label='losses_lnlstm({}s)'.format(duration_lnlstm))
# plt.plot(losses_dropoutlstm, color='yellow', label='losses_dropoutlstm({}s)'.format(duration_dropoutlstm))
# plt.plot(losses_dropoutlnlstm, color='cyan', label='losses_dropoutlnlstm({}s)'.format(duration_dropoutlnlstm))
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('网络结构对比_{0}层{1}epochs.png'.format(num_layers, num_epochs))
# plt.show()



