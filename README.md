# RNN_tensorflow_implement
simple RNN implementation using tensorflow.
由 https://github.com/lawlite19/Blog-Back-Up/tree/master/code/rnn 修改而来，添加了一些注释，对比实验。
based on tensorflow-1.6.0.

rnn1.py详细说明了数据处理过程，手动实现rnn，tf.nn.static_rnn，tf.nn.dynamic_rnn的用法，cell为BasicRNNCell。
rnn2.py在前面的基础上对比了tf.nn.static_rnn，tf.nn.dynamic_rnn和tf.scan三种RNN实现方法的性能，分析了Dropout层、Layer Normalization对网络
性能的影响，利用LN_LSTM网络生成了一段文本序列。
