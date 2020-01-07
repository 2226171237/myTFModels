# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers


rnn=layers.SimpleRNN(64)
rnn.build(input_shape=(None,None,100))
x=tf.random.normal(shape=(4,80,100))
y=rnn(x)
print(y.shape)
