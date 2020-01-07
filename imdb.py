# imdb文本数据集情感分类

import  tensorflow as tf
from tensorflow.keras import datasets,layers,preprocessing,models,losses,optimizers
import numpy as np
num_words=10000
MAX_NUM_WORDS = 20000
batch_size=128
seq_len=80
(x_train, y_train), (x_test, y_test)=datasets.imdb.load_data(num_words=num_words)

word_idx=datasets.imdb.get_word_index()
word_idx=dict((w,word_idx[w]+3) for w in word_idx)
word_idx['<PAD>']=0
word_idx['<START>']=1
word_idx['<UNK>']=2
word_idx['<UNUSED>']=3


#
# idx_word=dict((word_idx[w],w) for w in word_idx)
#
# def decode_review(text):
#     return ' '.join([idx_word.get(i,'?') for i in text])

GLOVE_DIR='../glove.6B/glove.6B.100d.txt'
embedding_index={}
with open(GLOVE_DIR,encoding='utf-8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vector=np.asarray(values[1:],dtype=np.float32)
        embedding_index[word]=vector
print('Found %d words vector' % len(embedding_index))

num_words=min(MAX_NUM_WORDS, len(word_idx)) + 1
embedding_matrix=np.zeros(shape=(num_words,100))
for w,i in word_idx.items():
    if i>MAX_NUM_WORDS:
        continue
    vector=embedding_index.get(word)
    if vector is not None:
        embedding_matrix[i]=vector

x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=seq_len)
x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=seq_len)

db_train=tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train=db_train.shuffle(1000).batch(batch_size,drop_remainder=True)

db_test=tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test=db_test.batch(batch_size,drop_remainder=True)

class MyRNN(tf.keras.Model):
    def __init__(self,units):
        super(MyRNN, self).__init__()
        self.state0=[tf.zeros(shape=(batch_size,units))]
        self.state1=[tf.zeros(shape=(batch_size,units))]
        self.embedding=layers.Embedding(num_words,100,input_length=seq_len)
        self.cell0=layers.SimpleRNNCell(units,dropout=0.2)
        self.cell1=layers.SimpleRNNCell(units,dropout=0.2)
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        x=self.embedding(inputs)
        state0=self.state0
        state1=self.state1
        for word in tf.unstack(x,axis=1):
            out0,state0=self.cell0(word,state0,training)
            out1,state1=self.cell1(out0,state1,training)
        out=self.outlayer(out1)
        prob=tf.sigmoid(out)
        return prob

class MyLSTM(tf.keras.Model):
    def __init__(self,units):
        super(MyLSTM, self).__init__()
        self.state0=[tf.zeros(shape=(batch_size,units)),tf.zeros(shape=(batch_size,units))]
        self.state1=[tf.zeros(shape=(batch_size,units)),tf.zeros(shape=(batch_size,units))]
        self.embedding=layers.Embedding(num_words,100,input_length=seq_len)
        self.cell0=layers.LSTMCell(units,dropout=0.5)
        self.cell1=layers.LSTMCell(units,dropout=0.5)
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        x=self.embedding(inputs)
        state0=self.state0
        state1=self.state1
        for word in tf.unstack(x,axis=1):
            out0,state0=self.cell0(word,state0,training)
            out1,state1=self.cell1(out0,state1,training)
        out=self.outlayer(out1)
        prob=tf.sigmoid(out)
        return prob

class MyGRU(tf.keras.Model):
    def __init__(self,units):
        super(MyGRU, self).__init__()
        self.state0=[tf.zeros(shape=(batch_size,units))]
        self.state1=[tf.zeros(shape=(batch_size,units))]
        self.embedding=layers.Embedding(num_words,100,input_length=seq_len)
        self.cell0=layers.GRUCell(units,dropout=0.5)
        self.cell1=layers.GRUCell(units,dropout=0.5)
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        x=self.embedding(inputs)
        state0=self.state0
        state1=self.state1
        for word in tf.unstack(x,axis=1):
            out0,state0=self.cell0(word,state0,training)
            out1,state1=self.cell1(out0,state1,training)
        out=self.outlayer(out1)
        prob=tf.sigmoid(out)
        return prob

def main():
    units=64
    epochs=40
    model=MyLSTM(units)
    model.build(input_shape=(None,seq_len))
    model.embedding.set_weights([embedding_matrix])
   # x=tf.random.uniform(shape=(batch_size,80),minval=0,maxval=80,dtype=tf.int32)
   # model(x,training=False)

    loss = losses.BinaryCrossentropy()
    optimizer = optimizers.RMSprop(0.001)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  run_eagerly=True)  # 指定为eager 模式
    model.fit(db_train,epochs=epochs,validation_data=db_test)
    model.evaluate(db_test)

if __name__ == '__main__':
    main()

# none pretrained :loss: 0.4302 - accuracy: 0.8269
# pretrained: loss: 0.3903 - accuracy: 0.8377