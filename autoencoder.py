import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Sequential,Model,optimizers
from PIL import Image
import os
BATCH_SIZE=64
TEST_BATCH_SIZE=50
EPOCHS=50
(x_train,_),(x_test,_)=tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape)
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
db_train=tf.data.Dataset.from_tensor_slices(x_train)
db_train=db_train.shuffle(1000).batch(BATCH_SIZE)

db_test=tf.data.Dataset.from_tensor_slices(x_test)
db_test=db_test.batch(TEST_BATCH_SIZE,drop_remainder=True)


class AE(Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder=Sequential(
            [
                layers.Dense(256,activation='relu'),
                layers.Dense(128,activation='relu'),
                layers.Dense(20)
            ]
        )
        self.decoder=Sequential(
            [
                layers.Dense(128,activation='relu'),
                layers.Dense(256,activation='relu'),
                layers.Dense(784)
            ]
        )

    def call(self, inputs, training=None, mask=None):
        x=self.encoder(inputs,training)
        x=self.decoder(x,training)
        return tf.sigmoid(x)


def save_imgs(imgs,name):
    big_img=Image.new(mode='L',size=(280,280))
    idx=0
    for i in range(0,280,28):
        for j in range(0,280,28):
            img=imgs[idx]
            img=Image.fromarray(img,mode='L')
            big_img.paste(img,(j,i))
            idx+=1
    if not os.path.exists(os.path.dirname(name)):
        os.mkdir(os.path.dirname(name))
    big_img.save(name)

model=AE()
model.build(input_shape=(None,784))
model.summary()

optimizer=optimizers.Adam(lr=0.01)

for epoch in range(EPOCHS):
    for step ,x in enumerate(db_train):
        with tf.GradientTape() as tape:
            x=tf.reshape(x,(-1,784))
            x_=model(x)
            loss=tf.losses.mse(x,x_)
            loss=tf.reduce_mean(loss)
        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        if (step+1)%100==0:
            print('epoch=%d, step=%d, loss=%f' % (epoch+1,step+1,loss.numpy()))
    indices=[]
    [indices.extend([i,i+5]) for i in range(5)]
    print(indices)
    for step,x in enumerate(db_test):
        imgs=model(tf.reshape(x,(-1,784)))
        imgs=tf.reshape(imgs,(-1,28,28))
        concat_imgs=tf.reshape(tf.concat([x,imgs],axis=0),(10,10,28,28))
        concat_imgs=tf.gather(concat_imgs,indices=indices,axis=0)
        concat_imgs=tf.reshape(concat_imgs,(-1,28,28))*255.
        concat_imgs=concat_imgs.numpy().astype(np.uint8)
        save_imgs(concat_imgs,name='./imgs/epoch_%d/%d.png' % (epoch,step))







