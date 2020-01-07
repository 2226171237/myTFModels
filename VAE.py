import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Sequential,Model,optimizers
from PIL import Image
import os
BATCH_SIZE=128
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


class VAE(Model):
    def __init__(self,dim_z):
        super(VAE, self).__init__()
        self.fc1=layers.Dense(128,activation='relu')
        self.fc_mu=layers.Dense(dim_z)
        self.fc_log_var=layers.Dense(dim_z)

        self.fc4=layers.Dense(128,activation='relu')
        self.fc5=layers.Dense(784)

    def encoder(self,inputs):
        x = self.fc1(inputs)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        esp = tf.random.normal(log_var.shape)
        z = mu + esp * tf.exp(log_var) ** 0.5
        return z,mu,log_var

    def decoder(self,inputs):
        outputs = self.fc4(inputs)
        outputs = self.fc5(outputs)
        return outputs

    def call(self, inputs, training=None, mask=None):
        z,mu,log_var=self.encoder(inputs)
        outputs=self.decoder(z)
        return outputs,mu,log_var


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

model=VAE(10)
model.build(input_shape=(1,784))
model.summary()

optimizer=optimizers.Adam(lr=0.001)

for epoch in range(EPOCHS):
    for step ,x in enumerate(db_train):
        with tf.GradientTape() as tape:
            x=tf.reshape(x,(-1,784))
            x_,mu,log_var=model(x)
            rec_loss=tf.nn.sigmoid_cross_entropy_with_logits(x,x_)
            rec_loss=tf.reduce_mean(rec_loss)
            kl_div=-0.5*(log_var+1-mu**2-tf.exp(log_var))
            kl_div=tf.reduce_mean(kl_div)
            loss=5*rec_loss+1.0*kl_div
        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        if (step+1)%100==0:
            print('epoch=%d, step=%d, total_loss=%f,rec_loss=%f,kl_loss=%f' % (epoch+1,step+1,loss.numpy(),rec_loss.numpy(),kl_div.numpy()))
    indices=[]
    [indices.extend([i,i+5]) for i in range(5)]
    print(indices)
    for step,x in enumerate(db_test):
        imgs,_,_=model(tf.reshape(x,(-1,784)))
        imgs=tf.sigmoid(imgs)
        imgs=tf.reshape(imgs,(-1,28,28))
        concat_imgs=tf.reshape(tf.concat([x,imgs],axis=0),(10,10,28,28))
        concat_imgs=tf.gather(concat_imgs,indices=indices,axis=0)
        concat_imgs=tf.reshape(concat_imgs,(-1,28,28))*255.
        concat_imgs=concat_imgs.numpy().astype(np.uint8)
        save_imgs(concat_imgs,name='./imgs/epoch_%d/%d.png' % (epoch,step))

    z=tf.random.normal(shape=(100,10))
    logits=model.decoder(z)
    imgs=tf.sigmoid(logits)
    imgs=tf.reshape(imgs,(-1,28,28))*255.0
    imgs=imgs.numpy().astype(np.uint8)
    save_imgs(imgs,name='./imgs/generator/generator_%d.png' % epoch)




