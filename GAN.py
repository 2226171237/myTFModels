import tensorflow as tf

from tensorflow.keras import layers,optimizers
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time


class Generator(tf.keras.Model):
    def __init__(self,filters=64):
        super(Generator, self).__init__()
        co_params={'kernel_size':(4, 4),'strides':2,'padding':'same','use_bias':False}
        self.upsample1=layers.Conv2DTranspose(filters=filters*8,kernel_size=(4,4),padding='valid',use_bias=False)  # 1*1 -> 4*4
        self.bn1=layers.BatchNormalization()

        self.upsample2 = layers.Conv2DTranspose(filters=filters*4, **co_params) # 4*4 -> 8*8
        self.bn2 = layers.BatchNormalization()

        self.upsample3 = layers.Conv2DTranspose(filters=filters*2, **co_params) # 8*8 -> 16*16
        self.bn3 = layers.BatchNormalization()

        self.upsample4 = layers.Conv2DTranspose(filters=filters, **co_params) # 16*16 -> 32*32
        self.bn4 = layers.BatchNormalization()

        self.upsample5 = layers.Conv2DTranspose(filters=3,**co_params) # 32*32 -> 64*64

    def call(self, inputs, training=None, mask=None):
        x=tf.reshape(inputs,[-1,1,1,inputs.shape[-1]])
        x=self.upsample1(x)
        x=tf.nn.relu(self.bn1(x,training))
        x=self.upsample2(x)
        x=tf.nn.relu(self.bn2(x,training))
        x=self.upsample3(x)
        x=tf.nn.relu(self.bn3(x,training))
        x = self.upsample4(x)
        x = tf.nn.relu(self.bn4(x,training))
        x = self.upsample5(x)
        x = tf.nn.tanh(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self,filters=64):
        super(Discriminator, self).__init__()

        self.conv1=layers.Conv2D(filters=filters,kernel_size=4,strides=2,padding='valid',use_bias=False)  #64,64->31,31
        self.bn1=layers.BatchNormalization()

        self.conv2=layers.Conv2D(filters=filters*2,kernel_size=4,strides=2,padding='valid',use_bias=False) # 31,31->14,14
        self.bn2=layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filters=filters * 4, kernel_size=4, strides=2, padding='valid', use_bias=False) # 14,14->6,6
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='valid', use_bias=False) # 6,6->4,4
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(filters=filters * 16, kernel_size=3, strides=1, padding='valid', use_bias=False) # 4,4 -> 2,2
        self.bn5 = layers.BatchNormalization()

        self.pool=layers.GlobalAveragePooling2D()  # 1024
        self.flatten=layers.Flatten()
        self.fc=layers.Dense(1)

    def call(self, inputs, training=None, mask=None):

        x=self.bn1(self.conv1(inputs),training)
        x=tf.nn.leaky_relu(x)

        x = self.bn2(self.conv2(x), training)
        x = tf.nn.leaky_relu(x)

        x = self.bn3(self.conv3(x), training)
        x = tf.nn.leaky_relu(x)

        x = self.bn4(self.conv4(x), training)
        x = tf.nn.leaky_relu(x)

        x = self.bn5(self.conv5(x), training)
        x = tf.nn.leaky_relu(x)

        x=self.pool(x)
        x=self.flatten(x)
        x=self.fc(x)
        return x


def save_imgs(imgs,name):
    b,h,w,_=imgs.shape
    big_img=Image.new(mode='RGB',size=(h*8,w*8))
    idx=0
    for i in range(0,h*8,h):
        for j in range(0,w*8,w):
            img=imgs[idx]
            img=Image.fromarray(img)
            big_img.paste(img,(j,i))
            idx+=1
            if idx>=b or idx>=64:
                break
        if idx>=b or idx>=64:
            break
    if not os.path.exists(os.path.dirname(name)):
        os.mkdir(os.path.dirname(name))
    big_img.save(name)

class DCGAN:
    def __init__(self,generator,discriminator,dim_z):
        self.dim_z=dim_z
        self.generator=generator
        self.discriminator=discriminator

    def build(self,input_shape):
        self.generator.build(input_shape=(None,self.dim_z))
        self.discriminator.build(input_shape=input_shape)

    def compile(self,g_optimizer,d_optimizer,loss=None):
        self.g_optimizer=g_optimizer
        self.d_optimizer=d_optimizer
        self.loss=loss

    def fit(self,db_train,epochs,steps,train_g_steps=1,train_d_steps=5,save_weights_path=None,save_imgs_path=None,gen_imgs_nums=64):
        train_d_step = 0
        train_g_step = 0
        d_loss,g_loss=0.,0.
        start_time=time.time()
        for epoch in range(epochs):
            for step in range(steps):
                imgs=next(db_train)
                if train_d_step<train_d_steps:
                    p_z=tf.random.normal([imgs.shape[0],self.dim_z])
                    with tf.GradientTape() as tape:
                        d_loss=self.get_d_loss(imgs,p_z,True)
                    grads=tape.gradient(d_loss,self.discriminator.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(grads,self.discriminator.trainable_variables))
                    train_d_step+=1
                else:
                    if train_g_step<train_g_steps:
                        p_z = tf.random.normal([imgs.shape[0], self.dim_z])
                        with tf.GradientTape() as tape:
                            g_loss = self.get_g_loss(p_z,True)
                        grads = tape.gradient(g_loss, self.generator.trainable_variables)
                        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
                        train_g_step += 1
                    else:
                        train_g_step=0
                        train_d_step=0
                if (step+1)%100==0:
                    print('epoch=%d, step=%d, d_loss=%f, g_loss=%f, total_loss=%f, times=%fs' %
                          (epoch+1,step+1,d_loss.numpy(),g_loss.numpy(),
                           d_loss.numpy()+g_loss.numpy(),time.time()-start_time))
                    start_time=time.time()
            if save_imgs_path:
                self.generator_imgs(gen_imgs_nums,save_imgs_path,epoch)

            if (epoch+1)%100==0 and save_weights_path is not None:
                self.generator.save_weights(os.path.join(save_weights_path,'generator.h5'))
                self.discriminator.save_weights(os.path.join(save_weights_path,'discriminator.h5'))
                print('save model weight to ',save_weights_path)

        print('training over!!')

    def get_g_loss(self,batch_z,training):
        fake_image=self.generator(batch_z,training)
        d=self.discriminator(fake_image,training)
        d_true=tf.ones_like(d)
        loss=tf.reduce_mean(self.loss(d_true,d))
        return loss


    def get_d_loss(self,batch_x,batch_z,training):
        fake_image = self.generator(batch_z,training)
        fake_d=self.discriminator(fake_image,training)
        real_d=self.discriminator(batch_x,training)
        fake_label=tf.zeros_like(fake_d)
        real_label=tf.ones_like(real_d)
        loss=tf.reduce_mean(self.loss(real_label,real_d))+\
             tf.reduce_mean(self.loss(fake_label,fake_d))
        return loss

    def generator_imgs(self,img_nums,save_path,epoch):
        p_z=tf.random.normal((img_nums,self.dim_z))
        imgs=self.generator(p_z,False).numpy()
        imgs=(imgs+1.)*127.5
        imgs=imgs.astype(np.uint8)
        save_imgs(imgs,os.path.join(save_path,'epoch_%d.png' % epoch))


class WGAN_GP(DCGAN):
    def __init__(self,gamma=1.0,*args,**kwargs):
        super(WGAN_GP, self).__init__(*args,**kwargs)
        self.gamma=gamma # pg 权重

    def gradient_penalty(self,real_imgs,fake_imgs):
        t=tf.random.uniform(shape=(fake_imgs.shape[0],1,1,1))
        interplate=t*real_imgs+(1-t)*fake_imgs
        with tf.GradientTape() as tape:
            tape.watch([interplate])
            d_logits=self.discriminator(interplate,True)
        grads=tape.gradient(d_logits,interplate)
        grads=tf.reshape(grads,shape=(grads.shape[0],-1))
        gp=tf.norm(grads,axis=1)
        gp=tf.reduce_mean((gp-1.0)**2)
        return gp

    def get_g_loss(self,batch_z,training):
        fake_image=self.generator(batch_z,training)
        d_logits=self.discriminator(fake_image,training)
        loss=-tf.reduce_mean(d_logits)
        return loss

    def get_d_loss(self,batch_x,batch_z,training):
        fake_image = self.generator(batch_z,training)
        fake_d_logits=self.discriminator(fake_image,training)
        real_d_logits=self.discriminator(batch_x,training)
        gp=self.gradient_penalty(batch_x,fake_image)
        loss=tf.reduce_mean(fake_d_logits)-tf.reduce_mean(real_d_logits)+self.gamma*gp
        return loss

if __name__ == '__main__':

    FACE_DIR = 'E:\\myWork\\dataset\\faces'
    BATCH_SIZE = 32
    EPOCHS=200

    def rescale(img):
        return img / 127.5 - 1.0

    faces_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=rescale)
    faces_train_gen = faces_train_generator.flow_from_directory(directory=FACE_DIR,
                                                                target_size=(64, 64),
                                                                batch_size=BATCH_SIZE,
                                                                shuffle=True, seed=10,
                                                                class_mode=None)

    g=Generator(32)
    d=Discriminator(32)
    #GAN=DCGAN(g,d,dim_z=100)
    GAN = WGAN_GP(1.0,g, d, dim_z=100)

    g_optimzer=optimizers.RMSprop(lr=0.0005)
    d_optimzer=optimizers.RMSprop(lr=0.0005)
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)

    GAN.compile(g_optimzer,d_optimzer,loss)
    GAN.build(input_shape=(None,64,64,3))

    steps=5122//BATCH_SIZE
    GAN.fit(faces_train_gen,steps=steps,epochs=EPOCHS,save_weights_path='./save_model/',save_imgs_path='./imgs/GAN/')




