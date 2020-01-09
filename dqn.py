import gym
import tensorflow as tf
from tensorflow.keras import layers,optimizers
import numpy as np
from  collections import namedtuple,deque
import random
SARS=namedtuple('SARS',['state','action','reward','next_state','done_mask'])

class QNet(tf.keras.Model):
    def __init__(self,action_dims):
        super(QNet, self).__init__()
        self.action_dims=action_dims
        self.fc1=layers.Dense(256,activation='relu')
        self.fc2=layers.Dense(128,activation='relu')
        self.fc3=layers.Dense(action_dims)

    def call(self, inputs, training=None, mask=None):
        x=self.fc1(inputs)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

class DQN(object):
    def __init__(self,qnet,target_net,optimizer,loss,gamma):
        super(DQN, self).__init__()
        self.qnet=qnet
        self.target_net=target_net
        self.optimizer=optimizer
        self.loss=loss
        self.buffer=deque(maxlen=2000)
        self.gamma=gamma #折扣率

    def buffer_len(self):
        return len(self.buffer)

    def put_data(self,sars):
        self.buffer.append(sars)

    def sample_action(self,s,epsilon):
        s=tf.constant(s,dtype=tf.float32)
        s=tf.expand_dims(s,axis=0)
        q_a=self.qnet(s)
        a=tf.argmax(q_a,axis=1)[0]
        other_action_prob=np.random.rand()
        if other_action_prob<epsilon:
            return np.random.choice(range(0,self.qnet.action_dims))
        else:
            return int(a)

    def sample_data(self,batch_size):
        batch_data=random.sample(self.buffer,batch_size)
        state=tf.constant([d.state for d in batch_data],dtype=tf.float32)
        action=tf.constant([d.action for d in batch_data],dtype=tf.int32)
        reward = tf.constant([d.reward for d in batch_data], dtype=tf.float32)
        next_state = tf.constant([d.next_state for d in batch_data], dtype=tf.float32)
        done_mask=tf.constant([d.done_mask for d in batch_data], dtype=tf.float32)
        return state,action,reward,next_state,done_mask

    def copy_weights_to_tagetnet(self):
        for src,dst in zip(self.qnet.trainable_variables,self.target_net.trainable_variables):
            dst.assign(src)

    def train(self,batch_size):
        for step in range(10):
            state, action, reward, next_state,done_mask=self.sample_data(batch_size)
            with tf.GradientTape() as tape:
                q=self.qnet(state)
                indices=tf.range(0,q.shape[0])
                indices=tf.stack([indices,action],axis=1)
                q=tf.gather_nd(q,indices)
                with tape.stop_recording():
                    target_q=self.target_net(next_state)
                    target_q=tf.reduce_max(target_q,axis=-1)
                    target_q=reward+self.gamma*target_q*done_mask
                loss=self.loss(target_q,q)
                loss=tf.reduce_mean(loss)

            grads=tape.gradient(loss,self.qnet.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.qnet.trainable_variables))

if __name__ == '__main__':
    EPOCHS=1000
    BATCH_SIZE=32
    GAMMA=0.98
    LEARNING_RATE=0.0002
    EPSILON=0.2

    qnet=QNet(2)
    target_net=QNet(2)

    qnet.build(input_shape=(None,4))
    target_net.build(input_shape=(None, 4))


    agent=DQN(qnet=qnet,
              target_net=target_net,
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              loss=tf.losses.Huber(),
              gamma=GAMMA)
    agent.copy_weights_to_tagetnet()
    env=gym.make('CartPole-v1')
    returns=[]
    smooth_results=[]
    scores=0
    for epoch in range(EPOCHS):
        EPSILON=max(0.01, 0.08 - 0.01 * (epoch / 200))
        state=env.reset()
        for t in range(600):
            action=agent.sample_action(state,EPSILON)
            next_state,reward,done,_=env.step(action)
            done_mask=0. if done else 1.
            scores+=reward
            agent.put_data(SARS(state,action,reward/100,next_state,done_mask))
            state=next_state

            if done:
                break
        returns.append(scores)
        if not smooth_results:
            smooth_results.append(scores)
        else:
            smooth_results.append(0.95*smooth_results[-1]+0.05*scores)

        if agent.buffer_len()>200:
            agent.train(BATCH_SIZE)

        if epoch%10==0 and epoch!=0:
            agent.copy_weights_to_tagetnet()
            print('>%d/%d: score=%f' % (epoch+1,EPOCHS,scores))
        scores = 0.

    import matplotlib.pyplot as plt

    plt.plot(returns,'g',alpha=0.8)
    plt.plot(smooth_results)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(['origin score','smooth score'])
    plt.show()

