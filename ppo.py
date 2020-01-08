import tensorflow as tf
from tensorflow.keras import layers,optimizers
from collections import namedtuple
import numpy as np

class Actor(tf.keras.Model):
    def __init__(self,action_dims):
        super(Actor, self).__init__()
        self.fc1=layers.Dense(100,activation='relu',kernel_initializer='he_normal')
        self.fc2=layers.Dense(action_dims,kernel_initializer='he_normal')

    def call(self, inputs, training=None, mask=None):
        x=self.fc1(inputs)
        x=self.fc2(x)
        x=tf.nn.softmax(x)
        return x

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1=layers.Dense(100,activation='relu',kernel_initializer='he_normal')
        self.fc2=layers.Dense(1,kernel_initializer='he_normal')

    def call(self, inputs, training=None, mask=None):
        x=self.fc1(inputs)
        x=self.fc2(x)
        return x

SAPRS=namedtuple('SAPRS',['state','action','action_prob','reward','next_state'])

class PPO2:
    def __init__(self,actor,critic,optimizers,gamma=1.0,epsilon=0.2):
        self.buffer=[]
        self.actor=actor
        self.critic=critic
        self.actor_optimizer=optimizers[0]
        self.critic_optimizer=optimizers[1]
        self.gamma=gamma #折扣率
        self.epsilon=epsilon #PPO loss超参数

    def select_action(self,s):
        s=tf.constant(s,dtype=tf.float32)
        s=tf.expand_dims(s,axis=0)
        actions_prob=self.actor(s)
        a=tf.random.categorical(tf.math.log(actions_prob),1)[0]
        a=int(a)
        return a,float(actions_prob[0][a])

    def get_value(self,s):
        s=tf.constant(s,dtype=tf.float32)
        s=tf.expand_dims(s,axis=0)
        v=self.critic(s)
        return float(v[0])

    def put_data(self,saprs):
        self.buffer.append(saprs)

    def optimizer(self,batch_size):
        # 一次游戏的结果
        states=tf.constant([d.state for d in self.buffer],dtype=tf.float32)

        actions=tf.constant([d.action for d in self.buffer],dtype=tf.int32)
        actions=tf.reshape(actions,(-1,1))

        action_probs=tf.constant([d.action_prob for d in self.buffer],dtype=tf.float32)
        action_probs=tf.reshape(action_probs,(-1,1))

        rewards=[d.reward for d in self.buffer]


        R=0
        Rs=[0 for _ in range(len(self.buffer))]
        for i,r in enumerate(rewards[::-1]):
            R=self.gamma*R+r
            Rs[len(Rs)-i-1]=R
        Rs=tf.constant(Rs,dtype=tf.float32)

        for step in range(10*len(self.buffer)//batch_size):
            inds=np.random.choice(range(len(self.buffer)),size=batch_size,replace=False)  # replace表示无放回采用。
            with tf.GradientTape() as tape1,tf.GradientTape() as tape2:
                v_target=tf.expand_dims(tf.gather(Rs,inds,axis=0),axis=1)
                v=self.critic(tf.gather(states,inds,axis=0))
                delta=v_target-v
                advantage=tf.stop_gradient(delta)

                a_target=tf.gather(actions,inds,axis=0)
                prob=self.actor(tf.gather(states,inds,axis=0))
                indices=tf.reshape(tf.range(0,batch_size,dtype=tf.int32),shape=(-1,1))
                indices=tf.concat([indices,a_target],axis=1)
                prob_a=tf.gather_nd(prob,indices)
                prob_a=tf.reshape(prob_a,shape=(-1,1))

                ratios=prob_a/tf.gather(action_probs,inds,axis=0)
                actor_loss_clip=tf.minimum(ratios*advantage,
                                           tf.clip_by_value(ratios,1-self.epsilon,1+self.epsilon)*advantage)
                actor_loss_clip=-tf.reduce_mean(actor_loss_clip)

                critic_loss=tf.reduce_mean(tf.losses.mse(v_target,v))
            actor_grads=tape1.gradient(actor_loss_clip,self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads,self.actor.trainable_variables))

            critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.buffer=[]

if __name__ == '__main__':
    import gym

    EPOCHS=500
    BATCH_SIZE=32

    agent=PPO2(actor=Actor(action_dims=2),
               critic=Critic(),
               optimizers=[optimizers.Adam(1e-3),optimizers.Adam(3e-3)],
               gamma=0.98)
    env=gym.make('CartPole-v1')

    returns=[]
    smooth_returns=[]
    scores=0.
    for epoch in range(EPOCHS):
        state=env.reset()
        for _ in range(500):
            action,action_prob=agent.select_action(state)
            next_state,reward,done,_=env.step(action)
            agent.put_data(SAPRS(state,action,action_prob,reward,next_state))
            state=next_state
            scores+=reward
            if done:
                break
        if not smooth_returns:
            smooth_returns.append(scores)
        else:
            smooth_returns.append(0.95*smooth_returns[-1]+0.05*scores)
        returns.append(scores)
        scores=0.
        if len(agent.buffer)>=BATCH_SIZE:
            agent.optimizer(BATCH_SIZE)
        if epoch%20==0:
            print('epoch=%d,scores=%f' % (epoch+1,returns[-1]))

    import matplotlib.pyplot as plt
    plt.plot(returns,'g',alpha=0.7)
    plt.plot(smooth_returns,'r')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(['origin scores','smooth scores'])
    plt.show()