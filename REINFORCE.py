import gym

import tensorflow as tf
from tensorflow.keras import layers

# envids = [spec.id for spec in gym.envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)

class Policy(tf.keras.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.data=[]
        self.fc1=layers.Dense(128,activation='relu')
        self.fc2=layers.Dense(2)

    def call(self, inputs, training=None, mask=None):
        x=self.fc1(inputs)
        x=self.fc2(x)
        x=tf.nn.softmax(x)
        return x

    def put_data(self,item):
        # item=(reward,logp(a|s))
        self.data.append(item)

    def train(self,tape,gamma):
        R=0
        loss=0.
        for r, log_prob in self.data[::-1]:
            R = gamma * R + r
            loss = -log_prob * R
            with tape.stop_recording():
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.data = []
        #self.b=0.9*self.b+0.1*R
        return loss

env=gym.make('CartPole-v1')

Proc=Policy()
optimizer=tf.keras.optimizers.Adam(lr=0.0002)
gamma=0.98 # 回报折扣率
beta=0.9 #平滑系数

returns=[]
losses=[]
for n_epi in range(800):
    s=env.reset()
    score=0
    # persistent=False 则调用一次tape.gradient方法 tape资源就会被释放(这个是默认情况)
    # 要在同一计算上计算多个梯度，请创建一个持久梯度 tape。即persistent=True
    with tf.GradientTape(persistent=True) as tape:
        for t in range(500):
            s=tf.constant(s,dtype=tf.float32)
            s=tf.expand_dims(s,axis=0)
            prob_a=Proc(s)
            a=tf.random.categorical(tf.math.log(prob_a),1)[0]
            a=int(a)
            s,reward,done,_=env.step(a)
            Proc.put_data((reward,tf.math.log(prob_a[0][a])))
            score+=reward
            if done:
                break
        loss=Proc.train(tape,gamma)
        del tape
    if n_epi%100==0:
        print('%d: loss=%f,score=%f' % (n_epi+1,loss.numpy(),score))
    if len(returns)==0:
        returns.append(score)
    else:
        returns.append((beta*returns[-1]+(1-beta)*score))
    losses.append(loss.numpy())

import matplotlib.pyplot as plt

plt.plot(returns)
plt.show()

