"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.
Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/
Environment
-----------
Openai Gym Pendulum-v0, continual action space
Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
To run
------
python tutorial_DDPG.py --train/test
"""

import argparse
import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

#  hyper parameters  #

ENV_NAME = 'myEnv-v0'  # environment name
RANDOMSEED = 1  # random seed

LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.85  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY = 100000  # size of replay buffer
BATCH_SIZE = 64  # update batchsize

MAX_EPISODES = 1000  # total number of episodes for training
MAX_EP_STEPS = 200  # total number of steps for each episode
TEST_PER_EPISODES = 10  # test the model per episodes
sigma = 0.2  # control exploration

project_dir = os.path.dirname(os.getcwd())


#  DDPG  #
class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, a_dim, s_dim, a_bound):
        # memory用于储存跑的数据的数组：
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=10, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)  # 注意这里，先用tanh把范围限定在[-1,1]之间，再进行映射
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        # 建立Critic网络，输入s，a。输出Q值
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat()([s, a])
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([5, 10])
        self.critic = get_critic([5, 10], [5, 10])
        self.actor.train()
        self.critic.train()

        # 更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        # 建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([5, 10], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        # 建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([5, 10], [5, 10], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        # 建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights  # 获取要更新的参数包括actor和critic的
        self.ema.apply(paras)  # 主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))  # 用滑动平均赋值

    # 选择动作，把s带进入，输出a
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        ret = self.actor(np.array([s], dtype=np.float32))[0]
        return ret

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)  # 随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]  # 根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]  # 从bt获得数据s
        bs = np.reshape(bs, (BATCH_SIZE, 5, 10))
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  # 从bt获得数据a
        ba = np.reshape(ba, (BATCH_SIZE, 5, 10))
        br = bt[:, -self.s_dim - 1:-self.s_dim]  # 从bt获得数据r
        br = np.repeat(br, 5).reshape((BATCH_SIZE, 5, 1))
        bs_ = bt[:, -self.s_dim:]  # 从bt获得数据s'
        bs_ = np.reshape(bs_, (BATCH_SIZE, 5, 10))
        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(inputs=bs_)
            q_ = self.critic_target(inputs=[bs_, a_])
            q = self.critic([bs, ba])
            y = br + GAMMA * q_
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        ss = np.asarray(s).flatten()
        # s = np.concatenate([s[0], s[1][0: 2], s[2], s[3], s[4][0: 2]])
        ss_ = np.asarray(s_).flatten()
        # s_ = np.concatenate([s_[0], s_[1][0: 2], s_[2], s_[3], s_[4][0: 2]])

        # 把s, a, [r], s_横向堆叠
        transition = np.concatenate([ss, a.flatten(), [r], ss_])

        # pointer是记录了曾经有多少数据进来。
        # index是记录当前最新进来的数据位置。
        # 所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # 把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5(project_dir + 'model\\ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5(project_dir + 'model\\ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5(project_dir + 'model\\ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5(project_dir + 'model\\ddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order(project_dir + 'model\\ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order(project_dir + 'model\\ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(project_dir + 'model\\ddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order(project_dir + 'model\\ddpg_critic_target.hdf5', self.critic_target)
