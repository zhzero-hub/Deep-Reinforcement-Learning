import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import sys
sys.path.append('H:\projects\Deep-Reinforment-Learning-Linux-master\env')  # 此时假设此 py 文件和 env 文件夹在同一目录下

from devices import *


class IoTEnv(gym.Env):

    def __init__(self):
        self.t = 0
        # t: time
        self.accuracy_deficit_queue = np.zeros(Number_of_services)
        self.accuracy = np.zeros(Number_of_services)

        '''
        self.action_space = np.zeros(shape=(Number_of_total_devices, Sample_rate_types.__len__() + 1), dtype=np.int8),
        self.observation_space = np.zeros((5, Number_of_total_devices))
        '''
        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(Sample_rate_types.__len__() + 1, Number_of_total_devices), dtype=np.int8)
        '''
        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(Number_of_total_devices, Sample_rate_types.__len__()), dtype=np.int8),
            # X^t: 第i列向量表示第i个device的采样频率向量
            #      向量中只有一个值为1，表示采样频率为对应采样频率（见config.py）向量中的值
            spaces.Box(low=0, high=1, shape=(1, Number_of_total_devices), dtype=np.int8),
            # o^t: 对应的值为0表示在服务器计算，1表示在本地计算
            # spaces.Box(low=0.0, high=1.0, shape=(Number_of_services, 1), dtype=np.float32)
            # c^t: 为对应服务分配的计算资源
        ))
        '''
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0.0, high=Local_queue_capacity, shape=(1, Number_of_total_devices), dtype=np.float64),
            # B_n: local computing queues
            spaces.Box(low=0.0, high=Edge_queue_capacity, shape=(1, Number_of_services), dtype=np.float64),
            # Q_m: edge computing queues
            spaces.Box(low=0.0, high=2.0, shape=(1, Number_of_total_devices), dtype=np.float32),
            # H_n: channel conditions of all devices
            spaces.Box(low=0.0, high=Raw_data_size, shape=(1, Number_of_total_devices), dtype=np.float64),
            # \eta_n: raw data size
            spaces.Box(low=-sys.maxsize, high=sys.maxsize, shape=(1, Number_of_services), dtype=np.float64)
            # Z_m: accuracy deficit queue
        ))

        self.state = None
        pass

    def step(self, action):
        x = o = action
        datum = []
        delays = []
        for i in range(Number_of_total_devices):
            devices[i].set_sample_rate(x[i])
            devices[i].set_offload_decision(o[i])
            data = devices[i].reset_state()
            datum.append(data)
            devices[i].delay_cal()
        edge.computation_resource_allocation_cal()
        _delay = edge.delay_cal()
        for device in devices:
            delay = device.update_service_delay(_delay[device.type][device.number])
            delays.append(delay)

        self.accuracy = np.array(Long_term_accuracy) - self.accuracy_cal()

        reward = -1 * Balance_parameter * self.reward_function()
        for i in range(Number_of_services):
            reward -= self.accuracy_deficit_queue[i] * self.accuracy[i]

        self.update_accuracy_deficit_queue()
        self.update_state()

        self.t += 1
        if self.t == Time_slots_per_episode:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def update_accuracy_deficit_queue(self):
        self.accuracy_deficit_queue = np.max(self.accuracy + self.accuracy_deficit_queue, np.zeros(Number_of_services))

    def reset(self):
        self.accuracy = 0
        datum = []
        for device in devices:
            data_size = device.reset_state()
            datum.append(data_size)
        self.update_state()
        return self.state

    @staticmethod
    def reward_function():
        reward = 0
        count = 0
        for device in devices:
            reward += device.service_delay
            if device.dropped_task > 0:
                count += 1
        for i in range(Number_of_services):
            if edge.dropped_task[i] > 0:
                count += 1
        return reward + Unit_penalty_for_queue_overflow * count

    @staticmethod
    def accuracy_cal():
        _sum = np.zeros(Number_of_services)
        for device in devices:
            _sum[device.type] += device.accuracy_cal()
        return _sum

    def update_state(self):
        B = []
        Q = []
        H = []
        eta = []
        Z = []
        for device in devices:
            B.append(device.backlog_computation_task)
            Q.append(edge.backlog_computation_task[device.type])
            H.append(device.channel_state)
            eta.append(device.data_size)
            Z.append(self.accuracy_deficit_queue[device.type])
        # for i in range(Number_of_services):
        #     Q.append(edge.backlog_computation_task[i])
        #     Z.append(self.accuracy_deficit_queue[i])
        self.state = [B, Q, H, eta, Z]

    def states(self):
        return self.update_state()

    def actions(self):
        return self.action_space

    @staticmethod
    def max_episode_timesteps():
        return Time_slots_per_episode / Time_slot_duration
