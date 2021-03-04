import random
from env.config import *
from env.edge import *
import numpy as np
import math


class Device:
    def __init__(self, number, types):
        self.number = number  # device index
        self.type = types  # 0, 1
        self.service = types + 1  # device's type is corresponded to its service
        self.CPU_frequency = 0.0
        self.computation_intensity = 0.0
        self.capacity = 0.0
        self.dropped_task = 0.0
        self.CPU_frequency = Device_server_CPU_frequency
        self.rate = 0.0
        self.capacity = Local_queue_capacity
        self.channel_state = 1  # 0: good 1: normal 2: bad

        if types == 0:
            self.computation_intensity = Intensity_of_compressed_DNN_for_serviceI
        else:
            self.computation_intensity = Intensity_of_compressed_DNN_for_serviceII

        self.backlog_computation_task = 0.0  # B_n^t

        self.average_task_arrival_rate = 0.0  # \lambda_n^t
        self.raw_data_size = 0.0  # \eta_n^t = \lambda_n^t * v_m
        self.data_size = 0.0  # \eta(x_n^t) = sum_{k=1}^K...

        self.service_delay = 0.0  # d^t_{n, l}

        self.sample_rate = Initial_sample_rate  # x_n^t
        self.task_offload_decision = 0  # o, 0: offload 1: local

    def reset_state(self):
        self.reset_channel_state()
        self.average_task_arrival_rate = random.uniform(Average_task_arrival_rate_range[0],
                                                        Average_task_arrival_rate_range[1])
        # Question: task arrival rate follow which distribution?
        self.raw_data_size = self.average_task_arrival_rate * Raw_data_size
        index = np.argmax(self.sample_rate) + 1
        self.data_size = self.raw_data_size * index / Sample_rate_types.__len__()
        return self.data_size

    def delay_cal(self):
        if self.task_offload_decision == 1:
            delay = (self.backlog_computation_task + self.data_size) / self.CPU_frequency
            self.service_delay = delay * self.computation_intensity
            self.update_local_backlog()
        else:
            self.transmission_rate_cal()
            edge.upload_data_to_edge(device=self)
            task_offload_delay = self.data_size / self.rate
            self.service_delay = task_offload_delay
            # task_process_delay = edge.computation_intensity[self.type] * self.data_size / \
            #                           (edge.computation_resource_allocation[self.type] * edge.CPU_frequency)
            # task_queue_delay = edge.backlog_computation_task[self.type] * edge.computation_intensity[self.type] / \
            #                         (edge.computation_resource_allocation[self.type] * edge.CPU_frequency)
            # self.service_delay = task_offload_delay + task_process_delay + task_queue_delay
            # return self.service_delay

    def update_local_backlog(self):
        update_value = self.backlog_computation_task + self.data_size - \
                       self.CPU_frequency * Time_slot_duration / self.computation_intensity
        self.dropped_task = max(update_value - self.capacity, 0)
        update_value = max(update_value, 0)
        self.backlog_computation_task = min(update_value, self.capacity)

    def reset_channel_state(self):
        index = random.uniform(0.0, 1.0)
        left = 0.0
        for i in range(Channel_condition_transition_matrix[0].__len__()):
            right = left + Channel_condition_transition_matrix[self.channel_state][i]
            if left <= index < right:
                self.channel_state = i
            left = right
        self.channel_state = 1

    def transmission_rate_cal(self):
        self.rate = Communication_bandwidth / Number_of_services
        sigma_square = self.rate * Thermal_noise_spectrum_density
        channel_gain = 0.0
        if self.channel_state == 0:
            channel_gain = Channel_gain_good
        elif self.channel_state == 1:
            channel_gain = Channel_gain_normal
        else:
            channel_gain = Channel_gain_bad
        self.rate = self.rate * math.log2(1 + Transmit_power * channel_gain / Noise_figure / sigma_square)

    def accuracy_cal(self):
        index = np.argmax(self.sample_rate)
        ret = Accuracy_to_sample_rate[index] / Number_of_devices[self.type]
        if self.task_offload_decision == 1:
            return ret * Accuracy_of_compressed_DNN[self.type]
        else:
            return ret * Accuracy_of_uncompressed_DNN[self.type]

    def update_service_delay(self, x):
        self.service_delay += x
        return self.service_delay

    def set_sample_rate(self, x):
        self.sample_rate = x

    def set_offload_decision(self, o):
        self.task_offload_decision = o


devices = []
devices_for_service_I = []
devices_for_service_II = []
for i in range(Number_of_Type_I_devices):
    _device = Device(number=i, types=0)
    devices.append(_device)
    devices_for_service_I.append(_device)
for i in range(Number_of_Type_II_devices):
    _device = Device(number=i, types=1)
    devices.append(_device)
    devices_for_service_II.append(_device)
