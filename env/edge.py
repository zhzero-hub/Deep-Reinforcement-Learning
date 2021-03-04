import numpy as np
from env.config import *


class Edge:
    def __init__(self, number):
        self.number = number  # device index
        self.CPU_frequency = Edge_server_CPU_frequency
        self.computation_intensity = np.array(Intensity_of_uncompressed_DNN_for_service)

        self.data_uploaded = np.zeros((Number_of_services, Number_of_total_devices))
        self.data_delay = np.zeros((Number_of_services, Number_of_total_devices))
        self.capacity = np.array(Edge_queue_capacity)
        # 只给了总edge的queue capacity，没有具体到每个service的
        self.dropped_task = np.zeros(Number_of_services)

        self.backlog_computation_task = np.zeros(Number_of_services)  # Q_m^t

        self.computation_resource_allocation = np.empty(Number_of_services)

    def computation_resource_allocation_cal(self):
        _sum = np.zeros((Number_of_services, Number_of_total_devices))
        for i in range(Number_of_services):
            _sum[i] += self.computation_intensity[i] * self.data_uploaded[i]
            _sum[i] += self.computation_intensity[i] * self.data_delay[i] / 2
            _sum[i] += self.backlog_computation_task[i] * self.computation_intensity[i]
        molecule = _sum.sum(axis=1)
        denominator = molecule.sum()
        self.computation_resource_allocation = molecule / denominator

    def upload_data_to_edge(self, device):
        self.data_uploaded[device.type][self.number] += device.data_size

    def reset_data_uploaded(self):
        self.data_uploaded = np.zeros(Number_of_services)

    def update_edge_backlog(self):
        data_uploaded_sum = self.data_uploaded.sum(axis=1)
        update_value = self.backlog_computation_task + data_uploaded_sum - \
                       (self.computation_resource_allocation * self.CPU_frequency * Time_slot_duration /
                        self.computation_intensity)
        self.dropped_task = np.max(update_value - self.capacity, np.zeros(Number_of_services))
        update_value = np.max(update_value, np.zeros(Number_of_services))
        self.backlog_computation_task = np.min(update_value, self.capacity)

    def delay_cal(self):
        # task_process_delay = edge.computation_intensity[self.type] * self.data_size / \
        #                           (edge.computation_resource_allocation[self.type] * edge.CPU_frequency)
        # task_queue_delay = edge.backlog_computation_task[self.type] * edge.computation_intensity[self.type] / \
        #                         (edge.computation_resource_allocation[self.type] * edge.CPU_frequency)
        # self.service_delay = task_offload_delay + task_process_delay + task_queue_delay
        # return self.service_delay

        _sum = self.data_uploaded.sum(axis=1)
        task_process_delay = np.zeros((Number_of_services, Number_of_total_devices))
        task_queue_delay = np.zeros((Number_of_services, Number_of_total_devices))
        wait_time_delay = np.zeros((Number_of_services, Number_of_total_devices))
        for i in range(Number_of_services):
            task_process_delay[i] = self.computation_intensity[i] * self.data_uploaded[i] / \
                                    (self.computation_resource_allocation[i] * self.CPU_frequency)
            task_queue_delay[i] = self.backlog_computation_task[i] * self.computation_intensity[i] / \
                                  (self.computation_resource_allocation[i] * self.CPU_frequency)
            wait_time_delay[i] = _sum[i] - self.data_uploaded[i]
            self.data_delay[i] = wait_time_delay[i]
            wait_time_delay[i] = self.computation_intensity[i] * wait_time_delay[i] / \
                                 (2 * self.computation_resource_allocation[i] * self.CPU_frequency)
        return task_process_delay + task_queue_delay + wait_time_delay


edge = Edge(1)
