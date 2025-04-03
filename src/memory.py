from collections import deque
import numpy as np
import random
import torch

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size,device):
        return random.sample(self.memory, batch_size)

    def load(self, path,num_trajectories,seed):
        data = ExpertDataset(path,num_trajectories, seed)
        for i in range(len(data)):
            self.push(data[i])

    def __len__(self):
        return len(self.memory)
