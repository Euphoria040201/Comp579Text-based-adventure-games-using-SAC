from collections import deque
import numpy as np
import random
import torch

from collections import namedtuple
# Define a State tuple.
State = namedtuple('State', ('obs', 'look', 'inv'))

# Extended Transition tuple for look-back advice.
Transition = namedtuple('Transition', (
    'prev_state',   # Previous state (can be None on episode start)
    'prev_valids',  # Valid actions in the previous state
    'prev_act',     # The previous action (encoded)
    'state',        # Current state
    'next_state',   # Next state
    'act',          # Action taken in current state (encoded)
    'valids',       # Valid actions for the current state
    'next_valids',  # Valid actions for the next state
    'rew',          # Received reward at this step
    'done'          # Whether the episode terminated after this transition
))

class ReplayMemory(object):
    def __init__(self, capacity, sampling_strategy='uniform'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.sampling_strategy = sampling_strategy


    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device, agent):
        """Sample a batch of transitions from the memory."""
        if self.sampling_strategy == 'uniform':
            return random.sample(self.memory, batch_size)
        elif self.sampling_strategy == 'recency':
            return self.sample_recency(batch_size)
        elif self.sampling_strategy == 'prioritized':
            return self.sample_prioritized(batch_size, agent, device)
        else:
            raise ValueError("Unsupported sampling strategy")

    def sample_recency(self, batch_size):
        # Recent transitions are given more weight
        recent_weights = np.linspace(1, 0, len(self.memory))
        probabilities = recent_weights / np.sum(recent_weights)
        indices = np.random.choice(len(self.memory), size=batch_size, p=probabilities)
        return [self.memory[i] for i in indices]

    def sample_prioritized(self, batch_size, agent, device):
        """Sample transitions based on prioritized experience replay."""
        td_errors = self.compute_td_errors(agent, device)
        probabilities = td_errors / np.sum(td_errors)
        indices = np.random.choice(len(self.memory), size=batch_size, p=probabilities)
        return [self.memory[i] for i in indices]


    def compute_td_errors(self, agent, device):
        td_errors = []
        for transition in self.memory:
            _, _, _, state, next_state, action, valid, next_valid, reward, done = transition
        
            current_state = State(*state)
            next_state = State(*next_state)
            with torch.no_grad():
                # Q-values for taken action
                current_Q1 = agent.critic1((current_state,), (valid,))[0]
                current_Q2 = agent.critic2((current_state,), (valid,))[0]
                index = valid.index(action)
                current_Q1 = current_Q1[index]
                current_Q2 = current_Q2[index]

                # Target Q-value
                next_Q1 = agent.critic_target((next_state,), (next_valid,))[0]
                next_Q2 = agent.critic_target2((next_state,), (next_valid,))[0]
                next_Q = torch.min(
                    next_Q1.max(),  
                    next_Q2.max()   
                )

                target = reward + (1 - done) * agent.discount * next_Q
                td_error1 = (target - current_Q1).abs().item()
                td_error2 = (target - current_Q2).abs().item()
                
                # Conservative error estimation taking max
                td_errors.append(max(td_error1, td_error2))

        return np.array(td_errors)



    def load(self, path, num_trajectories, seed):
        data = ExpertDataset(path, num_trajectories, seed)
        for i in range(len(data)):
            self.push(data[i])

    def __len__(self):
        return len(self.memory)
