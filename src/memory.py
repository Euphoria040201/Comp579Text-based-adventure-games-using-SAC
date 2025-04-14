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
        """Push a new transition into memory."""
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
        """Sample based on recency (recent transitions have higher probability)."""
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
        """
        Compute TD errors for each transition.
        Since our Transition now has 10 fields:
          (prev_state, prev_valids, prev_act, state, next_state, act, valids, next_valids, rew, done)
        we ignore the first three fields here.
        """
        td_errors = []
        for transition in self.memory:
            # Unpack ignoring the previous state fields.
            _, _, _, state, next_state, action, valid, next_valid, reward, done = transition
            
            # Reconstruct State tuples for current and next states.
            current_state = State(*state)
            next_state = State(*next_state)
            with torch.no_grad():
                # Compute Q-values for the taken action in the current state.
                current_Q1 = agent.critic1((current_state,), (valid,))[0]
                current_Q2 = agent.critic2((current_state,), (valid,))[0]
                index = valid.index(action)
                current_Q1 = current_Q1[index]
                current_Q2 = current_Q2[index]

                # Compute target Q-value using next state.
                next_Q1 = agent.critic_target((next_state,), (next_valid,))[0]
                next_Q2 = agent.critic_target2((next_state,), (next_valid,))[0]
                # Here, we take the minimum of the maximum Q-values.
                next_Q = torch.min(next_Q1.max(), next_Q2.max())

                # Compute the target.
                target = reward + (1 - done) * agent.discount * next_Q
                # Compute absolute TD errors for both critics.
                td_error1 = (target - current_Q1).abs().item()
                td_error2 = (target - current_Q2).abs().item()
                # Conservative estimate: take the maximum.
                td_errors.append(max(td_error1, td_error2))
        return np.array(td_errors)

    def load(self, path, num_trajectories, seed):
        # This function assumes an ExpertDataset exists; adjust as needed.
        data = ExpertDataset(path, num_trajectories, seed)
        for i in range(len(data)):
            self.push(data[i])

    def __len__(self):
        return len(self.memory)
