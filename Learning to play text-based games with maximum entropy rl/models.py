import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
import itertools
import utils
from torch.distributions import Categorical
from utils import soft_update, pad_sequences
device = "cuda:0" if torch.cuda.is_available() else "cpu"
State = namedtuple('State', ('obs', 'look', 'inv'))


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class DoubleQCritic(nn.Module):
    """
        Based on Deep Reinforcement Relevance Network - He et al. '16
    """
    """Critic network, employes double Q-learning."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DoubleQCritic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder =  nn.GRU(embedding_dim, hidden_dim)
        self.hidden1 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.Q1 = nn.Linear(hidden_dim,1)
        self.outputs = dict()
        self.log_alpha = torch.tensor(np.log(0.1))#.to(self.device)
        self.apply(weight_init)

    def packed_rnn(self, x, rnn):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.
            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """

        lengths = torch.tensor([len(n) for n in x], dtype=torch.long)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort).to(device)
        idx_unsort = torch.autograd.Variable(idx_unsort).to(device)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths)
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0).to(device)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out

    def forward(self, state_batch, act_batch):
        state = State(*zip(*state_batch))
        act_sizes = [len(a) for a in act_batch]
        act_batch = list(itertools.chain.from_iterable(act_batch))
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(state.obs, self.obs_encoder)
        look =  self.packed_rnn(state.look, self.look_encoder)
        inv = self.packed_rnn(state.inv, self.inv_encoder)
        state_out = torch.cat((obs_out,look, inv), dim=1)
        state_out = torch.cat([state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
        z1 = F.relu(self.hidden1(z))
        z1 = F.relu(self.hidden2(z1))
        q1 = self.Q1(z1).squeeze(-1)
        self.outputs['q1'] = q1
        q1 =  q1.split(act_sizes)
        return q1

class CateoricalPolicy(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CateoricalPolicy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder =  nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.hidden = nn.Linear(4 * hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_scorer = nn.Linear(hidden_dim, 1)
        self.apply(weight_init)

    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort).to(device)
        idx_unsort = torch.autograd.Variable(idx_unsort).to(device)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths)
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0).to(device)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)

        return out

    def forward(self, state_batch, act_batch):
        state = State(*zip(*state_batch))
        act_sizes = [len(a) for a in act_batch]
        act_batch = list(itertools.chain.from_iterable(act_batch))
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        obs_out = self.packed_rnn(state.obs, self.obs_encoder)
        look =  self.packed_rnn(state.look, self.look_encoder)
        inv = self.packed_rnn(state.inv, self.inv_encoder)
        state_out = torch.cat((obs_out,look, inv), dim=1)
        # Expand the state to match the batches of actions
        state_out = torch.cat([state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        z = F.relu(self.hidden2(z))
        act_values =self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch
        return act_values.split(act_sizes)

    def act(self, states, act_ids, sample=True):
        act_values= self.forward(states, act_ids)
        if sample:
            act_probs = [F.softmax(vals, dim=0) for vals in act_values]
            probs = [Categorical(probs) \
                        for probs in act_probs]
            act_idxs = [Categorical(probs).sample() \
                        for probs in act_probs]
            Z =[(act == 0.0).float() * 1e-8 for act in act_probs]
            log_action_probs = [torch.log(a_p + z)for a_p,z in zip(act_probs,Z)]
        else:
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]
            act = [vals[idx] for vals,idx in zip(act_values,act_idxs)]
            log_prob = [torch.log(a) for a in act]
        return act_idxs,act_probs,log_action_probs
