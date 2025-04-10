import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models import *
import utils
import sentencepiece as spm
import random
from collections import namedtuple
from torch.distributions import Categorical
from utils import soft_update,  pad_sequences,set_seed_everywhere
import os.path
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0

#SAC is inspired by https://github.com/toshikwa/sac-discrete.pytorch

State = namedtuple('State', ('obs','look', 'inv'))
Transition = namedtuple('Transition', ('state', 'next_state', 'act', 'valids','next_valids', 'rew', 'done'))

class SACAgent(nn.Module):
    """SAC algorithm."""
    def __init__(self, args):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.discount = 0.90
        self.rs_discount = 0.99
        self.critic_tau = 0.1
        self.actor_update_frequency = 1
        self.critic_target_update_frequency = 2
        self.batch_size = args.batch_size
        self.learnable_temperature = None
        self.clip = 5
        self.critic1 = DoubleQCritic(len(self.sp),args.embedding_dim, args.hidden_dim).to(self.device)
        self.critic2 = DoubleQCritic(len(self.sp),args.embedding_dim, args.hidden_dim).to(self.device)
        self.critic_target = DoubleQCritic(len(self.sp),args.embedding_dim, args.hidden_dim).to(
            self.device)
        self.critic_target2 = DoubleQCritic(len(self.sp),args.embedding_dim, args.hidden_dim).to(
                    self.device)

        self.critic_target.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        self.actor = CateoricalPolicy(len(self.sp),args.embedding_dim, args.hidden_dim).to(device)#hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(0.1)).to(self.device)
        self.log_alpha.requires_grad = True


        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=0.0003,
                                                betas= [0.9, 0.999])

        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(),
                                                 lr=0.0003,
                                                 betas= [0.9, 0.999])
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(),
                                                         lr=0.0003,
                                                         betas= [0.9, 0.999])

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=0.0003,
                                                    betas= [0.9, 0.999])

        self.train()
        self.critic_target.train()
        set_seed_everywhere(args.seed)
        self.save_path =  os.path.join( args.output_dir, 'model'+'.pt')


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic1.train(training)
        self.critic2.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def choose_action(self,states, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                idxs, values,_= self.actor.act(states, poss_acts)
                act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        else:

            idxs = torch.tensor([random.randrange(len(act)) for act in poss_acts], device=device, dtype=torch.long)
            act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]

        return  act_ids,idxs

    def act(self, states, act_ids, sample= True):
        act_values = self.actor(states, act_ids)
        act_probs = [F.softmax(vals, dim=0) for vals in act_values]

        act_idxs = [torch.multinomial(probs, num_samples=1).item() \
                            for probs in act_probs]

        return act_idxs, act_values

    def update_critic(self, reward_shaping, batch,logger,step,t=0):
        with torch.no_grad():
            act_idxs, act_probs, log_prob = self.actor.act(batch.next_state,batch.next_valids)
            next_action = tuple([[next_valids[idx]] for next_valids,idx in  zip(batch.next_valids,act_idxs)])
            target_Q1 = self.critic_target(batch.next_state, batch.next_valids)
            target_Q2 = self.critic_target2(batch.next_state, batch.next_valids)
            target_V = [(act*(torch.min(t1,t2) - self.alpha.detach() * log)).sum(dim = 0, keepdim =True) for act,t1,t2,log in zip(act_probs,target_Q1,target_Q2,log_prob)]
            target_V = torch.cat((target_V),0)

            if reward_shaping == True:
                reward = torch.tensor(batch.rew,dtype=torch.float, device=device)
                current_act_idxs, current_act_probs, current_log_prob = self.actor.act(batch.state, batch.valids)
                current_Q1 = self.critic_target(batch.state,batch.valids)
                current_Q2 = self.critic_target2(batch.state,batch.valids)
                current_V = [(act*(torch.min(t1,t2) - self.alpha.detach() * log)).sum(dim = 0, keepdim =True) for act,t1,t2,log in zip(current_act_probs,current_Q1,current_Q2,current_log_prob)]
                current_V = torch.cat((current_V),0)


                current_V = [(1-0.1)*c_v + 0.1*(rew + self.rs_discount*t_v) for rew, c_v,t_v in zip(batch.rew, current_V,target_V)]
                current_V=torch.stack(current_V)

                reward_shaping = self.rs_discount*target_V - current_V
                rewards = reward_shaping + reward
            else:
                rewards = batch.rew

        target_Q = torch.tensor(rewards, dtype=torch.float, device=device) + ((1-torch.tensor(batch.done, dtype=torch.float, device=device)) * self.discount *  target_V.clone())#.detach()


        #####Q(a)############
        index = [valids.index(x) for valids, x in zip(batch.valids,batch.act)]
        index = torch.LongTensor(index).to(device)
        current_Q1 = self.critic1(batch.state,batch.valids)
        current_Q2 = self.critic2(batch.state,batch.valids)
        current_Q1 = [current_q1.gather(0,idx) for current_q1,idx in zip(current_Q1,index)]
        current_Q1= torch.stack(current_Q1)
        current_Q2 = [current_q2.gather(0,idx) for current_q2,idx in zip(current_Q2,index)]
        current_Q2= torch.stack(current_Q2)

        Q1_loss = torch.mean((current_Q1 - target_Q).pow(2) )
        Q2_loss = torch.mean((current_Q2 - target_Q).pow(2) )

        # Optimize the critic
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        Q1_loss.backward()
        Q2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()
        return Q1_loss + Q2_loss



    def update_actor_and_alpha(self, batch, logger, step):
        act_idxs, act_probs, log_prob = self.actor.act(batch.state, batch.valids)

        with torch.no_grad():
            actor_Q1 = self.critic1(batch.state,batch.valids)#,batch.valids
            actor_Q2 = self.critic2(batch.state,batch.valids)
            actor_Q = [torch.min(q1, q2) for q1, q2 in zip(actor_Q1,actor_Q2)]

        # Expectations of entropies.
        entropies = [-torch.sum(
            act * log, dim=0, keepdim=True) for act,log in zip(act_probs,log_prob)]

        # Expectations of Q.
        q = [torch.sum(torch.min(q1, q2) * act, dim=0, keepdim=True)for q1,q2,act in zip(actor_Q1,actor_Q2,act_probs)]

        entropies = torch.cat(entropies)
        q = torch.cat(q)

        actor_loss = (-q-self.alpha*entropies).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.actor_optimizer.step()

        if self.learnable_temperature:
            target_entropy = torch.tensor([[0.98 * -np.log(1 / len(i))] for i in batch.valids]).to(device)
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-entropies- target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        return actor_loss

    def update(self, args,replay_buffer, logger, step):

        transitions = replay_buffer.sample(self.batch_size, self.device, self)

        batch = Transition(*zip(*transitions))

        critic_loss = self.update_critic(args.reward_shaping, batch, logger,step)

        if step % self.actor_update_frequency == 0:
            actor_loss = self.update_actor_and_alpha(batch, logger, step)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic1, self.critic_target,
                                     self.critic_tau)
            soft_update(self.critic2, self.critic_target2,
                                                 self.critic_tau)
        return critic_loss, actor_loss

    def save(self):
        try:
            torch.save({
            'actor_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'self.critic_target1':self.critic_target.state_dict(),
            'self.critic_target2':self.critic_target2.state_dict(),
            'actor_optimizier_state_dict':self.actor_optimizer.state_dict(),
            'critic_optimizer1': self.critic_optimizer1.state_dict(),
            'critic_optimizer2': self.critic_optimizer2.state_dict()
            }, self.save_path) #pjoin(self.save_path, 'model.pt')
        except Exception as e:
            print("Error saving model.")



class REMCritic(nn.Module):
    """
    Random Ensemble Mixture (REM) Critic for SAC, it mintains original 
    DoubleQCritic interface but with ensemble voting
    Paper: "Random Ensemble Mixture for RL" (Agarwal et al., 2020)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_ensemble=4):
        super().__init__()
        
        self.num_ensemble = num_ensemble
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #shared encoders 
        self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)#(more efficient than separate ones)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        
        # Ensemble heads
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1))
            for _ in range(num_ensemble)
        ])
        for i, head in enumerate(self.ensemble):
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=0.1*i+0.1)
                    nn.init.constant_(layer.bias, 0.1*i)

    def packed_rnn(self, x, rnn):
        """Batch processing of variable-length seq"""
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long)
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort).to(device)
        idx_unsort = torch.autograd.Variable(idx_unsort).to(device)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)
        embed = self.embedding(x_tt).permute(1,0,2)
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths)
        out, _ = rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0).to(device)
        out = out.gather(0, idx).squeeze(0)
        return out.index_select(0, idx_unsort)

    def forward(self, state_batch, act_batch):
        """process each batch elmt separately to handle variable actions"""
        # =state components is processed in batch (shared across ensemble)
        obs_out = self.packed_rnn([s.obs for s in state_batch], self.obs_encoder)
        look_out = self.packed_rnn([s.look for s in state_batch], self.look_encoder)
        inv_out = self.packed_rnn([s.inv for s in state_batch], self.inv_encoder)
        state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
        
        batch_q_values = []
        for i in range(len(state_batch)):
            # actions for specific elmt
            valid_acts = act_batch[i]
            act_out = self.packed_rnn(valid_acts, self.act_encoder)
            
            #combo with state feats
            state_expanded = state_out[i].unsqueeze(0).expand(len(valid_acts), -1)
            z = torch.cat((state_expanded, act_out), dim=1)
            
            # get ensemble pred
            q_values = torch.stack([head(z).squeeze(-1) for head in self.ensemble])  # shape=[K, num_acts]
            batch_q_values.append(q_values)
            
        return batch_q_values

class REMSACAgent(SACAgent):
    """SAC with REM critics"""
    def __init__(self, args):
        super().__init__(args)
        # original critics -> to ver with REM versions
        self.critic1 = REMCritic(len(self.sp), args.embedding_dim, args.hidden_dim).to(self.device)
        self.critic2 = REMCritic(len(self.sp), args.embedding_dim, args.hidden_dim).to(self.device)
        self.critic_target = REMCritic(len(self.sp), args.embedding_dim, args.hidden_dim).to(self.device)
        self.critic_target2 = REMCritic(len(self.sp), args.embedding_dim, args.hidden_dim).to(self.device)

        self.critic_target.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.prev_targets= None

    def update_critic(self, reward_shaping, batch, logger, step):
        with torch.no_grad():
            # random mixture weights for ensemble
            # alpha = torch.rand(self.critic1.num_ensemble, device=device).softmax(0)
            # Replace random weights with prioritized ensemble
            alpha = F.softmax(torch.randn(self.critic1.num_ensemble, device=device), dim=0)
            
            #targets per batch elmt
            next_Q_values = []
            target_V = []
            for i in range(len(batch.next_state)):
                #ensemble Q-values for next state
                next_Q1 = self.critic_target([batch.next_state[i]], [batch.next_valids[i]])[0]  # shape is [K, num_acts]
                next_Q2 = self.critic_target2([batch.next_state[i]], [batch.next_valids[i]])[0]
                
                next_Q = (alpha.unsqueeze(-1) * torch.min(next_Q1, next_Q2)).sum(0)
                next_Q_values.append(next_Q.max())
                
                act_idxs, act_probs, log_prob = self.actor.act([batch.next_state[i]], [batch.next_valids[i]])
                v = (act_probs[0] * (next_Q - self.alpha.detach() * log_prob[0])).sum()
                target_V.append(v)
            
            next_Q = torch.stack(next_Q_values) 
            target_V = torch.stack(target_V)  # [B] shape

            if reward_shaping:
                reward = torch.tensor(batch.rew, dtype=torch.float, device=device)
 
                current_V = []
                for i in range(len(batch.state)):
                    current_Q1 = self.critic_target([batch.state[i]], [batch.valids[i]])[0] 
                    current_Q2 = self.critic_target2([batch.state[i]], [batch.valids[i]])[0]
                    current_Q = (alpha.unsqueeze(-1) * torch.min(current_Q1, current_Q2)).sum(0)
                    
                    act_idxs, act_probs, log_prob = self.actor.act([batch.state[i]], [batch.valids[i]])
                    v = (act_probs[0] * (current_Q - self.alpha.detach() * log_prob[0])).sum()
                    current_V.append(v)
                
                current_V = torch.stack(current_V)

                shaped_rewards = (1-0.1)*current_V + 0.1*(reward + self.rs_discount*target_V)
                reward_shaping = self.rs_discount*target_V - shaped_rewards
                targets = reward_shaping + reward
            else:
                targets = torch.tensor(batch.rew, dtype=torch.float, device=device)
            # Add target value clipping
           
            next_Q = torch.clamp(next_Q, min=-10, max=10) #-50~50
            targets = torch.clamp(targets, min=-10, max=10) #-20~20

            targets = targets + (1-torch.tensor(batch.done, dtype=torch.float, device=device)) * self.discount * next_Q
        
            

        with torch.no_grad():
            # Add EMA smoothing to targets (β=0.95)
            if self.prev_targets is None:
                batch_size = len(batch.state)
                self.prev_targets = torch.zeros(batch_size, device=device)
            raw_targets = reward_shaping + reward + self.discount * next_Q
            if self.prev_targets.size(0) != raw_targets.size(0):
                self.prev_targets = torch.zeros_like(raw_targets)
                
            targets = 0.95 * self.prev_targets + 0.05 * raw_targets
            self.prev_targets = targets.detach().clone()
            targets = torch.clamp(targets, 
                                min=-np.percentile(targets.cpu().numpy(), 10),  # preserve lower 10% 
                                max=np.percentile(targets.cpu().numpy(), 90))    #cap upper 10%
        current_Q1 = []
        current_Q2 = []
        for i in range(len(batch.state)):
            q1_all = self.critic1([batch.state[i]], [batch.valids[i]])[0] 
            q2_all = self.critic2([batch.state[i]], [batch.valids[i]])[0]

            act_idx = batch.valids[i].index(batch.act[i])

            current_Q1.append(q1_all[:, act_idx]) 
            current_Q2.append(q2_all[:, act_idx])
        
        current_Q1 = torch.stack(current_Q1).t()
        current_Q2 = torch.stack(current_Q2).t()
        Q1_loss = (alpha.unsqueeze(-1) * (current_Q1 - targets).pow(2)).mean()
        Q2_loss = (alpha.unsqueeze(-1) * (current_Q2 - targets).pow(2)).mean()

        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        Q1_loss.backward()
        Q2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()
        
        return Q1_loss + Q2_loss
    def update_actor_and_alpha(self, batch, logger, step):
        act_idxs, act_probs, log_prob = self.actor.act(batch.state, batch.valids)
        
        with torch.no_grad():
            actor_Q1 = self.critic1(batch.state, batch.valids)
            actor_Q2 = self.critic2(batch.state, batch.valids)
            actor_Q = [torch.min(q1, q2) for q1, q2 in zip(actor_Q1, actor_Q2)]

        # entropies and Q-values per batch elmt
        losses = []
        for i in range(len(batch.state)):
            q = actor_Q[i]
            act = act_probs[i]
            log_p = log_prob[i]

            if q.dim() == 1:
                q = q.unsqueeze(0)
            if act.dim() == 1:
                act = act.unsqueeze(0)
            if isinstance(log_p, torch.Tensor) and log_p.dim() == 0:
                log_p = log_p.unsqueeze(0)

            entropy = -torch.sum(act * log_p, dim=-1)
            q_value = torch.sum(q * act, dim=-1)
            losses.append(-q_value - self.alpha.detach() * entropy)
        
        # avg losses across batch
        actor_loss = torch.stack(losses).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.actor_optimizer.step()

        if self.learnable_temperature:
            avg_entropy = torch.stack([
                -torch.sum(act * log, dim=-1) 
                for act, log in zip(act_probs, log_prob)
            ]).mean()
            
            # target_entropy = 0.98 * -torch.log(1 / torch.tensor(
            #     [len(valids) for valids in batch.valids], 
            #     device=device
            # )).mean()
            target_entropy = 0.5 * -torch.log(1 / torch.tensor(
            [len(valids) for valids in batch.valids], device=device
            )).mean()
            self.alpha = torch.clamp(self.alpha, min=0.01, max=1.0)
                
            alpha_loss = (self.alpha * (target_entropy - avg_entropy)).mean()
            
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            logger.log('train_alpha/loss', alpha_loss, step)

        return actor_loss

class RNDModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNDModel, self).__init__()

        self.target = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.predictor(x), self.target(x)

class RNDAgent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.discount = 0.90
        self.batch_size = args.batch_size
        self.clip = 5

        self.critic1 = DoubleQCritic(len(self.sp), args.embedding_dim, args.hidden_dim).to(self.device)
        self.critic2 = DoubleQCritic(len(self.sp), args.embedding_dim, args.hidden_dim).to(self.device)
        self.actor = CateoricalPolicy(len(self.sp), args.embedding_dim, args.hidden_dim).to(self.device)

        self.rnd = RNDModel(args.embedding_dim, args.hidden_dim).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=1e-4)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=0.0003)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=0.0003)

        set_seed_everywhere(args.seed)

    def compute_intrinsic_reward(self, states):
        with torch.no_grad():
            features = self.actor.encode_state(states)
            pred, target = self.rnd(features)
            return F.mse_loss(pred, target, reduction='none').mean(dim=1)

    def choose_action(self,states, poss_acts, sample=True):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                idxs, values,_= self.actor.act(states, poss_acts)
                act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        else:

            idxs = torch.tensor([random.randrange(len(act)) for act in poss_acts], device=device, dtype=torch.long)
            act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]

        return  act_ids,idxs


    def update(self, args,replay_buffer, logger, step):
        transitions = replay_buffer.sample(self.batch_size, self.device, self)
        batch = Transition(*zip(*transitions))

        with torch.no_grad():
            intrinsic_reward = self.compute_intrinsic_reward(batch.state)
            reward = torch.tensor(batch.rew, dtype=torch.float, device=self.device)
            rewards = reward + args.rnd_scale * intrinsic_reward

        index = [valids.index(x) for valids, x in zip(batch.valids, batch.act)]
        index = torch.LongTensor(index).to(self.device)
        current_Q1 = self.critic1(batch.state, batch.valids)
        current_Q2 = self.critic2(batch.state, batch.valids)
        current_Q1 = torch.stack([q1.gather(0, idx) for q1, idx in zip(current_Q1, index)])
        current_Q2 = torch.stack([q2.gather(0, idx) for q2, idx in zip(current_Q2, index)])

        with torch.no_grad():
            target_Q = rewards  # no next state value for simplicity as offline RND

        Q1_loss = F.mse_loss(current_Q1, target_Q)
        Q2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        Q1_loss.backward()
        Q2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        # RND pred update
        features = self.actor.encode_state(batch.state)
        pred, target = self.rnd(features)
        rnd_loss = F.mse_loss(pred, target.detach())

        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        return Q1_loss + Q2_loss, rnd_loss

