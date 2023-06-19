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
        self.transitions = transitions = replay_buffer.sample(
            self.batch_size,self.device)
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
