import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import jericho
from jericho import *
from jericho.util import clean
from jericho.util import *
from env import JerichoEnv
from collections import namedtuple
from memory import *
from utils import *
from sac_rs import *
import logger
import logging
import wandb
import sentencepiece as spm
from os.path import join as pjoin
import  pickle



def configure_logger(log_dir, add_tb=1, add_wb=1, args=None):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    log_types = [logger.make_output_format('csv', log_dir), logger.make_output_format('json', log_dir),
                 logger.make_output_format('stdout', log_dir)]
    if add_tb: log_types += [logger.make_output_format('tensorboard', log_dir)]
    if add_wb: log_types += [logger.make_output_format('wandb', log_dir, args=args)]
    tb = logger.Logger(log_dir, log_types)
    global log
    log = logger.log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='game')
    parser.add_argument('--rom_path', default= '/game_path')
    parser.add_argument('--env_name', default='game_name')
    parser.add_argument('--spm_path', default='.../unigram_8k.model')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--batch_size', default= 32, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--load_checkpoint', default="")
    #Reward shaping
    parser.add_argument('--reward_shaping', default=False, type=bool)
    # logger
    parser.add_argument('--tensorboard', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='', type=str)
    parser.add_argument('--wandb_group', default='', type=str)
    return parser.parse_args()


def train(agent, envs, args, max_steps, update_freq, checkpoint_freq, log_freq):
    LEARN_STEPS = int(args.max_steps)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_path)



    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.output_dir, args.env_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')


    expert_memory_replay =ReplayMemory(args.memory_size)

    #print(f'--> Initial Expert memory size: {len(expert_memory_replay)}')

    start_time = time.time()
    episode = 1
    losses = 0
    learn_steps = 0
    begin_learn = False
    episode_reward = 0
    start = time.time()
    recover_reward = 0

    obs, rewards, dones, infos, transitions = [], [], [], [], []
    env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)
    #####INITIAL_STATES####
    for env in envs:
        ob, info = env.reset()
        ob = clean(ob)
        obs, rewards, dones, infos, transitions = \
            obs + [ob], rewards + [0], dones + [False], infos + [info], transitions + [[]]

    prev_act = 'None'
    states = build_state(obs,infos,prev_act,sp)
    #valid_ids = [[a for a in info['valid']] for info in infos]
    valid_ids = [[sp.EncodeAsIds(a) for a in info['valid']] for info in infos]


    for step in range(1, max_steps+1):
        #print('Step',step)
        action_ids,action_idxs = agent.choose_action(states,valid_ids)
        #print('####126 info ',infos)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        for i, (env, action) in enumerate(zip(envs, action_strs)):
            if dones[i]:
                env_steps += infos[i]['moves']
                score = infos[i]['score']
                ob, info = env.reset()
                action_strs[i], action_ids[i], transitions[i] = 'reset', [], []
                #print('####146 done',score)
                next_obs, next_rewards, next_dones, next_infos = next_obs + [ob], next_rewards + [0], next_dones + [
                    False], next_infos + [info]
                continue


            ob, reward, done, info = env.step(action)

            score = info['score']
            next_obs, next_rewards, next_dones, next_infos = \
                next_obs + [ob], next_rewards + [reward], next_dones + [done], next_infos + [info]


        rewards, dones, infos = next_rewards, next_dones, next_infos
        log('>> Action{}: {}'.format(step, action_strs[0]))
        log("Reward{}: {}, Score {}, Done {}\n".format(step, rewards[0], infos[0]['score'], dones[0]))


        # generate valid actions
        prev_acts=action_strs
        next_states = build_state(next_obs,next_infos,prev_acts,sp)
        next_valids = [[sp.EncodeAsIds(a) for a in info['valid']] for info in infos]

        for state, act, rew, next_state, valid ,next_valid, done, transition in zip(states, action_ids, rewards, next_states,valid_ids,
                                                                 next_valids, dones, transitions):

            if len(act) != 0:  # not [] (i.e. reset)
                transition.append(Transition(state,next_state, act,valid ,next_valid, rew, done))
                expert_memory_replay.push((transition[-1]))


        obs, states, valid_ids = next_obs, next_states, next_valids

        if len(expert_memory_replay) > args.batch_size:
            # Start learning
            if begin_learn is False:
                print('Learn begins!')
                if args.reward_shaping==True:
                    print('-->using reward_shaping')
                begin_learn = True
            learn_steps += 1

            if learn_steps == LEARN_STEPS:
                print('Finished!')
                return

            critic_loss, actor_loss=agent.update(args,expert_memory_replay,logger,step)

        if step % checkpoint_freq == 0:
            agent.save()
            pickle.dump(expert_memory_replay, open(pjoin(args.output_dir, 'memory.pkl'), 'wb'))

        if  step % args.log_freq==0:
            tb.logkv('train/Last100EpisodeScores', sum(env.get_end_scores(last=100) for env in envs) / len(envs))
            tb.logkv('critc_loss',critic_loss.item())
            tb.dumpkvs()


        if step % 5000==0:
            for env in envs:
                log('env_es',env.end_scores[-100:])


def main():
    torch.set_num_threads(2)
    args = parse_args()
    configure_logger(args.output_dir, args.tensorboard, args.wandb, args)
    set_seed_everywhere(args.seed)
    agent = SACAgent(args)
    envs = [JerichoEnv(args.rom_path, args.seed, args.env_step_limit)
                for _ in range(args.num_envs)]

    train(agent, envs, args, args.max_steps, args.update_freq,
          args.checkpoint_freq, args.log_freq)


def build_state(obs,infos,prev_acts,sp):
    states = []
    for prev_act, ob, info in zip(prev_acts, obs, infos):
        inv = info['inv']
        look = info['look']
        states.append(State(sp.EncodeAsIds(ob), sp.EncodeAsIds(look), sp.EncodeAsIds(inv)))
    return states


if __name__ == "__main__":
    main()
