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
from drrn_agent import DRRN_Agent




def configure_logger(log_dir, add_tb=1, add_wb=True, args=None,name="default_name"):

    logger.configure(log_dir, format_strs=['log'])
    global tb
    log_types = [logger.make_output_format('csv', log_dir), logger.make_output_format('json', log_dir),
                 logger.make_output_format('stdout', log_dir)]
    if add_tb: log_types += [logger.make_output_format('tensorboard', log_dir)]

    if add_wb: log_types += [logger.make_output_format('wandb', log_dir, args=args,run_name=name)]

    tb = logger.Logger(log_dir, log_types)
    global log
    log = logger.log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='game')
    parser.add_argument('--rom_path', default= '/905.z5')
    parser.add_argument('--env_name', default='game_name')
    parser.add_argument('--spm_path', default='.../unigram_8k.model')

    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=10000, type=int)

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

    parser.add_argument('--rnd_scale', default=0.3, type=float)
    #Reward shaping
    parser.add_argument('--reward_shaping', default=True, type=bool)
    parser.add_argument('--rs_method', default='lookback', type=str)    # potential/lookback

    parser.add_argument('--use_aux_reward', default=True, type=bool)

    # logger
    parser.add_argument('--tensorboard', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='', type=str)
    parser.add_argument('--wandb_group', default='', type=str)

    parser.add_argument('--agent_type', default='SAC', type=str) #REM/RND/SAC
    parser.add_argument('--sample_strat', default='uniform', type=str) #recency,'prioritized'
    return parser.parse_args()


def build_state(obs, infos, prev_acts, sp):
    states = []
    for prev_act, ob, info in zip(prev_acts, obs, infos):
        inv = info['inv']
        look = info['look']
        states.append(State(sp.EncodeAsIds(ob), sp.EncodeAsIds(look), sp.EncodeAsIds(inv)))
    return states

def train(agent, envs, args, max_steps, update_freq, checkpoint_freq, log_freq):
    LEARN_STEPS = int(args.max_steps)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_path)

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.output_dir, args.env_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')

    expert_memory_replay = ReplayMemory(args.memory_size, sampling_strategy=args.sample_strat)

    start_time = time.time()
    learn_steps = 0
    begin_learn = False

    # Initialize environment states
    obs, rewards, dones, infos, transitions = [], [], [], [], []
    for env in envs:
        ob, info = env.reset()
        ob = clean(ob)
        obs.append(ob)
        rewards.append(0)
        dones.append(False)
        infos.append(info)
        transitions.append([])  # Each environment gets its own transition list

    # For the very first step, use a default previous action string.
    default_prev_act = 'None'
    states = build_state(obs, infos, [default_prev_act] * len(obs), sp)
    valid_ids = [[sp.EncodeAsIds(a) for a in info['valid']] for info in infos]

    # Initialize trackers for previous state, valid actions, and previous action.
    prev_states = [None for _ in envs]
    prev_valids = [None for _ in envs]
    prev_actions = [None for _ in envs]  # store encoded actions

    for step in range(1, max_steps + 1):
        # Choose actions based on current states
        action_ids, action_idxs = agent.choose_action(states, valid_ids)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        for i, (env, action) in enumerate(zip(envs, action_strs)):
            if dones[i]:
                ob, info = env.reset()
                next_obs.append(ob)
                next_rewards.append(0)
                next_dones.append(False)
                next_infos.append(info)
                continue
            ob, reward, done, info = env.step(action)
            next_obs.append(ob)
            next_rewards.append(reward)
            next_dones.append(done)
            next_infos.append(info)

        log(f'>> Action{step}: {action_strs[0]}')
        log(f"Reward{step}: {next_rewards[0]}, Score {infos[0]['score']}, Done {next_dones[0]}")

        # Build next states and valid actions (for next step, pass current actions as previous)
        next_states = build_state(next_obs, next_infos, action_strs, sp)
        next_valids = [[sp.EncodeAsIds(a) for a in info['valid']] for info in next_infos]

        # For each environment, create a Transition and push it in replay memory.
        for i, (state, act, rew, next_state, valid, next_valid, done, trans_list) in enumerate(
                zip(states, action_ids, rewards, next_states, valid_ids, next_valids, dones, transitions)):
            if len(act) != 0:  # Only if an action was taken (not a reset)
                new_transition = Transition(
                    prev_state = prev_states[i],
                    prev_valids = prev_valids[i],
                    prev_act = prev_actions[i],
                    state = state,
                    next_state = next_state,
                    act = act,
                    valids = valid,
                    next_valids = next_valid,
                    rew = next_rewards[i],
                    done = next_dones[i]
                )
                trans_list.append(new_transition)
                expert_memory_replay.push(new_transition)
            # Update trackers for environment i.
            prev_states[i] = state
            prev_valids[i] = valid
            prev_actions[i] = act  # make sure act (from action_ids) is stored in encoded format

        # Update current states for the next step.
        obs, states, valid_ids, rewards, dones, infos = next_obs, next_states, next_valids, next_rewards, next_dones, next_infos

        # Learning update
        if len(expert_memory_replay) > args.batch_size:
            if not begin_learn:
                print('Learn begins!')
                if args.reward_shaping:
                    print('--> using reward_shaping')
                begin_learn = True
            learn_steps += 1
            if learn_steps == LEARN_STEPS:
                print('Finished!')
                return
            if args.agent_type != "DRRN":
                critic_loss, actor_loss = agent.update(args, expert_memory_replay, logger, step)
            else:
                agent.update()

        if step % checkpoint_freq == 0:
            agent.save()
            pickle.dump(expert_memory_replay, open(pjoin(args.output_dir, 'memory.pkl'), 'wb'))

        if step % args.log_freq == 0:
            avg_score = sum(env.get_end_scores(last=100) for env in envs) / len(envs)
            tb.logkv('train/Last100EpisodeScores', avg_score)
            if args.agent_type != "DRRN":
                tb.logkv('critic_loss', critic_loss.item())
            tb.dumpkvs()

        if step % 5000 == 0:
            for env in envs:
                log('env_end_scores', env.end_scores[-100:])

def main():
    torch.set_num_threads(2)
    args = parse_args()

    game_name = args.rom_path.split("/")[-1]
    run_name = f"{args.seed}_{game_name}"
    configure_logger(args.output_dir, args.tensorboard, args.wandb, args, name=run_name)
    set_seed_everywhere(args.seed)
    if args.agent_type == 'SAC':
        agent = SACAgent(args)
    elif args.agent_type == "RND":
        agent = RNDAgent(args)
    elif args.agent_type == "REMSAC":
        agent = REMSACAgent(args)
    elif args.agent_type == "DRRN":
        agent = DRRN_Agent(args)
    else:
        raise ValueError("----Invalid Agent Type!----")
        
    envs = [JerichoEnv(args.rom_path, args.seed, args.env_step_limit, use_aux_reward=args.use_aux_reward)
            for _ in range(args.num_envs)]
    train(agent, envs, args, args.max_steps, args.update_freq, args.checkpoint_freq, args.log_freq)

def build_state(obs, infos, prev_acts, sp):
    states = []
    for prev_act, ob, info in zip(prev_acts, obs, infos):
        inv = info['inv']
        look = info['look']
        states.append(State(sp.EncodeAsIds(ob), sp.EncodeAsIds(look), sp.EncodeAsIds(inv)))
    return states

if __name__ == "__main__":
    main()