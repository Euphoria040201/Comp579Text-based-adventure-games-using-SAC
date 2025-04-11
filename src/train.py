import datetime
import os
import random
import time
from collections import defaultdict, deque
from itertools import count
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
from memory import ReplayMemory, State, Transition
from utils import *
from sac_rs import SACAgent, RNDAgent, REMSACAgent
import logger
import logging
import wandb
import sentencepiece as spm
from os.path import join as pjoin
import pickle

def configure_logger(log_dir, add_tb=1, add_wb=True, args=None, name="default_name"):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    log_types = [
        logger.make_output_format('csv', log_dir),
        logger.make_output_format('json', log_dir),
        logger.make_output_format('stdout', log_dir)
    ]
    if add_tb:
        log_types += [logger.make_output_format('tensorboard', log_dir)]
    if add_wb:
        log_types += [logger.make_output_format('wandb', log_dir, args=args, run_name=name)]
    tb = logger.Logger(log_dir, log_types)
    global log
    log = logger.log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='game')
    parser.add_argument('--rom_path', default='/905.z5')
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
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--load_checkpoint', default="")
    parser.add_argument('--rnd_scale', default=0.3, type=float)
    # Reward shaping options:
    parser.add_argument('--reward_shaping', default=True, type=bool)
    parser.add_argument('--rs_method', default='lookback', type=str,
                        help='Reward shaping method: potential or lookback')
    # Logger options:
    parser.add_argument('--tensorboard', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='', type=str)
    parser.add_argument('--wandb_group', default='', type=str)
    parser.add_argument('--agent_type', default='SAC', type=str)  # Options: SAC, RND, REM
    parser.add_argument('--sample_strat', default='uniform', type=str)  # Options: uniform, recency, prioritized
    return parser.parse_args()

def build_state(obs, infos, prev_acts, sp):
    states = []
    for prev_act, ob, info in zip(prev_acts, obs, infos):
        inv = info['inv']
        look = info['look']
        # Build a State tuple with encoded observation, look, and inventory.
        states.append(State(sp.EncodeAsIds(ob), sp.EncodeAsIds(look), sp.EncodeAsIds(inv)))
    return states

def train(agent, envs, args, max_steps, update_freq, checkpoint_freq, log_freq):
    LEARN_STEPS = int(args.max_steps)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_path)

    # Setup logging (Tensorboard, wandb, etc.)
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.output_dir, args.env_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')

    expert_memory_replay = ReplayMemory(args.memory_size, sampling_strategy=args.sample_strat)

    start_time = time.time()
    learn_steps = 0
    begin_learn = False

    # Initialize lists for initial states.
    obs, rewards, dones, infos, transitions = [], [], [], [], []
    for env in envs:
        ob, info = env.reset()
        ob = clean(ob)
        obs.append(ob)
        rewards.append(0)
        dones.append(False)
        infos.append(info)
        transitions.append([])  # Each environment keeps its own list of transitions

    # For the very first step, we use a default previous action.
    default_prev_act = 'None'
    states = build_state(obs, infos, [default_prev_act] * len(obs), sp)
    valid_ids = [[sp.EncodeAsIds(a) for a in info['valid']] for info in infos]

    # Initialize trackers for previous state, valid actions, and previous action per environment.
    prev_states = [None for _ in envs]
    prev_valids = [None for _ in envs]
    prev_actions = [None for _ in envs]  # For encoded actions, you may choose to store action_ids

    for step in range(1, max_steps + 1):
        # Choose actions based on the current states.
        action_ids, action_idxs = agent.choose_action(states, valid_ids)
        # Convert chosen action indices to string for logging.
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]
        
        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        # Step through each environment.
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
        
        # Log info from the first environment.
        log(f'>> Action{step}: {action_strs[0]}')
        log(f"Reward{step}: {next_rewards[0]}, Score {infos[0]['score']}, Done {next_dones[0]}")
        
        # Build next states and valid action lists.
        # For building next states, we pass the current actions as the "previous action" for next state.
        next_states = build_state(next_obs, next_infos, action_strs, sp)
        next_valids = [[sp.EncodeAsIds(a) for a in info['valid']] for info in next_infos]
        
        # For each environment, create a new Transition and push it into replay memory.
        for i in range(len(states)):
            if len(action_ids[i]) != 0:  # Only add a transition if an action was taken (not a reset)
                transition = Transition(
                    prev_state = prev_states[i],
                    prev_valids = prev_valids[i],
                    prev_act = prev_actions[i],
                    state = states[i],
                    next_state = next_states[i],
                    act = action_ids[i],
                    valids = valid_ids[i],
                    next_valids = next_valids[i],
                    rew = next_rewards[i],
                    done = next_dones[i]
                )
                transitions[i].append(transition)
                expert_memory_replay.push(transition)
            # Update trackers for environment i.
            prev_states[i] = states[i]
            prev_valids[i] = valid_ids[i]
            # To be consistent, store encoded action here (action_ids) rather than string.
            prev_actions[i] = action_ids[i]
        
        # Update current states for the next step.
        obs, states, valid_ids, rewards, dones, infos = next_obs, next_states, next_valids, next_rewards, next_dones, next_infos
        
        # Begin learning when enough transitions are collected.
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
            critic_loss, actor_loss = agent.update(args, expert_memory_replay, logger, step)
        
        if step % checkpoint_freq == 0:
            agent.save()
            pickle.dump(expert_memory_replay, open(pjoin(args.output_dir, 'memory.pkl'), 'wb'))
        
        if step % args.log_freq == 0:
            avg_score = sum(env.get_end_scores(last=100) for env in envs) / len(envs)
            tb.logkv('train/Last100EpisodeScores', avg_score)
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
    else:
        agent = REMSACAgent(args)
    
    envs = [JerichoEnv(args.rom_path, args.seed, args.env_step_limit) for _ in range(args.num_envs)]
    train(agent, envs, args, args.max_steps, args.update_freq, args.checkpoint_freq, args.log_freq)

if __name__ == "__main__":
    main()
