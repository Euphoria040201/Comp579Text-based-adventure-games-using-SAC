import collections
import numpy as np
import random
import jericho
from jericho.template_action_generator import *
from jericho.defines import TemplateAction
from jericho.util import *

class JerichoEnv:
    """
        Based on: Keep CALM and Explore - Yao et al. '20
        Interactive Fiction Games: A Colossal Adventure - Hausknecht '20
    """
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, rom_path, seed, step_limit=None, get_valid=False):
        self.rom_path = rom_path
        self.env = jericho.FrotzEnv(rom_path,seed)
        self.bindings = self.env.bindings
        self.act_gen = TemplateActionGenerator(self.bindings)
        self.max_word_len = self.bindings['max_word_length']
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.end_scores = []


    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        cur_loc = 'unknown'
        cur_inv = 'unknown'
        info['valid'] = ['wait','yes','no']
        info['inv'] = 'unknown'
        info['look'] = 'unknown'

        if done == False:
            try:
                save = self.env.get_state()
                self.env.set_state(save)
                look, _, _, _ = self.env.step('look')
                info['look'] = look.lower()
                cur_inv,_, _, _= self.env.step('inventory')
                info['inv'] = cur_inv.lower()

                valid = self.env.get_valid_actions(use_parallel=False)

                if len(valid) == 0:
                    valid = ['wait','yes','no']
                info['valid'] = valid
            except RuntimeError:
                print('RuntimeError: {}, Done: {}, Info: {}'.format(clean(ob), done, info))
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done: self.end_scores.append(info['score'])
        return ob.lower(), reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.get_state()
        valid = self.env.get_valid_actions(use_parallel=False)
        info['valid'] = valid
        look, _, _, _ = self.env.step('look')
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        info['look'] = look
        self.env.set_state(save)
        self.steps = 0
        self.max_score = 0
        return initial_ob, info


    def get_action_set(self):
        return None

    def get_end_scores(self, last=1):
        last = min(last, len(self.end_scores))
        return sum(self.end_scores[-last:]) / last if last else 0

    def close(self):
        self.env.close()
