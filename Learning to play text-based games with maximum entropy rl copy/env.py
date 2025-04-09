import collections
import numpy as np
import random
import jericho
from jericho.template_action_generator import *
from jericho.defines import TemplateAction
from jericho.util import *
import hashlib
import re
class JerichoEnv:
    """
        Based on: Keep CALM and Explore - Yao et al. '20
        Interactive Fiction Games: A Colossal Adventure - Hausknecht '20
    """
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, rom_path, seed, step_limit=None, get_valid=False, use_aux_reward=True):
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
        #The code below for auxiliary reward
        self.good_word_list = [
            "better", "success", "open", "unlock", "new", "discover",
            "explore", "gain", "win", "reward", "treasure", "gold",
            "magic", "power", "strong", "found", "secret", "hidden",
            "achievement", "safe", "light", "progress", "advance",
            "level up", "heal", "improve", "key", "special", "victory",
            "bonus", "item", "weapon", "helpful", "great", "taken"
        ]
        self.bad_word_list = [
            "ignore", "hit", "miss", "fail", "lose", "blocked", "stuck",
            "invalid", "impossible", "wrong", "cannot", "locked", "unable",
            "error", "nothing", "empty", "dead", "danger",
            "hostile", "injured", "hurt", "poison", "damage", "useless",
            "waste", "boring", "repeat", "already", "bad", "badly", "grue", "impassable"
        ]
        self.aux_reward_value = 0.0
        self.not_repeat_value = 1
        self.visited_scenes = set()
        self.scene_visit_count = collections.defaultdict(int)
        self.punishment_multiple_same_scene_in_one_game_value = 0.08 #if entering the same scene in one game, the base punishment
        self.use_aux_reward = use_aux_reward
        

    def get_scene_id(self, observation):
        #This function returns hash of observation and avoid reward hacking through repeatedly entering same scene.
        return hashlib.md5(observation.encode('utf-8')).hexdigest()

    def calculate_aux_reward(self, observation, reward):
        aux_reward = 0.0
        scene_id = self.get_scene_id(observation)
        obs_lower = observation.lower()

        matched_good_words = [word for word in self.good_word_list 
                            if re.search(r'\b{}\b'.format(re.escape(word)), obs_lower)]
        matched_bad_words = [word for word in self.bad_word_list 
                            if re.search(r'\b{}\b'.format(re.escape(word)), obs_lower)]

        if matched_good_words and (scene_id not in self.visited_scenes) and reward == 0:
            print("Good words matched:", matched_good_words)
            print("OBSERVATION:", observation)
            aux_reward += self.aux_reward_value

        if matched_bad_words and reward == 0:
            print("Bad words matched:", matched_bad_words)
            print("OBSERVATION:", observation)
            aux_reward -= self.aux_reward_value

        
        
        
        times_visited = self.scene_visit_count[scene_id] #this gives the number of times the same scene appear in one episode
        if times_visited > 0:
            repeated_scene_punishment = times_visited * self.punishment_multiple_same_scene_in_one_game_value
            aux_reward -= repeated_scene_punishment
            print(f"Punishing repeated scene: times_visited={times_visited}, punishment={repeated_scene_punishment:.2f}")
        self.scene_visit_count[scene_id] += 1


        if scene_id not in self.visited_scenes:
            print("New Scene!")
            print("OBSERVATION:", observation)
            aux_reward += self.not_repeat_value #encourage not reentering scenes
        self.visited_scenes.add(scene_id)
        return aux_reward
    #TODO: add increasing punishment for 重复的状态，但是要在reset后清除decay
    

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
                print(f"[Step {self.steps}] Valid Actions: {valid}")
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

        if self.use_aux_reward:
            aux_reward = self.calculate_aux_reward(ob, reward)
            reward += aux_reward

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
        self.scene_visit_count.clear()
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
