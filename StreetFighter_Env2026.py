import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy, WindowEvent
import numpy as np
import time
import pandas as pd
from stable_baselines3 import A2C, PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from collections import deque
#from memory_address import *

import torch
from tqdm import tqdm
import json
from typing import Dict, Any

def save_report(env_instance):
    """
    Save fight result to database.
    Creates own database connection (works with multiprocessing).
    
    Args:
        env_instance: Environment instance containing report data and session_id
    """
    if not env_instance.session_id:
        return  # No session ID set, skip saving
    
    try:
        # Always create a new connection (works in both main process and subprocesses)
        # SQLite handles concurrent access properly
        from training_db import TrainingDBSession
        with TrainingDBSession(env_instance.db_path) as temp_session:
            temp_session.save_fight_result(env_instance.session_id, env_instance.report)
    except Exception as e:
        tqdm.write(f"Warning: Failed to save fight result to database: {e}")
    
if torch.cuda.is_available():
    tqdm.write(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    tqdm.write("CUDA is not available. Using CPU.")

class streetfigher(gym.Env):
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize Street Fighter environment.
        
        Args:
            config_dict: Dictionary of configuration values from database.
                        If None, will raise error (config MUST come from database).
        """
        if config_dict is None:
            raise ValueError("config_dict must be provided from database. ENV_CONFIG.py is not used.")
        
        tqdm.write('Starting env')
        super(streetfigher, self).__init__()
        
        # Store config from database
        self.config = self._convert_config_values(config_dict)
        
        self.total_ticks = 0  # Number of ticks to wait before taking a screenshot
        self.total_steps = 0
        self.enemy_hp = [0,0]
        self.actor_hp = [0,0]
        self.score = [0,0]
        self.start_time = time.time()
        self.fights_won = 0
        self.fights_lost = 0
        self.round_over = False
        
        self.current_iteration = 0 
        
        self.status3 = deque(maxlen=self.config['STATUS_HISTORY_SIZE'])
        self.action_combo4 = deque(maxlen=self.config['ACTION_COMBO4_HISTORY_SIZE'])
        self.action_combo3 = deque(maxlen=self.config['ACTION_COMBO3_HISTORY_SIZE'])
        self.energyHist = deque([0,0],maxlen=self.config['ENERGY_HISTORY_SIZE'])
        self.past_action = deque([0,0],maxlen=self.config['PAST_ACTION_HISTORY_SIZE'])
        self.action_history = deque(maxlen=self.config['ACTION_HISTORY_SIZE'])
        
        self.damage_taken = 0
        self.total_reward = 0
        self.stun_duration = 0
        self.session_id = None  # Will be set by training script
        self.db_path = "training_results.db"  # Default database path
        
        self.report = {
            'env': str(gym.Env.np_random)[20:-1],
            'Total_Ticks': self.total_ticks,
            'Total_Steps': self.total_steps,
            'Fight_Num': 0,
            'result': 2,  # 0=loss, 1=win, 2=incomplete
            'fight_time':0,
            'iteration':1,
            # Action counts
            'action_0':0,
            'action_1':0,
            'action_2':0,
            'action_3':0,
            'action_4':0,
            'action_5':0,
            'action_6':0,
            'action_7':0,
            
            # Action damages
            'damage_0':0,
            'damage_1':0,
            'damage_2':0,
            'damage_3':0,
            'damage_4':0,
            'damage_5':0,
            'damage_6':0,
            'damage_7':0,
            
            # Additional stats
            'total_reward' :0,
            'damage_taken' :0,
            'rounds_won': 0,
            'rounds_lost': 0,
            'stun_time': 0  # If tracking stun
            }

        self.pyboy = PyBoy(self.config['GAME_ROM_FILE'],window_type='headless',debugging=False,disable_input=False,) #window_type='headless'  # window_type='SDL2'
        self.load_file_path = self.config['SAVE_STATE_FILE']
        # Open the file in 'read bytes' mode and pass it to load_state
        with open(self.load_file_path, 'rb') as load_file:
            self.pyboy.load_state(load_file)
        
        self.pyboy.set_emulation_speed(self.config['EMULATION_SPEED'])  # Max speed

        # Define the action space (8 directions + no action)
        self.action_space = spaces.Discrete(6)
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160,3), dtype=np.uint8)

        self.coordinates = set()
        self.maps = set()

        # Define move-specific timing delays (in ticks)
        self.MOVE_DELAYS = {
            # Basic movement
            0: self.config['MOVE_DELAY_UP'],
            1: self.config['MOVE_DELAY_DOWN'],
            2: self.config['MOVE_DELAY_LEFT'],
            3: self.config['MOVE_DELAY_RIGHT'],
            
            # Attacks
            4: self.config['MOVE_DELAY_PUNCH'],
            5: self.config['MOVE_DELAY_KICK'],
            
            # Special moves
            6: self.config['MOVE_DELAY_HURRICANE_KICK'],
            7: self.config['MOVE_DELAY_HADOUKEN']
        }

    def action_to_pyboy_event(self, action,orientation,energy):
    # Example mapping for pressing a button
        action_mappings = {

        0: [WindowEvent.PRESS_ARROW_UP],
        1: [WindowEvent.PRESS_ARROW_DOWN],
        2: [WindowEvent.PRESS_ARROW_LEFT],
        3: [WindowEvent.PRESS_ARROW_RIGHT],
        4: [WindowEvent.PRESS_BUTTON_A],
        5: [WindowEvent.PRESS_BUTTON_B]
        #6: "Huricane Kick",
        #7: "Hadouken",
        }

        
        #hurricane
        if action == 6 and orientation == 1:
            return [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A,WindowEvent.PRESS_ARROW_LEFT] #Hurricane Kick LEFT
        if action == 6 and orientation == 0:
            return [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A,WindowEvent.PRESS_ARROW_RIGHT] #Hurricane Kick RIGHT
        
        #hadouken
        if action == 7 and orientation == 1:
            return [WindowEvent.PRESS_ARROW_DOWN,WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B] #Hadouken LEFT
        if action == 7 and orientation == 0:
            return [WindowEvent.PRESS_ARROW_DOWN,WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B]

        return action_mappings.get(action, None)

    def action_to_pyboy_release_event(self, action,orientation,energy):
        # Example mapping for releasing a button
        release_mappings = {
        0: [WindowEvent.RELEASE_ARROW_UP],
        1: [WindowEvent.RELEASE_ARROW_DOWN],
        2: [WindowEvent.RELEASE_ARROW_LEFT],
        3: [WindowEvent.RELEASE_ARROW_RIGHT],
        4: [WindowEvent.RELEASE_BUTTON_A],
        5: [WindowEvent.RELEASE_BUTTON_B]
        #6: "Huricane Kick",
        #7: "Hadouken",
        }

        
        #hurricane
        if action == 6 and orientation == 1:
            return [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A,WindowEvent.RELEASE_ARROW_LEFT] #Hurricane Kick LEFT
        if action == 6 and orientation == 0:
            return [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A,WindowEvent.RELEASE_ARROW_RIGHT] #Hurricane Kick RIGHT
        
        #hadouken
        if action == 7 and orientation == 1:
            return [WindowEvent.RELEASE_ARROW_DOWN,WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_B] #Hadouken LEFT
        if action == 7 and orientation == 0:
            return [WindowEvent.RELEASE_ARROW_DOWN,WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_B] #Hadouken RIGHT

        return release_mappings.get(action, None)

    def verify_combos(self, action_combo4, action_combo3,orientation,status3):
        hadouken_ori0 = self.config['HADOUKEN_ORIENTATION_0']
        hadouken_ori1 = self.config['HADOUKEN_ORIENTATION_1']
        huricane_ori0 = self.config['HURRICANE_KICK_ORIENTATION_0']
        huricane_ori1 = self.config['HURRICANE_KICK_ORIENTATION_1']
        
        if list(status3) == self.config['COMBO_REQUIRED_STATUS']:
            if orientation == 0:
                if hadouken_ori0 == list(action_combo3):
                    #print('HADOUKEN')
                    self.report['action_7'] = self.report['action_7'] + 1
                    return True
                if huricane_ori0 == list(action_combo4):
                    self.report['action_6'] = self.report['action_6'] + 1
                    #print('HURICANE')
                    return True               
            if orientation == 1:
                if hadouken_ori1 == list(action_combo3):
                    #print('HADOUKEN')
                    self.report['action_7'] = self.report['action_7'] + 1
                    return True
                if huricane_ori1 == list(action_combo4):
                    self.report['action_6'] = self.report['action_6'] + 1
                    #print('HURICANE')
                    return True               
            return False
        
    def set_iteration(self, iteration):
        """Called from main training loop to set current iteration"""
        self.current_iteration = iteration
    
    def set_session_id(self, session_id, db_path="training_results.db"):
        """Set the training session ID and database path for database tracking"""
        self.session_id = session_id
        self.db_path = db_path
    
    def _convert_config_values(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert database config values to proper Python types.
        
        Args:
            config_dict: Raw config dictionary from database
            
        Returns:
            Converted config dictionary with proper types
        """
        converted = {}
        for key, value in config_dict.items():
            if value is None:
                converted[key] = None
                continue
            
            # If already correct type (int, float, bool), return as-is
            if isinstance(value, (int, float, bool)):
                converted[key] = value
                continue
            
            # Handle strings
            if isinstance(value, str):
                # Try JSON parsing for lists/dicts
                if value.startswith('[') or value.startswith('{'):
                    try:
                        converted[key] = json.loads(value)
                        continue
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # Try converting to int (handle underscore format like "1_228_800")
                try:
                    cleaned = value.replace('_', '')
                    if cleaned.isdigit() or (cleaned.startswith('-') and cleaned[1:].isdigit()):
                        converted[key] = int(cleaned)
                        continue
                except (ValueError, AttributeError):
                    pass
                
                # Try converting to float
                try:
                    converted[key] = float(value)
                    continue
                except (ValueError, TypeError):
                    pass
                
                # Try converting to bool
                if value.lower() in ('true', '1', 'yes', 'on'):
                    converted[key] = True
                    continue
                if value.lower() in ('false', '0', 'no', 'off', ''):
                    converted[key] = False
                    continue
            
            # Return as-is if no conversion worked
            converted[key] = value
        
        return converted

    def compute_repetition_penalty(self, action_history, current_action):
        
        # Count how many times the current action was repeated
        repeated = sum(1 for a in action_history if a == current_action)
        # Apply a scaled penalty after a threshold
        if repeated >= self.config['REPETITION_PENALTY_THRESHOLD']:
            return self.config['REPETITION_PENALTY_BASE'] * (repeated - (self.config['REPETITION_PENALTY_THRESHOLD'] - 1))
        return 0.0

    def compute_movement_diversity_reward(self, action_history):
        movement_keys = self.config['MOVEMENT_KEYS']
        attack_keys = self.config['ATTACK_KEYS']

        if not action_history:
            return 0.0
        if len(action_history) < self.config['MIN_ACTION_HISTORY_FOR_DIVERSITY']:
            return 0.0
        last_action = action_history[-1]

        if last_action in movement_keys:
            return 0.0

        elif last_action in attack_keys:
            movement_buffer = []

            # Check the last 3 actions before the attack
            for action in reversed(action_history[:-1]):
                if len(movement_buffer) >= 3:
                    break
                if action in movement_keys:
                    movement_buffer.append(action)

            unique_movements = set(movement_buffer)
            return self.config['MOVEMENT_DIVERSITY_BASE_REWARD'] * len(unique_movements)

        return 0.0

    def step(self, action):
        self.stunned = self.pyboy.get_memory_value(50219)
        self.status3.append(self.stunned)
        
        self.orientation= self.pyboy.get_memory_value(50866)
        self.action_combo4.append(action)
        self.action_combo3.append(action)
        
        self.past_action.append(action)

        
        #print(list(self.action_combo4))
        #print(list(self.action_combo3))
        #print("--")
        
        self.combo = self.verify_combos(self.action_combo4,self.action_combo3,self.orientation, self.status3)
        

        
        self.report['action_'+str(action)] = self.report['action_'+str(action)] + 1
        self.done = False
        self.info = {}
        self.reward = 0
        self.truncated = False
        self.total_steps += 1
        
        
        
        # Get move-specific timing
        press_duration = self.MOVE_DELAYS.get(action, self.config['DEFAULT_MOVE_DELAY'])
        release_duration = self.config['RELEASE_DURATION']

        # Press actions
        action_press_events = self.action_to_pyboy_event(action,self.orientation,self.pyboy.get_memory_value(50295))
        for event in action_press_events:
            self.pyboy.tick()
            self.pyboy.send_input(event)
            self.total_ticks += 1

        '''
        # Hold buttons for move-specific duration
        for _ in range(press_duration):
            self.pyboy.tick()
            self.total_ticks += 1
        '''
        self.pyboy.tick()
        
        # Release actions
        action_release_events = self.action_to_pyboy_release_event(list(self.past_action)[0],self.orientation,self.pyboy.get_memory_value(50295))
        for event in action_release_events:
            self.pyboy.tick()
            self.pyboy.send_input(event)
            self.total_ticks += 1

        # Capture observation on the next tick after release
        for _ in range(self.config['OBSERVATION_TICKS_AFTER_RELEASE']):
            self.pyboy.tick()
            self.total_ticks += 1

        # For special moves, add extra ticks to capture animation
        if action in [6, 7]:
            for _ in range(self.config['SPECIAL_MOVE_EXTRA_TICKS']):
                self.pyboy.tick()
                self.total_ticks += 1

        #get current match time.
        self.match_time = self.pyboy.get_memory_value(53001)

        # Calculate rewards
        self.current_enemy_hp = self.pyboy.get_memory_value(50869)
        self.enemy_hp.pop(0)
        self.enemy_hp.append(self.current_enemy_hp)
        
        # Check if enemy took damage (HP decreased and enemy is not dead)
        if self.enemy_hp[0] > self.enemy_hp[1] and self.current_enemy_hp != 255 and self.current_enemy_hp != 0:
            self.damage_dealt = (self.enemy_hp[0] - self.enemy_hp[1])
            self.reward += (self.damage_dealt * self.config['DAMAGE_DEALT_REWARD_MULTIPLIER']) + self.compute_movement_diversity_reward(list(self.action_history))
            self.report['damage_'+str(action)] = self.report['damage_'+str(action)] + self.damage_dealt

        self.current_player_hp = self.pyboy.get_memory_value(50357)
        self.actor_hp.pop(0)
        self.actor_hp.append(self.current_player_hp)
        
        if self.actor_hp[0] > self.actor_hp[1] and self.current_player_hp != 255 and self.current_player_hp != 0:
            self.damage_taken = (self.actor_hp[1] - self.actor_hp[0])
            self.report['damage_taken'] = self.report['damage_taken'] + abs(self.damage_taken)
            # Subtract damage taken penalty from reward (don't overwrite!)
            self.reward -= abs(self.damage_taken) * self.config['DAMAGE_TAKEN_PENALTY_MULTIPLIER']
            
        # Round logic: detect when both players have HP = 0 (round reset state)
        if self.current_enemy_hp == 0 and self.current_player_hp == 0 and self.round_over == False:
            self.round_over = True

        # Round logic: detect when a new round starts (both players have valid HP again)
        if (self.current_enemy_hp > 0 and self.current_enemy_hp != 255 and 
            self.current_player_hp > 0 and self.current_player_hp != 255 and 
            self.round_over == True):
            self.round_over = False

        # Enemy died
        if self.current_enemy_hp  == 255 and self.current_player_hp != 0 and self.round_over == False:
            self.reward = self.reward + (self.match_time * self.config['ROUND_WIN_TIME_MULTIPLIER'])
            self.score[0] = self.score[0] + 1
            self.report['rounds_won'] = self.score[0]
            self.report['fight_time'] = self.report['fight_time'] + (self.config['MATCH_TIME_LIMIT'] - self.match_time)
            self.round_over = True


        # Player died
        if self.current_player_hp == 255 and self.current_enemy_hp != 0 and self.round_over == False:
            self.reward = self.reward - (self.match_time / self.config['ROUND_LOSS_TIME_DIVISOR'])
            self.score[1] = self.score[1] + 1
            self.report['rounds_lost'] = self.score[1]
            self.report['fight_time'] = self.report['fight_time'] + (self.config['MATCH_TIME_LIMIT'] - self.match_time)
            self.round_over = True

            
        # Player won fight
        if self.score[0] == self.config['ROUNDS_TO_WIN_FIGHT']:
            # Calculate fight number before incrementing
            fight_number = self.fights_won + self.fights_lost + 1
            rounds_won = self.score[0]
            rounds_lost = self.score[1]
            
            self.report['Fight_Num'] = self.fights_won
            self.report['result'] = 1
            
            self.report['Total_Ticks'] = self.total_ticks
            self.report['Total_Steps'] = self.total_steps
            self.report['iteration'] = self.current_iteration
            self.fights_won = self.fights_won + 1
            self.reward = self.reward + (self.config['FIGHT_WIN_BASE_REWARD'] * self.fights_won)
            self.report['total_reward'] = round(self.report['total_reward'] + self.reward,2)
            
            # Improved log message: Fight #X won (rounds score), Episode progress
            tqdm.write(f"✓ Fight #{fight_number} WON ({rounds_won}-{rounds_lost} rounds) | Episode: {self.fights_won}W-{self.fights_lost}L | Progress: {self.fights_won}/{self.config['MAX_FIGHTS_TO_WIN']+1} fights to complete")
            self.score = [0,0]
            self.done = True
            save_report(self)

        # Player lost fight
        if self.score[1] == self.config['ROUNDS_TO_WIN_FIGHT']:
            # Calculate fight number before incrementing
            fight_number = self.fights_won + self.fights_lost + 1
            rounds_won = self.score[0]
            rounds_lost = self.score[1]
            
            self.fights_lost = self.fights_lost + 1
            # Track total fights (won + lost) for reporting
            total_fights = self.fights_won + self.fights_lost
            self.report['Fight_Num'] = total_fights - 1  # 0-indexed
            self.report['result'] = 0

            self.report['Total_Ticks'] = self.total_ticks
            self.report['Total_Steps'] = self.total_steps
            self.report['iteration'] = self.current_iteration
            
            # Improved log message: Fight #X lost (rounds score), Episode progress
            tqdm.write(f"✗ Fight #{fight_number} LOST ({rounds_won}-{rounds_lost} rounds) | Episode: {self.fights_won}W-{self.fights_lost}L")
            self.score = [0,0]
            self.reward += self.config['FIGHT_LOSS_PENALTY']
            self.report['total_reward'] = round(self.report['total_reward'] + self.reward, 2)
            self.done = True
            save_report(self)

        # End game conditions
        if self.fights_won == self.config['MAX_FIGHTS_TO_WIN']:
            self.reward = self.reward + self.config['GAME_COMPLETION_REWARD']
            self.done = True
            tqdm.write(f'🏆 EPISODE COMPLETE! Won {self.config["MAX_FIGHTS_TO_WIN"] + 1} fights (achieved {self.config["MAX_FIGHTS_TO_WIN"] + 1} wins)')

        # Check if max fights lost (truncation condition)
        if self.fights_lost >= self.config['MAX_FIGHTS_TO_LOSE']:
            self.truncated = True
            tqdm.write(f'⚠️  EPISODE TRUNCATED - Lost {self.fights_lost} fights (limit: {self.config["MAX_FIGHTS_TO_LOSE"]})')
        
        # Timeout: match time expired (time <= threshold)
        if self.match_time <= self.config['MATCH_TIME_THRESHOLD']:
            self.reward += self.config['TIMEOUT_PENALTY']
            tqdm.write(f"Timeout - Match time: {self.match_time}")
            self.truncated = True
            self.score = [0,0]

        # Apply stun penalty (always applies if stunned, regardless of other rewards)
        if self.pyboy.get_memory_value(50219) == 1:
            self.report['stun_time'] = self.report['stun_time'] + 1
            self.reward += self.config['STUN_PENALTY']
        
        # Apply repetition penalty (before final reward calculation)
        self.action_history.append(action)
        self.reward += self.compute_repetition_penalty(self.action_history, action)
        
        observation = self._get_observation()
        
        # Report total reward AFTER all calculations (including repetition penalty)
        self.report['total_reward'] = self.report['total_reward'] + self.reward
        
        # Energy tracking (for potential future rewards)
        self.energy = self.pyboy.get_memory_value(50295)
        self.energyHist.append(self.energy)
        if list(self.energyHist)[0] > list(self.energyHist)[1]:
            if self.round_over == False and self.current_enemy_hp != 144:
                pass
        if self.combo == True:
            pass

        
        return observation, self.reward, self.done, self.truncated, self.info


    def seed(self, seed=None):
        # Use a bit generator with the Generator
        self.seed_value = seed
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        load_file_path = self.config['SAVE_STATE_FILE']
        with open(self.load_file_path, 'rb') as load_file:
            self.pyboy.load_state(load_file)
        # Reset the game to start a new episode
        
        if seed is not None:
            self.seed(seed)



        observation = self._get_observation()

        self.report = {
            'env': str(gym.Env.np_random)[20:-1],
            'Total_Ticks': self.total_ticks,
            'Total_Steps': self.total_steps,
            'Fight_Num': 0,
            'result': 2,  # 0=loss, 1=win, 2=incomplete
            'fight_time':0,
            'iteration':1,
            # Action counts
            'action_0':0,
            'action_1':0,
            'action_2':0,
            'action_3':0,
            'action_4':0,
            'action_5':0,
            'action_6':0,
            'action_7':0,
            
            # Action damages
            'damage_0':0,
            'damage_1':0,
            'damage_2':0,
            'damage_3':0,
            'damage_4':0,
            'damage_5':0,
            'damage_6':0,
            'damage_7':0,
            
            # Additional stats
            'total_reward' :0,
            'damage_taken' :0,
            'rounds_won': 0,
            'rounds_lost': 0,
            'stun_time': 0  # If tracking stun
            }        
        info = {}

        for event in [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_UP,WindowEvent.RELEASE_BUTTON_A,WindowEvent.RELEASE_ARROW_LEFT]:
            self.pyboy.tick()
            self.pyboy.send_input(event)
            self.total_ticks += 1

        self.status3 = deque(maxlen=self.config['RESET_STATUS_HISTORY_SIZE'])
        self.action_combo4 = deque(maxlen=self.config['ACTION_COMBO4_HISTORY_SIZE'])
        self.action_combo3 = deque(maxlen=self.config['ACTION_COMBO3_HISTORY_SIZE'])
        self.energyHist = deque([0,0],maxlen=self.config['ENERGY_HISTORY_SIZE'])
        self.past_action = deque([0,0],maxlen=self.config['PAST_ACTION_HISTORY_SIZE'])
        self.action_history = deque(maxlen=self.config['ACTION_HISTORY_SIZE'])
        
        self.enemy_hp = [0,0]
        self.actor_hp = [0,0]
        # Note: total_ticks and total_steps are NOT reset here intentionally
        # They accumulate across episodes for reporting/tracking purposes
        # If you need per-episode tracking, uncomment the lines below:
        # self.total_ticks = 0
        # self.total_steps = 0
        self.score = [0,0]
        self.fights_won = 0
        self.fights_lost = 0
        self.round_over = False
        return observation, info

    def close(self):
        # Clean up PyBoy resources
        self.pyboy.stop()

    def _get_observation(self):
        # Capture the current screen as an image
        screen_image = self.pyboy.botsupport_manager().screen().screen_image()
        # Convert the PIL image to a numpy array
        observation = np.array(screen_image)
        return observation


def make_env(env_class, *args, **kwargs):
    """
    Utility function to create a new environment instance.
    
    Args:
    - env_class: The class of the environment to create.
    - args: Positional arguments to pass to the environment's constructor.
    - kwargs: Keyword arguments to pass to the environment's constructor.
    
    Returns:
    A callable that when called will create a new environment instance.
    """
    def _init():
        return env_class(*args, **kwargs)
    return _init

