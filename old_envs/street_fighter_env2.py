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


report_df = pd.DataFrame()

def save_report(self,env_name):
    new_row = pd.DataFrame([self.report])  # Wrap in list to make single-row DF
    
    # Append to main DataFrame (3 modern methods)
    
    # Method 1: pd.concat() (best for pandas >= 1.4)
    global report_df
    report_df = pd.concat([report_df, new_row], ignore_index=True)
    report_df.to_csv(f'./reports/report{env_name}.csv',sep=';')
    
if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

class streetfigher(gym.Env):
    def __init__(self):
        print('Stating env')
        super(streetfigher, self).__init__()
        self.total_ticks = 0  # Number of ticks to wait before taking a screenshot
        self.total_steps = 0
        self.enemy_hp = [0,0]
        self.actor_hp = [0,0]
        self.score = [0,0]
        self.start_time = time.time()
        self.fights_won = 0
        self.fights_lost = 0
        self.round_over = False
        
        self.status3 = deque(maxlen=3)
        self.action_combo4 = deque(maxlen=10)
        self.action_combo3 = deque(maxlen=3)
        self.energyHist = deque([0,0],maxlen=2)
        self.past_action = deque([0,0],maxlen=2)
        
        
        self.damage_taken = 0
        self.total_reward = 0
        self.stun_duration = 0

        
        self.report = {
            'env': str(gym.Env.np_random)[20:-1],
            'Total_Ticks': self.total_ticks,
            'Total_Steps': self.total_steps,
            'Fight_Num': 0,
            'result': 2,  # 0=loss, 1=win, 2=incomplete
            'fight_time':0,
            
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

        self.pyboy = PyBoy('SFA.gbc',window_type='headless',debugging=False,disable_input=False,) #window_type='headless'  # window_type='SDL2'
        self.load_file_path = "3rd_match.state"
        self.load_file_path = "ryu_arcade.sav"
        # Open the file in 'read bytes' mode and pass it to load_state
        with open(self.load_file_path, 'rb') as load_file:
            self.pyboy.load_state(load_file)
        
        self.pyboy.set_emulation_speed(0)  # Max speed

        # Define the action space (8 directions + no action)
        self.action_space = spaces.Discrete(6)
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160,3), dtype=np.uint8)

        self.coordinates = set()
        self.maps = set()

        # Define move-specific timing delays (in ticks)
        self.MOVE_DELAYS = {
            # Basic movement
            0: 2,  # Up
            1: 2,  # Down
            2: 2,  # Left
            3: 2,  # Right
            
            # Attacks
            4: 2,  # Punch (A)
            5: 2,  # Kick (B)
            
            # Special moves
            6: 12,  # Hurricane Kick
            7: 10   # Hadouken
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
        hadouken_ori0 = [3,2,5]
        hadouken_ori1 = [2,3,5]
        huricane_ori0 = [1,2,4,3]
        huricane_ori1 = [1,3,4,2]
        
        if list(status3) == [0,0,0]:
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
        press_duration = self.MOVE_DELAYS.get(action, 3)
        release_duration = 1  # Minimal release time before observation

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
        
        # Release actions
        action_release_events = self.action_to_pyboy_release_event(list(self.past_action)[0],self.orientation,self.pyboy.get_memory_value(50295))
        for event in action_release_events:
            self.pyboy.tick()
            self.pyboy.send_input(event)
            self.total_ticks += 1

        # Capture observation on the next tick after release
        #self.pyboy.tick()
        #self.total_ticks += 1



        # For special moves, add 2 extra ticks to capture animation
        if action in [6, 7]:
            self.pyboy.tick()
            self.pyboy.tick()
            self.total_ticks += 2

        #get current match time.
        self.match_time = self.pyboy.get_memory_value(53001)

        # Calculate rewards
        self.current_enemy_hp = self.pyboy.get_memory_value(50869)
        self.enemy_hp.pop(0)
        self.enemy_hp.append(self.current_enemy_hp)
        
        if self.enemy_hp[0] > self.enemy_hp[1] and self.enemy_hp != 255 and self.current_enemy_hp != 0:
            self.damage_dealt = (self.enemy_hp[0] - self.enemy_hp[1])
            self.reward = self.damage_dealt
            self.report['damage_'+str(action)] = self.report['damage_'+str(action)] + self.damage_dealt

        self.current_player_hp = self.pyboy.get_memory_value(50357)
        self.actor_hp.pop(0)
        self.actor_hp.append(self.current_player_hp)
        
        if self.actor_hp[0] > self.actor_hp[1] and self.current_player_hp != 255 and self.current_player_hp != 0:
            self.damage_taken = (self.actor_hp[1] - self.actor_hp[0])
            self.report['damage_taken'] = self.report['damage_taken'] + (self.damage_taken * -1)
            self.reward = self.damage_taken * 0.5
            
        # Round logic
        if self.current_enemy_hp == 0 and self.current_player_hp ==0 and self.round_over == False:
            self.round_over = True

        
        if self.current_enemy_hp > 0 and self.current_enemy_hp != 255 and self.current_player_hp > 0 and self.current_player_hp != 255 and self.round_over == True:
            self.round_over = False

        # Enemy died
        if self.current_enemy_hp  == 255 and self.current_player_hp != 0 and self.round_over == False:
            self.reward = self.reward + self.match_time
            self.score[0] = self.score[0] + 1
            self.report['rounds_won'] = self.score[0]
            self.round_over = True


        # Player died
        if self.current_player_hp == 255 and self.current_enemy_hp != 0 and self.round_over == False:
            self.reward = self.reward - (self.match_time/2)
            self.score[1] = self.score[1] + 1
            self.report['rounds_lost'] = self.score[1]
            self.round_over = True

            
        # Player won fight
        if self.score[0] == 2:
            self.report['Fight_Num'] = self.fights_won
            self.report['result'] = 1
            self.report['fight_time'] = self.report['fight_time'] + (90 - self.match_time)
            self.report['Total_Ticks'] = self.total_ticks
            self.report['Total_Steps'] = self.total_steps
            self.fights_won = self.fights_won + 1
            self.reward = self.reward + (50 *  self.fights_won)
            self.report['total_reward'] = self.report['total_reward'] + self.reward
            print(f"Player won fight!!!! --- {str(self.fights_won)}/4 ")
            self.score = [0,0]
            save_report(self,self.report['env'])

        # Player lost fight
        if self.score[1] == 2:
            self.report['Fight_Num'] = self.fights_won
            self.report['result'] = 0
            self.report['fight_time'] = self.report['fight_time'] + (90 - self.match_time)
            self.report['Total_Ticks'] = self.total_ticks
            self.report['Total_Steps'] = self.total_steps
            print(f"Player lost fight --- {str(self.fights_won)}/4 ")
            self.fights_lost = self.fights_lost + 1
            self.score = [0,0]
            self.reward = self.reward -50
            self.report['total_reward'] = self.report['total_reward'] + self.reward
            self.done = True
            save_report(self,self.report['env'])

        # End game conditions
        if self.fights_won == 4:
            self.reward = self.reward + 100
            self.done = True
            print(' --- Won 5 Fights -- ')

        if self.fights_won == 2:
            #self.done = True
            #self.reward = self.reward + 500
            #print(' --- Won 3 Fights -- ')
            pass

        if self.fights_lost == 10:
            self.truncated = True
            print('lost 10 Fights')
        
        # Timeout
        if self.match_time < 1:
            self.reward = self.reward - 5
            print("Timeout")
            self.truncated = True
            self.score = [0,0]

        # Return observation
        if self.pyboy.get_memory_value(50219) == 1:
            self.report['stun_time'] = self.report['stun_time'] + 1
            if self.reward == 0:
                self.reward = -0.1
            #print(f"Stunned: {str(self.reward)}")
        observation = self._get_observation()
        if self.reward != 0:
            #print(f"Reward: {str(self.reward)}")
            pass
        self.report['total_reward'] = self.report['total_reward'] + self.reward
        
        
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
        load_file_path = "ryu_arcade.sav"
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

        self.status3 = deque(maxlen=5)
        self.action_combo4 = deque(maxlen=10)
        self.action_combo3 = deque(maxlen=3)
        self.energyHist = deque([0,0],maxlen=2)
        self.past_action = deque([0,0],maxlen=2)
        
        self.enemy_hp = [0,0]
        self.actor_hp = [0,0]
        #self.total_ticks = 0
        #self.total_steps = 0
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