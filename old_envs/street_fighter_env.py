import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy, WindowEvent
import numpy as np
import time
from stable_baselines3 import A2C, PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
#from memory_address import *



import torch

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

        self.pyboy = PyBoy('SFA.gbc',window_type='SDL2',debugging=False,disable_input=False,) #window_type='headless'  # window_type='SDL2'
        self.load_file_path = "ryu_arcade.sav"

        # Open the file in 'read bytes' mode and pass it to load_state
        with open(self.load_file_path, 'rb') as load_file:
            self.pyboy.load_state(load_file)

        self.pyboy.set_emulation_speed(1)  # Max speed

        # Define the action space (8 directions + no action)
        self.action_space = spaces.Discrete(8)
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160,3), dtype=np.uint8)

        self.coordinates = set()
        self.maps = set()

    def action_to_pyboy_event(self, action,orientation,energy):
    # Example mapping for pressing a button
        action_mappings = {

        0: [WindowEvent.PRESS_ARROW_UP],
        1: [WindowEvent.PRESS_ARROW_DOWN],
        2: [WindowEvent.PRESS_ARROW_LEFT],
        3: [WindowEvent.PRESS_ARROW_RIGHT],
        4: [WindowEvent.PRESS_BUTTON_A],
        5: [WindowEvent.PRESS_BUTTON_B],
        6: "Huricane Kick",
        7: "Hadouken",
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


    def action_to_pyboy_release_event(elf, action,orientation,energy):
        # Example mapping for releasing a button
        release_mappings = {
        0: [WindowEvent.RELEASE_ARROW_UP],
        1: [WindowEvent.RELEASE_ARROW_DOWN],
        2: [WindowEvent.RELEASE_ARROW_LEFT],
        3: [WindowEvent.RELEASE_ARROW_RIGHT],
        4: [WindowEvent.RELEASE_BUTTON_A],
        5: [WindowEvent.RELEASE_BUTTON_B],
        6: "Huricane Kick",
        7: "Hadouken",
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


    def step(self, action):
        self.done = False
        self.info = {}
        self.reward = 0
        self.truncated = False
        self.wait_ticks_counter = 5
        self.total_steps += 1

        # Press actions
        action_press_events = self.action_to_pyboy_event(action,self.pyboy.get_memory_value(50866),self.pyboy.get_memory_value(50295))
        for event in action_press_events:
            self.pyboy.tick()
            self.pyboy.send_input(event)

        
        # Simulate wait ticks
        while self.wait_ticks_counter > 0:
            self.wait_ticks_counter -= 1
            self.pyboy.tick()
            self.total_ticks += 1

        # Release actions
        action_release_events = self.action_to_pyboy_release_event(action,self.pyboy.get_memory_value(50866),self.pyboy.get_memory_value(50295))
        for event in action_release_events:
            self.pyboy.tick()
            self.pyboy.send_input(event)

        self.pyboy.tick()
        
        
        if action in [6]:
            self.wait_ticks_counter = 10
        else:
            self.wait_ticks_counter = 5
        self.total_ticks += 1

        # Wait for some ticks before capturing the screenshot
        while self.wait_ticks_counter > 0:
            self.wait_ticks_counter -= 1  # Decrement the wait counter
            #print(self.wait_ticks_counter)
            self.pyboy.tick()
            self.total_ticks += 1



        #enemy damaged
        self.current_enemy_hp = self.pyboy.get_memory_value(50869)
        self.enemy_hp.pop(0)
        self.enemy_hp.append(self.current_enemy_hp)
        if self.enemy_hp[0] > self.enemy_hp[1] and self.enemy_hp != 255 and self.current_enemy_hp != 0:#135 discount initial XP
            #print(f'Enemy Damaged')
            self.reward = (self.enemy_hp[0] - self.enemy_hp[1]) * 0.5
            #print(f"Damage Done: {str(self.reward)}")

        #player damaged
        self.current_player_hp = self.pyboy.get_memory_value(50357)
        self.actor_hp.pop(0)
        self.actor_hp.append(self.current_player_hp)
        if self.actor_hp[0] > self.actor_hp[1] and self.current_player_hp != 255 and self.current_player_hp != 0:#135 discount initial XP
            #print(f'Player Damaged')
            self.reward = (self.actor_hp[1] - self.actor_hp[0]) * 0.3
            #print(f"Damage Taken: {str(self.reward)}")
            pass


        if self.current_enemy_hp == 0 and self.current_player_hp ==0 and self.round_over == False:
            #print("Round Over")
            self.round_over = True
        
        if self.current_enemy_hp > 0 and self.current_enemy_hp != 255 and self.current_player_hp > 0 and self.current_player_hp != 255 and self.round_over == True:
            self.round_over = False
            #print('New Round')

        #enemy died
        if self.current_enemy_hp  == 255 and self.current_player_hp != 0 and self.round_over == False:
            self.reward = self.reward + 20
            self.score[0] = self.score[0] + 1
            self.round_over = True
            #print("Enemy died!!!!")

        #player died
        if self.current_player_hp == 255 and self.current_enemy_hp != 0 and self.round_over == False:
            self.reward = self.reward - 10
            self.score[1] = self.score[1] + 1
            self.round_over = True
            #print("Player died")

        #player won fight
        if self.score[0] == 2:
            self.reward = self.reward + 50
            print("Player won fight!!!!")
            self.fights_won = self.fights_won + 1
            self.score = [0,0]

        #player lost fight
        if self.score[1] == 2:
            #self.reward = self.reward - 1
            print("Player lost fight")
            self.fights_lost = self.fights_lost + 1
            self.score = [0,0]
            self.truncated = True

        #end game conditions
        if self.fights_won == 2:
            #self.reward = self.reward + 50
            self.done = True
            print(' --- Won 3 Fights -- ')

        if self.fights_lost == 10:
            #self.reward = self.reward - 15
            self.truncated = True
            print('lost 10 Fights')
        #timeout
        
        if self.pyboy.get_memory_value(53001) < 1:
            self.reward = self.reward - 5
            print("Timeout")
            self.truncated = True
            self.score = [0,0]

        '''
        if action not in [0,1,2,3,10,11]:
             if self.reward == 0:
                self.reward = -1
        '''

        #print(self.reward)
        observation = self._get_observation()
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

        
        info = {}
        self.enemy_hp = [0,0]
        self.actor_hp = [0,0]
        self.total_ticks = 0
        self.total_steps = 0
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
       # screen_image = cv.Canny(screen_image, 500,505)
        # Convert the PIL image to a numpy array, for example
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

from stable_baselines3.common.callbacks import EvalCallback

'''

num_cpu = 4
env = make_vec_env(make_env(streetfigher), n_envs=num_cpu)


ep_length = 3072
model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=1024, n_epochs=8, gamma=0.999,seed=1337)
#model = PPO.load("./models/best_model", env)
#model = PPO.load("midsave2", env)
# Directory where the models will be saved
log_dir = "./models"
eval_freq = ep_length # Evaluate and potentially save the model every 2 episodes, adjust as needed


# Create the evaluation environment using the same environment settings
#eval_env = make_vec_env(make_env(streetfigher), n_envs=1)


# Train the model with the callback
model.learn(total_timesteps=(ep_length)*num_cpu*100,progress_bar=True)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=40)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
model.save("CnnPolicy")
'''
