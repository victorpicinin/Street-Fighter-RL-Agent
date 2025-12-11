from os.path import exists
from pathlib import Path
import uuid
from street_fighter_env import StreetFighter
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = StreetFighter()
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init




if __name__ == '__main__':

    use_wandb_logging = False

    ep_length = 2048
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    checkpoint_callback = CheckpointCallback(save_freq=ep_length*8, save_path=sess_path,
                                     name_prefix='street')
    
    num_cpu = 4  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    


    model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=3, gamma=0.998, tensorboard_log=sess_path)

    model.learn(total_timesteps=(ep_length)*num_cpu,progress_bar=True, callback=checkpoint_callback)
    learn_steps = 20

    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu, callback=checkpoint_callback)
