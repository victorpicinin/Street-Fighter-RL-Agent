from os.path import exists
from pathlib import Path
import uuid
from street_fighter_env2 import streetfigher
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN
import torch
import torch.nn as nn
import sys
import subprocess


# Check for TensorBoard installation
try:
    import tensorboard
    tb_available = True
except ImportError:
    print("TensorBoard not found. Installing now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
        import tensorboard
        tb_available = True
        print("TensorBoard installed successfully.")
    except subprocess.CalledProcessError:
        tb_available = False
        print("Failed to install TensorBoard. Continuing without TensorBoard logging.")

# Enhanced network architecture
class CustomNetwork(nn.Module):
    def __init__(self, observation_space, features_dim=512):
        super(CustomNetwork, self).__init__()
        self.cnn = NatureCNN(observation_space, features_dim=features_dim)
        self.additional_layers = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.features_dim = 512

    def forward(self, observations):
        return self.additional_layers(self.cnn(observations))

class CustomCNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=CustomNetwork, features_extractor_kwargs={})

class EnhancedCNN(NatureCNN):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        # Add residual connections
        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
    
    def forward(self, observations):
        x = super().forward(observations)
        residual_out = self.residual(x)
        return torch.relu(x + residual_out) 


def make_env(rank, seed=0):
    def _init():
        env = streetfigher()

        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    # Training configuration
    NUM_ENVS = 4 
    EP_LENGTH = 4096
    TOTAL_TIMESTEPS = 10_000_000
    SAVE_FREQ = EP_LENGTH * 10  # Save every 10 episodes per environment
    
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    
    # Create vectorized environments
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    
    # Add frame stacking (4 frames)
    env = VecFrameStack(env, n_stack=4)
    
    # Add image transpose (PyBoy uses HWC, PyTorch expects CHW)
    env = VecTransposeImage(env)

    # Enhanced model configuration
    model = PPO(
        CustomCNNPolicy,
        env,
        verbose=1,
        n_steps=EP_LENGTH,
        batch_size=1024,
        n_epochs=3,              # More optimization passes
        tensorboard_log=sess_path if tb_available else None,  # Only set if available
        device='auto'
    )
    #model = model.load('./session_32a0ed96/iteration_57.zip',env=env)
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=sess_path,
        name_prefix='street_fighter',
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=sess_path / 'best_model',
        log_path=sess_path / 'logs',
        eval_freq=EP_LENGTH * 10,
        deterministic=True,
        render=False
    )
    
    callback_list = CallbackList([checkpoint_callback])
    
    # Training loop with periodic saving
    try:
        total_iterations = TOTAL_TIMESTEPS // (EP_LENGTH * NUM_ENVS)
        
        for iteration in range(total_iterations):
            model.learn(
                total_timesteps=EP_LENGTH * NUM_ENVS,
                callback=callback_list,
                reset_num_timesteps=False,
                tb_log_name="PPO" if tb_available else None  # Conditional logging
            )
            
            # Save current model state
            model.save(sess_path / f"iteration_{iteration+1}")
            print(f"Completed iteration {iteration+1}/{total_iterations}")
    
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
        model.save(sess_path / "interrupted_model")
    
    # Final save
    model.save(sess_path / "final_model")
    print(f"Training complete. Models saved in {sess_path}")
    
    # Add TensorBoard to requirements.txt
    requirements_path = Path("requirements.txt")
    if tb_available and requirements_path.exists():
        with open(requirements_path, "a") as f:
            f.write("\ntensorboard==2.16.2\n")
        print("Added TensorBoard to requirements.txt")