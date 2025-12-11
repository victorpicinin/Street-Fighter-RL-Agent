from os.path import exists
from pathlib import Path
import uuid
from StreetFighter_Env2026 import streetfigher
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN
import torch
import torch.nn as nn
import sys
import subprocess
import ENV_CONFIG as config
from training_db import init_database, TrainingDBSession, TrainingMetricsCallback, get_config_dict


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


class CustomCNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_kwargs={})


def make_env(rank, seed=0):
    """Create environment factory function that's pickling-friendly for Python 3.12+."""
    # Use functools.partial or a class-based approach for better pickling
    # This helps avoid closure pickling issues with Python 3.12
    def _init():
        set_random_seed(seed + rank)
        env = streetfigher()
        env.reset(seed=(seed + rank))
        return env
    return _init

def train_single_config(session_info: dict, db_session: TrainingDBSession, db_path: str):
    """
    Train a single configuration from the database.
    
    Args:
        session_info: Dictionary with session_id, session_path, total_iterations, and config
        db_session: TrainingDBSession instance
        db_path: Path to database file
    """
    sess_id = session_info['session_id']
    sess_path = session_info['session_path']
    total_iterations = session_info['total_iterations']
    config_dict = session_info['config']
    
    print("\n" + "=" * 70)
    print(f"Starting training for Session: {sess_id[:8]}")
    print("=" * 70)
    print(f"Session Path: {sess_path}")
    print(f"Total Iterations: {total_iterations}")
    print(f"Configuration Summary:")
    print(f"  NUM_ENVS: {config_dict.get('NUM_ENVS', 'N/A')}")
    print(f"  EP_LENGTH: {config_dict.get('EP_LENGTH', 'N/A')}")
    print(f"  BATCH_SIZE: {config_dict.get('BATCH_SIZE', 'N/A')}")
    print(f"  N_EPOCHS: {config_dict.get('N_EPOCHS', 'N/A')}")
    print("=" * 70)
    
    # Create session directory
    sess_path.mkdir(exist_ok=True)
    
    # Temporarily update ENV_CONFIG module with values from database
    # Note: This modifies the config module in memory, so all environments created
    # after this will use the database-stored config values
    import ENV_CONFIG as config_module
    import json
    import importlib
    
    # Save original values to restore later if needed
    original_values = {}
    for key in config_dict.keys():
        if hasattr(config_module, key):
            original_values[key] = getattr(config_module, key)
    
    # Update config module with database values
    for key, value in config_dict.items():
        # Convert JSON strings back to lists if needed
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        setattr(config_module, key, value)
    
    # Reload the module to ensure all imports see the new values
    importlib.reload(config_module)
    
    # Also update the config reference used in this file
    import ENV_CONFIG as config
    importlib.reload(config)
    
    # Reload the environment module so it picks up the updated config
    import StreetFighter_Env2026
    importlib.reload(StreetFighter_Env2026)
    # Get the reloaded class
    streetfigher_class = StreetFighter_Env2026.streetfigher
    
    # Define a new make_env that uses the reloaded environment class
    def make_env_for_config(rank, seed=0):
        """Create environment factory function using the reloaded config."""
        def _init():
            set_random_seed(seed + rank)
            env = streetfigher_class()  # Use reloaded class
            env.reset(seed=(seed + rank))
            return env
        return _init
    
    # Get updated config values
    NUM_ENVS = config_dict.get('NUM_ENVS', config_module.NUM_ENVS)
    EP_LENGTH = config_dict.get('EP_LENGTH', config_module.EP_LENGTH)
    TOTAL_TIMESTEPS = config_dict.get('TOTAL_TIMESTEPS', config_module.TOTAL_TIMESTEPS)
    SAVE_FREQ = EP_LENGTH * config_dict.get('SAVE_FREQ_MULTIPLIER', config_module.SAVE_FREQ_MULTIPLIER)
    
    # Mark session as running
    db_session.update_session_status(sess_id, status='running', executed=False)
    
    # Check for Python 3.12+ and cloudpickle compatibility
    import os
    import cloudpickle
    
    # Allow manual override to use DummyVecEnv (single process, no multiprocessing)
    use_dummy = os.getenv('USE_DUMMY_ENV', '').lower() in ('1', 'true', 'yes')
    
    if use_dummy:
        print(f"Using DummyVecEnv (single process) - USE_DUMMY_ENV environment variable is set")
        env = DummyVecEnv([make_env_for_config(i) for i in range(NUM_ENVS)])
    else:
        print(f"Using SubprocVecEnv with {NUM_ENVS} parallel environments")
        print(f"cloudpickle version: {cloudpickle.__version__}")
        env = SubprocVecEnv([make_env_for_config(i) for i in range(NUM_ENVS)])
    
    # Set session ID in all environments
    env.env_method("set_session_id", sess_id, db_path)
    
    # Add frame stacking
    env = VecFrameStack(env, n_stack=config_dict.get('FRAME_STACK_SIZE', config_module.FRAME_STACK_SIZE))
    
    # Add image transpose (PyBoy uses HWC, PyTorch expects CHW)
    env = VecTransposeImage(env)

    # Create NEW model for this config (fresh start)
    model = PPO(
        CustomCNNPolicy,
        env,
        verbose=1,
        n_steps=EP_LENGTH,
        batch_size=config_dict.get('BATCH_SIZE', config_module.BATCH_SIZE),
        n_epochs=config_dict.get('N_EPOCHS', config_module.N_EPOCHS),
        tensorboard_log=sess_path if tb_available else None,
        device='auto'
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=sess_path,
        name_prefix=config_dict.get('CHECKPOINT_NAME_PREFIX', config_module.CHECKPOINT_NAME_PREFIX),
        save_replay_buffer=config_dict.get('SAVE_REPLAY_BUFFER', config_module.SAVE_REPLAY_BUFFER),
        save_vecnormalize=config_dict.get('SAVE_VECNORMALIZE', config_module.SAVE_VECNORMALIZE)
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=sess_path / 'best_model',
        log_path=sess_path / 'logs',
        eval_freq=EP_LENGTH * config_dict.get('EVAL_FREQ_MULTIPLIER', config_module.EVAL_FREQ_MULTIPLIER),
        deterministic=config_dict.get('EVAL_DETERMINISTIC', config_module.EVAL_DETERMINISTIC),
        render=config_dict.get('EVAL_RENDER', config_module.EVAL_RENDER)
    )
    
    # Create training metrics callback
    metrics_callback = TrainingMetricsCallback(db_session, sess_id, db_path)
    
    callback_list = CallbackList([checkpoint_callback, eval_callback, metrics_callback])
    
    # Training loop
    try:
        for iteration in range(total_iterations):
            # Set iteration for environments and callback
            env.env_method("set_iteration", iteration+1)
            metrics_callback.set_iteration(iteration+1)
            
            env.reset()
            model.learn(
                total_timesteps=EP_LENGTH * NUM_ENVS,
                callback=callback_list,
                reset_num_timesteps=config_dict.get('RESET_NUM_TIMESTEPS', config_module.RESET_NUM_TIMESTEPS),
                tb_log_name=config_dict.get('TB_LOG_NAME', config_module.TB_LOG_NAME) if tb_available else None
            )
            
            # Update session progress
            db_session.update_session_status(sess_id, completed_iterations=iteration+1)
            
            # Save current model state
            model.save(sess_path / f"iteration_{iteration+1}")
            print(f"Completed iteration {iteration+1}/{total_iterations}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        db_session.update_session_status(sess_id, status='interrupted', executed=True)
        model.save(sess_path / "interrupted_model")
        raise  # Re-raise to stop the loop
    
    except Exception as e:
        print(f"\nError during training: {e}")
        db_session.update_session_status(sess_id, status='interrupted', executed=True)
        model.save(sess_path / "error_model")
        raise  # Re-raise to stop the loop
    
    # Mark as completed
    db_session.update_session_status(sess_id, status='completed', executed=True)
    model.save(sess_path / "final_model")
    print(f"\n✓ Training complete for session {sess_id[:8]}")
    print(f"  Models saved in: {sess_path}")
    
    # Close environment to free resources
    env.close()


if __name__ == '__main__':
    # Initialize database
    db_path = "training_results.db"
    init_database(db_path)
    
    # Get all pending configurations
    with TrainingDBSession(db_path) as db_session:
        pending_sessions = db_session.get_pending_sessions()
        
        if not pending_sessions:
            print("=" * 70)
            print("No pending configurations found in database.")
            print("=" * 70)
            print("\nTo queue configurations:")
            print("  1. Modify ENV_CONFIG.py with your desired values")
            print("  2. Run: python init_config_queue.py")
            print("  3. Repeat for multiple configurations")
            print("  4. Then run this script again to train all pending configs")
            print()
            exit(0)
        
        print("=" * 70)
        print(f"Found {len(pending_sessions)} pending configuration(s)")
        print("=" * 70)
        print()
        
        # Train each pending configuration
        for idx, session_info in enumerate(pending_sessions, 1):
            try:
                print(f"\n[{idx}/{len(pending_sessions)}] Processing pending configuration...")
                train_single_config(session_info, db_session, db_path)
                print(f"\n✓ Successfully completed configuration {idx}/{len(pending_sessions)}")
            except KeyboardInterrupt:
                print("\n\nTraining stopped by user.")
                print(f"Completed: {idx-1}/{len(pending_sessions)} configurations")
                break
            except Exception as e:
                print(f"\n✗ Error in configuration {idx}/{len(pending_sessions)}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next configuration
                continue
        
        # Final summary
        remaining = db_session.get_pending_sessions()
        print("\n" + "=" * 70)
        print("Training Queue Summary")
        print("=" * 70)
        print(f"Total processed: {len(pending_sessions) - len(remaining)}")
        print(f"Remaining pending: {len(remaining)}")
        print("=" * 70)