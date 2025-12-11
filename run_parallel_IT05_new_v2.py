from os.path import exists
from pathlib import Path
import uuid
from typing import Any
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
from training_db import init_database, TrainingDBSession, TrainingMetricsCallback, get_config_dict
from tqdm import tqdm

# Note: ENV_CONFIG is only imported to inject database values into it in memory
# All actual config values during training come from the database, not ENV_CONFIG.py


# Check for TensorBoard installation
try:
    import tensorboard
    tb_available = True
except ImportError:
    tqdm.write("TensorBoard not found. Installing now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
        import tensorboard
        tb_available = True
        tqdm.write("TensorBoard installed successfully.")
    except subprocess.CalledProcessError:
        tb_available = False
        tqdm.write("Failed to install TensorBoard. Continuing without TensorBoard logging.")


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

def convert_db_value_to_type(key: str, value: Any) -> Any:
    """
    Convert database value to proper Python type based on key name and value.
    
    Args:
        key: Configuration key name
        value: Value from database (might be string)
        
    Returns:
        Converted value with proper type
    """
    if value is None:
        return None
    
    # If already correct type (int, float, bool), return as-is
    if isinstance(value, (int, float, bool)):
        return value
    
    # Try to convert to int if it looks like an integer
    if isinstance(value, str):
        # Check if it's a JSON string (list)
        if value.startswith('[') or value.startswith('{'):
            try:
                import json
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Try converting to int
        try:
            # Remove underscores (for numbers like "1_228_800")
            cleaned = value.replace('_', '')
            if cleaned.isdigit() or (cleaned.startswith('-') and cleaned[1:].isdigit()):
                return int(cleaned)
        except (ValueError, AttributeError):
            pass
        
        # Try converting to float
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        
        # Try converting to bool
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        if value.lower() in ('false', '0', 'no', 'off', ''):
            return False
    
    # Return as-is if conversion fails
    return value


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
    config_dict_raw = session_info['config']
    
    # Convert all database values to proper Python types
    config_dict = {}
    for key, value in config_dict_raw.items():
        config_dict[key] = convert_db_value_to_type(key, value)
    
    tqdm.write("\n" + "=" * 70)
    tqdm.write(f"Starting training for Session: {sess_id[:8]}")
    tqdm.write("=" * 70)
    tqdm.write(f"Session Path: {sess_path}")
    tqdm.write(f"Total Iterations: {total_iterations} (from database)")
    tqdm.write(f"Configuration Summary (from database):")
    tqdm.write(f"  NUM_ENVS: {config_dict.get('NUM_ENVS', 'N/A')}")
    tqdm.write(f"  EP_LENGTH: {config_dict.get('EP_LENGTH', 'N/A')}")
    tqdm.write(f"  BATCH_SIZE: {config_dict.get('BATCH_SIZE', 'N/A')}")
    tqdm.write(f"  N_EPOCHS: {config_dict.get('N_EPOCHS', 'N/A')}")
    tqdm.write(f"  TOTAL_TIMESTEPS: {config_dict.get('TOTAL_TIMESTEPS', 'N/A')} (stored in DB)")
    tqdm.write(f"  EMULATION_SPEED: {config_dict.get('EMULATION_SPEED', 'N/A')}")
    # Verify calculation matches
    db_total_timesteps = config_dict.get('TOTAL_TIMESTEPS')
    db_num_envs = config_dict.get('NUM_ENVS')
    db_ep_length = config_dict.get('EP_LENGTH')
    if db_total_timesteps is not None and db_num_envs is not None and db_ep_length is not None:
        # Values should already be converted to int by convert_db_value_to_type, but ensure they are
        try:
            # Handle case where value might still be string (e.g., "1_228_800")
            if isinstance(db_total_timesteps, str):
                db_total_timesteps = int(db_total_timesteps.replace('_', ''))
            else:
                db_total_timesteps = int(db_total_timesteps)
            
            if isinstance(db_num_envs, str):
                db_num_envs = int(db_num_envs.replace('_', ''))
            else:
                db_num_envs = int(db_num_envs)
                
            if isinstance(db_ep_length, str):
                db_ep_length = int(db_ep_length.replace('_', ''))
            else:
                db_ep_length = int(db_ep_length)
                
            calculated_iters = db_total_timesteps // (db_ep_length * db_num_envs)
            tqdm.write(f"  Calculated iterations: {calculated_iters} (should match {total_iterations})")
        except (ValueError, TypeError) as e:
            tqdm.write(f"  Warning: Could not verify iteration calculation: {e}")
            tqdm.write(f"    TOTAL_TIMESTEPS={db_total_timesteps} (type: {type(db_total_timesteps)})")
            tqdm.write(f"    NUM_ENVS={db_num_envs} (type: {type(db_num_envs)})")
            tqdm.write(f"    EP_LENGTH={db_ep_length} (type: {type(db_ep_length)})")
    tqdm.write("=" * 70)
    
    # Create session directory
    sess_path.mkdir(exist_ok=True)
    
    # Validate that all required config values are in the database
    required_config_keys = [
        'NUM_ENVS', 'EP_LENGTH', 'BATCH_SIZE', 'N_EPOCHS', 'TOTAL_TIMESTEPS',
        'FRAME_STACK_SIZE', 'SAVE_FREQ_MULTIPLIER', 'EVAL_FREQ_MULTIPLIER',
        'CHECKPOINT_NAME_PREFIX', 'SAVE_REPLAY_BUFFER', 'SAVE_VECNORMALIZE',
        'EVAL_DETERMINISTIC', 'EVAL_RENDER', 'RESET_NUM_TIMESTEPS', 'TB_LOG_NAME',
        'EMULATION_SPEED', 'GAME_ROM_FILE', 'SAVE_STATE_FILE'
    ]
    missing_keys = [key for key in required_config_keys if key not in config_dict]
    if missing_keys:
        raise ValueError(f"Missing required config values in database: {missing_keys}")
    
    # Import environment class (NO ENV_CONFIG import needed - environment gets config from database)
    from StreetFighter_Env2026 import streetfigher
    
    # Define environment factory that passes config_dict from database
    def make_env_for_config(rank, seed=0):
        """Create environment factory function using database config values."""
        def _init():
            set_random_seed(seed + rank)
            # Pass config_dict directly to environment (from database, not ENV_CONFIG.py)
            env = streetfigher(config_dict=config_dict)
            env.reset(seed=(seed + rank))
            return env
        return _init
    
    # Get ALL config values from database ONLY (no fallbacks)
    # Convert to proper types to ensure they're not strings (handles underscore format like "1_228_800")
    def safe_int(value):
        """Convert value to int, handling underscore format."""
        if isinstance(value, str):
            return int(value.replace('_', ''))
        return int(value)
    
    NUM_ENVS = safe_int(config_dict['NUM_ENVS'])
    EP_LENGTH = safe_int(config_dict['EP_LENGTH'])
    TOTAL_TIMESTEPS = safe_int(config_dict['TOTAL_TIMESTEPS'])
    
    # Recalculate total_iterations from database TOTAL_TIMESTEPS (use database value, not pre-calculated)
    # This ensures consistency even if database values differ from when session was created
    total_iterations = TOTAL_TIMESTEPS // (EP_LENGTH * NUM_ENVS)
    tqdm.write(f"  Recalculated iterations from DB: {total_iterations} (was stored as {session_info['total_iterations']})")
    
    SAVE_FREQ = EP_LENGTH * safe_int(config_dict['SAVE_FREQ_MULTIPLIER'])
    
    # Mark session as running
    db_session.update_session_status(sess_id, status='running', executed=False)
    
    # Check for Python 3.12+ and cloudpickle compatibility
    import os
    import cloudpickle
    
    # Allow manual override to use DummyVecEnv (single process, no multiprocessing)
    use_dummy = os.getenv('USE_DUMMY_ENV', '').lower() in ('1', 'true', 'yes')
    
    if use_dummy:
        tqdm.write(f"Using DummyVecEnv (single process) - USE_DUMMY_ENV environment variable is set")
        env = DummyVecEnv([make_env_for_config(i) for i in range(NUM_ENVS)])
    else:
        tqdm.write(f"Using SubprocVecEnv with {NUM_ENVS} parallel environments")
        tqdm.write(f"cloudpickle version: {cloudpickle.__version__}")
        env = SubprocVecEnv([make_env_for_config(i) for i in range(NUM_ENVS)])
    
    # Set session ID in all environments
    env.env_method("set_session_id", sess_id, db_path)
    
    # Add frame stacking (using database value only)
    env = VecFrameStack(env, n_stack=int(config_dict['FRAME_STACK_SIZE']))
    
    # Add image transpose (PyBoy uses HWC, PyTorch expects CHW)
    env = VecTransposeImage(env)

    # Create NEW model for this config (fresh start, using database values only)
    model = PPO(
        CustomCNNPolicy,
        env,
        verbose=1,
        n_steps=EP_LENGTH,
        batch_size=int(config_dict['BATCH_SIZE']),
        n_epochs=int(config_dict['N_EPOCHS']),
        tensorboard_log=sess_path if tb_available else None,
        device='auto'
    )
    
    # Create callbacks (using database values only)
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=sess_path,
        name_prefix=config_dict['CHECKPOINT_NAME_PREFIX'],
        save_replay_buffer=config_dict['SAVE_REPLAY_BUFFER'],
        save_vecnormalize=config_dict['SAVE_VECNORMALIZE']
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=sess_path / 'best_model',
        log_path=sess_path / 'logs',
        eval_freq=EP_LENGTH * int(config_dict['EVAL_FREQ_MULTIPLIER']),
        deterministic=bool(config_dict['EVAL_DETERMINISTIC']),
        render=bool(config_dict['EVAL_RENDER'])
    )
    
    # Create training metrics callback
    metrics_callback = TrainingMetricsCallback(db_session, sess_id, db_path)
    
    callback_list = CallbackList([checkpoint_callback, eval_callback, metrics_callback])
    
    # Training loop with progress bar
    pbar = tqdm(
        total=total_iterations,
        desc=f"Training {sess_id[:8]}",
        unit="iter",
        ncols=100,
        leave=True
    )
    
    try:
        for iteration in range(total_iterations):
            # Set iteration for environments and callback
            env.env_method("set_iteration", iteration+1)
            metrics_callback.set_iteration(iteration+1)
            
            env.reset()
            model.learn(
                total_timesteps=EP_LENGTH * NUM_ENVS,
                callback=callback_list,
                reset_num_timesteps=config_dict['RESET_NUM_TIMESTEPS'],
                tb_log_name=config_dict['TB_LOG_NAME'] if tb_available else None
            )
            
            # Update session progress
            db_session.update_session_status(sess_id, completed_iterations=iteration+1)
            
            # Save current model state
            model.save(sess_path / f"iteration_{iteration+1}")
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'iteration': f'{iteration+1}/{total_iterations}',
                'session': sess_id[:8]
            })
    
    except KeyboardInterrupt:
        pbar.close()
        tqdm.write("\nTraining interrupted. Saving final model...")
        db_session.update_session_status(sess_id, status='interrupted', executed=True)
        model.save(sess_path / "interrupted_model")
        raise  # Re-raise to stop the loop
    
    except Exception as e:
        pbar.close()
        tqdm.write(f"\nError during training: {e}")
        db_session.update_session_status(sess_id, status='interrupted', executed=True)
        model.save(sess_path / "error_model")
        raise  # Re-raise to stop the loop
    
    finally:
        pbar.close()
    
    # Mark as completed
    db_session.update_session_status(sess_id, status='completed', executed=True)
    model.save(sess_path / "final_model")
    tqdm.write(f"\n✓ Training complete for session {sess_id[:8]}")
    tqdm.write(f"  Models saved in: {sess_path}")
    
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
            tqdm.write("=" * 70)
            tqdm.write("No pending configurations found in database.")
            tqdm.write("=" * 70)
            tqdm.write("\nTo queue configurations:")
            tqdm.write("  1. Modify ENV_CONFIG.py with your desired values")
            tqdm.write("  2. Run: python init_config_queue.py")
            tqdm.write("  3. Repeat for multiple configurations")
            tqdm.write("  4. Then run this script again to train all pending configs")
            tqdm.write("")
            exit(0)
        
        tqdm.write("=" * 70)
        tqdm.write(f"Found {len(pending_sessions)} pending configuration(s)")
        tqdm.write("=" * 70)
        tqdm.write("")
        
        # Train each pending configuration with outer progress bar
        main_pbar = tqdm(
            total=len(pending_sessions),
            desc="Overall Progress",
            unit="config",
            ncols=100,
            position=0,
            leave=True
        )
        
        try:
            for idx, session_info in enumerate(pending_sessions, 1):
                try:
                    tqdm.write(f"\n[{idx}/{len(pending_sessions)}] Processing pending configuration...")
                    train_single_config(session_info, db_session, db_path)
                    main_pbar.update(1)
                    main_pbar.set_postfix({
                        'completed': f'{idx}/{len(pending_sessions)}',
                        'status': '✓'
                    })
                    tqdm.write(f"\n✓ Successfully completed configuration {idx}/{len(pending_sessions)}")
                except KeyboardInterrupt:
                    main_pbar.close()
                    tqdm.write("\n\nTraining stopped by user.")
                    tqdm.write(f"Completed: {idx-1}/{len(pending_sessions)} configurations")
                    break
                except Exception as e:
                    main_pbar.update(1)
                    tqdm.write(f"\n✗ Error in configuration {idx}/{len(pending_sessions)}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next configuration
                    continue
        finally:
            main_pbar.close()
        
        # Final summary
        remaining = db_session.get_pending_sessions()
        tqdm.write("\n" + "=" * 70)
        tqdm.write("Training Queue Summary")
        tqdm.write("=" * 70)
        tqdm.write(f"Total processed: {len(pending_sessions) - len(remaining)}")
        tqdm.write(f"Remaining pending: {len(remaining)}")
        tqdm.write("=" * 70)
