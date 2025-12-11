"""
Initialize Database and Insert ENV_CONFIG as Pending Training Session

This script:
1. Initializes the database if it doesn't exist
2. Reads current ENV_CONFIG.py values
3. Inserts a new training session entry with Status=False (pending)
4. Allows multiple configs to be queued for automatic training
"""

import uuid
from pathlib import Path
from training_db import init_database, TrainingDBSession, get_config_dict
import ENV_CONFIG as config


def insert_config_to_queue(db_path: str = "training_results.db", description: str = None):
    """
    Insert current ENV_CONFIG as a pending training session.
    
    Args:
        db_path: Path to database file
        description: Optional description for this config (can be added to session_id or notes)
    """
    # Initialize database
    init_database(db_path)
    
    # Get current config values
    config_dict = get_config_dict()
    
    # Calculate total iterations based on current config
    NUM_ENVS = config.NUM_ENVS
    EP_LENGTH = config.EP_LENGTH
    TOTAL_TIMESTEPS = config.TOTAL_TIMESTEPS
    total_iterations = TOTAL_TIMESTEPS // (EP_LENGTH * NUM_ENVS)
    
    # Generate unique session ID
    sess_id = str(uuid.uuid4())
    sess_path = Path(f'session_{sess_id[:8]}')
    
    # Create session with executed=False (pending)
    with TrainingDBSession(db_path) as db_session:
        db_session.create_session(
            session_id=sess_id,
            session_path=sess_path,
            total_iterations=total_iterations,
            config_dict=config_dict,
            executed=False  # Status = False (pending)
        )
        
        print(f"✓ Configuration queued successfully!")
        print(f"  Session ID: {sess_id}")
        print(f"  Session Path: {sess_path}")
        print(f"  Status: Pending (executed=False)")
        print(f"  Total Iterations: {total_iterations}")
        if description:
            print(f"  Description: {description}")
        print()
        
        # Show count of pending configs
        pending = db_session.get_pending_sessions()
        print(f"Total pending configurations in queue: {len(pending)}")


if __name__ == '__main__':
    import sys
    
    db_path = "training_results.db"
    description = None
    
    if len(sys.argv) > 1:
        description = sys.argv[1]
    
    print("=" * 60)
    print("ENV_CONFIG Queue Manager")
    print("=" * 60)
    print()
    print("Inserting current ENV_CONFIG.py as pending training session...")
    print()
    
    insert_config_to_queue(db_path, description)
    
    print()
    print("=" * 60)
    print("Next steps:")
    print("  1. Modify ENV_CONFIG.py with different values")
    print("  2. Run this script again to queue another config")
    print("  3. Run run_parallel_IT05_new_v2.py to automatically train all pending configs")
    print("=" * 60)

