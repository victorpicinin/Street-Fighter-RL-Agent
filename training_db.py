"""
Training Database Module
Manages SQLite database for storing training sessions, fight results, and training metrics.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
from stable_baselines3.common.callbacks import BaseCallback


def get_config_dict() -> Dict[str, Any]:
    """Extract all configuration values from ENV_CONFIG as a dictionary."""
    import ENV_CONFIG as config
    
    config_dict = {}
    # Get all uppercase attributes from config module
    for attr_name in dir(config):
        if attr_name.isupper() and not attr_name.startswith('_'):
            value = getattr(config, attr_name)
            # Convert lists to JSON strings for storage
            if isinstance(value, list):
                value = json.dumps(value)
            config_dict[attr_name] = value
    
    return config_dict


def init_database(db_path: str = "training_results.db") -> None:
    """
    Initialize SQLite database and create all tables with proper relationships.
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Get config dict to build columns dynamically
    config_dict = get_config_dict()
    
    # Build column definitions for ENV_CONFIG parameters
    config_columns = []
    for key, value in config_dict.items():
        # Determine SQL type based on Python type
        if isinstance(value, (int, bool)):
            sql_type = "INTEGER"
        elif isinstance(value, float):
            sql_type = "REAL"
        elif isinstance(value, str):
            # Lists are stored as JSON strings
            sql_type = "TEXT"
        else:
            sql_type = "TEXT"
        config_columns.append(f"{key} {sql_type}")
    
    config_columns_str = ",\n    ".join(config_columns)
    
    # Create training_sessions table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS training_sessions (
            session_id TEXT PRIMARY KEY,
            session_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'running' CHECK(status IN ('running', 'completed', 'interrupted')),
            executed INTEGER NOT NULL DEFAULT 0 CHECK(executed IN (0, 1)),
            total_iterations INTEGER NOT NULL,
            completed_iterations INTEGER DEFAULT 0,
            {config_columns_str}
        )
    """)
    
    # Create indexes for training_sessions
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON training_sessions(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_executed ON training_sessions(executed)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON training_sessions(created_at)")
    
    # Add executed column if table exists but column doesn't (for migration)
    try:
        cursor.execute("ALTER TABLE training_sessions ADD COLUMN executed INTEGER NOT NULL DEFAULT 0 CHECK(executed IN (0, 1))")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    # Create fight_results table with foreign key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fight_results (
            fight_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            env_instance_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_ticks INTEGER NOT NULL,
            total_steps INTEGER NOT NULL,
            fight_num INTEGER NOT NULL,
            result INTEGER NOT NULL CHECK(result IN (0, 1, 2)),
            fight_time REAL NOT NULL,
            iteration INTEGER NOT NULL,
            action_0 INTEGER DEFAULT 0,
            action_1 INTEGER DEFAULT 0,
            action_2 INTEGER DEFAULT 0,
            action_3 INTEGER DEFAULT 0,
            action_4 INTEGER DEFAULT 0,
            action_5 INTEGER DEFAULT 0,
            action_6 INTEGER DEFAULT 0,
            action_7 INTEGER DEFAULT 0,
            damage_0 REAL DEFAULT 0.0,
            damage_1 REAL DEFAULT 0.0,
            damage_2 REAL DEFAULT 0.0,
            damage_3 REAL DEFAULT 0.0,
            damage_4 REAL DEFAULT 0.0,
            damage_5 REAL DEFAULT 0.0,
            damage_6 REAL DEFAULT 0.0,
            damage_7 REAL DEFAULT 0.0,
            total_reward REAL NOT NULL,
            damage_taken REAL DEFAULT 0.0,
            rounds_won INTEGER DEFAULT 0,
            rounds_lost INTEGER DEFAULT 0,
            stun_time INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES training_sessions(session_id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes for fight_results
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fights_session_id ON fight_results(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fights_session_iteration ON fight_results(session_id, iteration)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fights_session_result ON fight_results(session_id, result)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fights_created_at ON fight_results(created_at)")
    
    # Create training_metrics table with foreign key and unique constraint
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            timesteps_total INTEGER NOT NULL,
            policy_loss REAL,
            value_loss REAL,
            entropy_loss REAL,
            approx_kl REAL,
            clip_fraction REAL,
            explained_variance REAL,
            learning_rate REAL,
            n_updates INTEGER,
            FOREIGN KEY (session_id) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
            UNIQUE(session_id, iteration)
        )
    """)
    
    # Create indexes for training_metrics
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON training_metrics(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session_iteration ON training_metrics(session_id, iteration)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_created_at ON training_metrics(created_at)")
    
    # Create trigger to auto-update updated_at timestamp
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp 
        AFTER UPDATE ON training_sessions
        FOR EACH ROW
        BEGIN
            UPDATE training_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = NEW.session_id;
        END
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {db_path}")


class TrainingDBSession:
    """
    Context manager for database operations.
    Provides methods to interact with the training database.
    """
    
    def __init__(self, db_path: str = "training_results.db"):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def create_session(self, session_id: str, session_path: str, total_iterations: int, 
                       config_dict: Optional[Dict[str, Any]] = None, executed: bool = False) -> None:
        """
        Create a new training session with all ENV_CONFIG parameters.
        
        Args:
            session_id: Unique session identifier (UUID)
            session_path: Path where models are saved
            total_iterations: Planned total iterations
            config_dict: Dictionary of ENV_CONFIG values (if None, reads from ENV_CONFIG module)
        """
        if config_dict is None:
            config_dict = get_config_dict()
        
        # Build column names and values
        columns = ['session_id', 'session_path', 'total_iterations', 'executed']
        placeholders = ['?', '?', '?', '?']
        values = [session_id, str(session_path), total_iterations, 1 if executed else 0]
        
        for key, value in config_dict.items():
            columns.append(key)
            placeholders.append('?')
            # Convert lists to JSON strings if needed
            if isinstance(value, list):
                value = json.dumps(value)
            values.append(value)
        
        columns_str = ', '.join(columns)
        placeholders_str = ', '.join(placeholders)
        
        query = f"INSERT INTO training_sessions ({columns_str}) VALUES ({placeholders_str})"
        
        cursor = self.conn.cursor()
        cursor.execute(query, values)
        self.conn.commit()
    
    def update_session_status(self, session_id: str, status: str = None, 
                             completed_iterations: int = None, executed: bool = None) -> None:
        """
        Update training session status and/or completed iterations.
        
        Args:
            session_id: Session identifier
            status: New status ('running', 'completed', 'interrupted')
            completed_iterations: Number of completed iterations
            executed: Whether this config has been executed (True = completed)
        """
        updates = []
        values = []
        
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        
        if completed_iterations is not None:
            updates.append("completed_iterations = ?")
            values.append(completed_iterations)
        
        if executed is not None:
            updates.append("executed = ?")
            values.append(1 if executed else 0)
        
        if updates:
            values.append(session_id)
            query = f"UPDATE training_sessions SET {', '.join(updates)} WHERE session_id = ?"
            cursor = self.conn.cursor()
            cursor.execute(query, values)
            self.conn.commit()
    
    def get_pending_sessions(self) -> list:
        """
        Get all training sessions that haven't been executed yet (executed = 0).
        
        Returns:
            List of dictionaries containing session_id and config_dict for each pending session
        """
        cursor = self.conn.cursor()
        
        # Get all columns except metadata columns
        cursor.execute("PRAGMA table_info(training_sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        metadata_cols = ['session_id', 'session_path', 'created_at', 'updated_at', 
                        'status', 'executed', 'total_iterations', 'completed_iterations']
        config_cols = [col for col in columns if col not in metadata_cols]
        
        all_cols_str = ', '.join(['session_id', 'session_path', 'total_iterations'] + config_cols)
        
        query = f"SELECT {all_cols_str} FROM training_sessions WHERE executed = 0 ORDER BY created_at"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        pending_sessions = []
        for row in rows:
            session_id = row[0]
            session_path = row[1]
            total_iterations = row[2]
            
            # Build config dict from remaining columns
            config_dict = {}
            for col, value in zip(config_cols, row[3:]):
                # Try to parse JSON strings back to lists
                if isinstance(value, str):
                    try:
                        config_dict[col] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        config_dict[col] = value
                else:
                    config_dict[col] = value
            
            pending_sessions.append({
                'session_id': session_id,
                'session_path': Path(session_path),
                'total_iterations': total_iterations,
                'config': config_dict
            })
        
        return pending_sessions
    
    def save_fight_result(self, session_id: str, fight_data: Dict[str, Any]) -> None:
        """
        Save a fight result to the database.
        
        Args:
            session_id: Session identifier
            fight_data: Dictionary containing fight result data (from env.report)
        """
        cursor = self.conn.cursor()
        
        # Extract data from fight_data dictionary
        data = {
            'session_id': session_id,
            'env_instance_id': fight_data.get('env', ''),
            'total_ticks': fight_data.get('Total_Ticks', 0),
            'total_steps': fight_data.get('Total_Steps', 0),
            'fight_num': fight_data.get('Fight_Num', 0),
            'result': fight_data.get('result', 2),
            'fight_time': fight_data.get('fight_time', 0.0),
            'iteration': fight_data.get('iteration', 1),
            'action_0': fight_data.get('action_0', 0),
            'action_1': fight_data.get('action_1', 0),
            'action_2': fight_data.get('action_2', 0),
            'action_3': fight_data.get('action_3', 0),
            'action_4': fight_data.get('action_4', 0),
            'action_5': fight_data.get('action_5', 0),
            'action_6': fight_data.get('action_6', 0),
            'action_7': fight_data.get('action_7', 0),
            'damage_0': fight_data.get('damage_0', 0.0),
            'damage_1': fight_data.get('damage_1', 0.0),
            'damage_2': fight_data.get('damage_2', 0.0),
            'damage_3': fight_data.get('damage_3', 0.0),
            'damage_4': fight_data.get('damage_4', 0.0),
            'damage_5': fight_data.get('damage_5', 0.0),
            'damage_6': fight_data.get('damage_6', 0.0),
            'damage_7': fight_data.get('damage_7', 0.0),
            'total_reward': fight_data.get('total_reward', 0.0),
            'damage_taken': fight_data.get('damage_taken', 0.0),
            'rounds_won': fight_data.get('rounds_won', 0),
            'rounds_lost': fight_data.get('rounds_lost', 0),
            'stun_time': fight_data.get('stun_time', 0),
        }
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        values = list(data.values())
        
        query = f"INSERT INTO fight_results ({columns}) VALUES ({placeholders})"
        cursor.execute(query, values)
        self.conn.commit()
    
    def save_training_metrics(self, session_id: str, iteration: int, timesteps_total: int,
                             metrics: Dict[str, Any]) -> None:
        """
        Save training metrics for a specific iteration.
        Uses INSERT OR REPLACE to handle UNIQUE constraint.
        
        Args:
            session_id: Session identifier
            iteration: Training iteration number
            timesteps_total: Total timesteps processed
            metrics: Dictionary containing training metrics (from PPO)
        """
        cursor = self.conn.cursor()
        
        data = {
            'session_id': session_id,
            'iteration': iteration,
            'timesteps_total': timesteps_total,
            'policy_loss': metrics.get('train/policy_loss'),
            'value_loss': metrics.get('train/value_loss'),
            'entropy_loss': metrics.get('train/entropy_loss'),
            'approx_kl': metrics.get('train/approx_kl'),
            'clip_fraction': metrics.get('train/clip_fraction'),
            'explained_variance': metrics.get('train/explained_variance'),
            'learning_rate': metrics.get('train/learning_rate'),
            'n_updates': metrics.get('train/n_updates'),
        }
        
        # Use INSERT OR REPLACE to handle UNIQUE constraint
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        values = list(data.values())
        
        query = f"""
            INSERT OR REPLACE INTO training_metrics ({columns})
            VALUES ({placeholders})
        """
        cursor.execute(query, values)
        self.conn.commit()
    
    def get_session_config(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve ENV_CONFIG parameters for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of configuration parameters
        """
        cursor = self.conn.cursor()
        
        # Get all columns except metadata columns
        cursor.execute("PRAGMA table_info(training_sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        metadata_cols = ['session_id', 'session_path', 'created_at', 'updated_at', 
                        'status', 'total_iterations', 'completed_iterations']
        config_cols = [col for col in columns if col not in metadata_cols]
        
        config_cols_str = ', '.join(config_cols)
        query = f"SELECT {config_cols_str} FROM training_sessions WHERE session_id = ?"
        cursor.execute(query, (session_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        config_dict = {}
        for col, value in zip(config_cols, row):
            # Try to parse JSON strings back to lists
            if isinstance(value, str):
                try:
                    config_dict[col] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    config_dict[col] = value
            else:
                config_dict[col] = value
        
        return config_dict


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to capture PPO training metrics and store them in the database.
    Captures metrics after each training update (after rollout collection and optimization).
    """
    
    def __init__(self, db_session: TrainingDBSession, session_id: str, db_path: str = "training_results.db", verbose: int = 0):
        """
        Initialize the callback.
        
        Args:
            db_session: TrainingDBSession instance for database operations (may be None in subprocesses)
            session_id: Training session identifier
            db_path: Path to database file (used if db_session is None)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.db_session = db_session
        self.session_id = session_id
        self.db_path = db_path
        self.current_iteration = 0
        self.last_log_metrics = {}  # Store last logged metrics
    
    def _on_step(self) -> bool:
        """Called at each step. Returns True to continue training."""
        # Try to capture metrics if available
        self._capture_metrics()
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Save any remaining metrics
        self._save_metrics()
    
    def _on_rollout_end(self) -> bool:
        """
        Called at the end of each rollout (after collecting data).
        Training metrics are typically available after optimization, so we capture them here.
        """
        self._capture_metrics()
        return True
    
    def _capture_metrics(self):
        """Capture metrics from the logger if available."""
        # Try to get metrics from logger
        metrics = {}
        if hasattr(self, 'logger') and self.logger is not None:
            try:
                # stable-baselines3 uses a logger with name_to_value dict
                if hasattr(self.logger, 'name_to_value'):
                    metrics = self.logger.name_to_value.copy()
                    self.last_log_metrics = metrics
            except Exception:
                pass
        
        # If we have metrics, save them
        if metrics:
            self._save_metrics(metrics)
    
    def _save_metrics(self, metrics=None):
        """Save metrics to database."""
        if metrics is None:
            metrics = self.last_log_metrics
        
        if not metrics:
            return
        
        try:
            # Get or create db_session
            db_session = self.db_session
            if db_session is None:
                # Create temporary connection
                temp_session = TrainingDBSession(self.db_path)
                temp_session.__enter__()
                db_session = temp_session
            
            timesteps = self.num_timesteps
            db_session.save_training_metrics(
                self.session_id,
                self.current_iteration,
                timesteps,
                metrics
            )
            
            # If we created a temporary session, close it
            if db_session is not self.db_session:
                db_session.__exit__(None, None, None)
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to save training metrics: {e}")
    
    def set_iteration(self, iteration: int):
        """Set the current training iteration number."""
        self.current_iteration = iteration

