"""
Environment Configuration File
Contains all configurable parameters for the Street Fighter environment.
Modify these values to adjust rewards, punishments, delays, and game settings.
"""

# ============================================================================
# REWARD CONFIGURATION
# ============================================================================

# Round win/loss rewards
ROUND_WIN_TIME_MULTIPLIER = 4.0  # Reward multiplier based on remaining match time when winning round
ROUND_LOSS_TIME_DIVISOR = 2.0    # Penalty divisor based on remaining match time when losing round

# Fight win/loss rewards
FIGHT_WIN_BASE_REWARD = 400     # Base reward for winning a fight (multiplied by fights_won)
FIGHT_LOSS_PENALTY = -50         # Penalty for losing a fight

# Game completion reward
GAME_COMPLETION_REWARD = 4000     # Bonus reward for completing all fights (winning 5 fights total)

# Timeout penalty
TIMEOUT_PENALTY = -15             # Penalty when match times out

# Stun penalty
STUN_PENALTY = -0.1              # Penalty when player is stunned

# Damage rewards/penalties
DAMAGE_DEALT_REWARD_MULTIPLIER = 4.0   # Direct reward for damage dealt (1 HP = 1 reward by default)
DAMAGE_TAKEN_PENALTY_MULTIPLIER = 0.5  # Penalty multiplier for damage taken

# Movement diversity reward
MOVEMENT_DIVERSITY_BASE_REWARD = 25     # Reward per unique movement before an attack

# Repetition penalty
REPETITION_PENALTY_THRESHOLD = 3       # Number of repeated actions before penalty applies
REPETITION_PENALTY_BASE = -2           # Base penalty for repetition (multiplied by (repeated - threshold))


# ============================================================================
# GAME CONFIGURATION
# ============================================================================

# Fight configuration
MAX_FIGHTS_TO_WIN = 3            # Number of fights to win (4 wins = complete the game)
ROUNDS_TO_WIN_FIGHT = 2          # Number of rounds needed to win a fight
MAX_FIGHTS_TO_LOSE = 3           # Maximum fights to lose before truncation

# Match timing
MATCH_TIME_LIMIT = 90            # Maximum match time in seconds
MATCH_TIME_THRESHOLD = 1         # Time threshold for timeout check (timeout if < threshold)


# ============================================================================
# ACTION DELAYS (in ticks)
# ============================================================================

# Move-specific timing delays
MOVE_DELAY_UP = 2                # Up movement delay
MOVE_DELAY_DOWN = 2              # Down movement delay
MOVE_DELAY_LEFT = 2              # Left movement delay
MOVE_DELAY_RIGHT = 2             # Right movement delay
MOVE_DELAY_PUNCH = 2             # Punch (Button A) delay
MOVE_DELAY_KICK = 2              # Kick (Button B) delay
MOVE_DELAY_HURRICANE_KICK = 12   # Hurricane Kick special move delay
MOVE_DELAY_HADOUKEN = 10         # Hadouken special move delay

# Default delay (used for unknown actions)
DEFAULT_MOVE_DELAY = 3           # Default delay for unmapped actions

# Observation and release timing
RELEASE_DURATION = 1             # Minimal release time before observation
OBSERVATION_TICKS_AFTER_RELEASE = 3  # Number of ticks to wait after release before observation
SPECIAL_MOVE_EXTRA_TICKS = 2     # Extra ticks for special moves to capture animation


# ============================================================================
# EMULATION SETTINGS
# ============================================================================

EMULATION_SPEED = 0              # Emulation speed multiplier (0 = unlimited speed, 16 = 16x speed for faster training)
                                  # At 16x speed, each iteration takes ~30 seconds instead of ~8 minutes


# ============================================================================
# HISTORY BUFFER SIZES
# ============================================================================

STATUS_HISTORY_SIZE = 3          # Size of status (stun) history buffer
ACTION_COMBO4_HISTORY_SIZE = 10  # Size of 4-action combo history buffer
ACTION_COMBO3_HISTORY_SIZE = 3   # Size of 3-action combo history buffer
ENERGY_HISTORY_SIZE = 2          # Size of energy history buffer
PAST_ACTION_HISTORY_SIZE = 2     # Size of past action history buffer
ACTION_HISTORY_SIZE = 4          # Size of general action history buffer (for repetition penalty)
RESET_STATUS_HISTORY_SIZE = 5    # Status history size after reset


# ============================================================================
# COMBO VERIFICATION
# ============================================================================

# Hadouken combo sequences (orientation 0 = facing right, 1 = facing left)
HADOUKEN_ORIENTATION_0 = [3, 2, 5]      # Right, Left, Kick
HADOUKEN_ORIENTATION_1 = [2, 3, 5]      # Left, Right, Kick

# Hurricane Kick combo sequences
HURRICANE_KICK_ORIENTATION_0 = [1, 2, 4, 3]  # Down, Left, Punch, Right
HURRICANE_KICK_ORIENTATION_1 = [1, 3, 4, 2]  # Down, Right, Punch, Left

# Status check for combo execution (must be [0,0,0] - not stunned)
COMBO_REQUIRED_STATUS = [0, 0, 0]


# ============================================================================
# MOVEMENT DIVERSITY
# ============================================================================

# Action categories for movement diversity reward
MOVEMENT_KEYS = [0, 1, 2, 3]     # Up, Down, Left, Right
ATTACK_KEYS = [4, 5]             # Punch (A), Kick (B)
MIN_ACTION_HISTORY_FOR_DIVERSITY = 4  # Minimum action history length to check diversity


# ============================================================================
# FILE PATHS
# ============================================================================

GAME_ROM_FILE = "SFA.gbc"        # Game ROM file path
SAVE_STATE_FILE = "ryu_arcade.sav"  # Save state file path (used for reset)
REPORTS_DIRECTORY = "./reports"  # Directory for saving reports


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Environment setup
NUM_ENVS = 2                      # Number of parallel environments (fixed at 2 for ~16GB RAM usage)
                                  # Each env uses ~500MB-1GB per subprocess
FRAME_STACK_SIZE = 4              # Number of frames to stack for observation
                                  # Memory impact: 4 frames × 67.5 KB = 270 KB per observation

# Training hyperparameters
# Optimized for 30 minutes training time with 2 envs and ~16GB RAM
EP_LENGTH = 8192                  # Steps per environment per update (increased for more memory usage)
                                  # Collected samples = EP_LENGTH × NUM_ENVS = 8192 × 2 = 16,384
                                  # Memory: 8192 × 270 KB × 2 ≈ 4.4 GB observations + 4.4 GB buffer = ~9 GB
TOTAL_TIMESTEPS = 491_520         # Total timesteps for training (optimized for ~30 minutes)
                                  # Calculation: 30 iterations × 16,384 timesteps = 491,520
                                  # With 16x speed and 2 envs: ~60 seconds per iteration = 30 minutes total
                                  # Each iteration collects: EP_LENGTH × NUM_ENVS = 8192 × 2 = 16,384 timesteps
BATCH_SIZE = 4096                 # Batch size for PPO updates (increased for more memory usage)
                                  # With 16,384 samples: 16,384 / 4096 = 4 batches per epoch
                                  # Memory impact: Larger batches use more VRAM/RAM during training
                                  # Total: ~2-3 GB additional during training
N_EPOCHS = 3                      # Number of optimization epochs per update (reduced for faster training)
                                  # Total batch updates per PPO step = N_EPOCHS × (collected_samples / BATCH_SIZE)
                                  # 3 epochs × 4 batches = 12 updates per iteration (faster than 4 epochs)

# Checkpoint and evaluation frequency
SAVE_FREQ_MULTIPLIER = 10        # Save frequency = EP_LENGTH * SAVE_FREQ_MULTIPLIER = 81,920 timesteps
EVAL_FREQ_MULTIPLIER = 10        # Evaluation frequency = EP_LENGTH * EVAL_FREQ_MULTIPLIER = 81,920 timesteps

# Callback configuration
CHECKPOINT_NAME_PREFIX = "street_fighter"  # Prefix for checkpoint filenames
SAVE_REPLAY_BUFFER = False        # Whether to save replay buffer in checkpoints
                                  # WARNING: Setting True adds ~4-5 GB memory overhead
                                  # Only enable if you need to resume training from exact buffer state
SAVE_VECNORMALIZE = True          # Whether to save VecNormalize stats
EVAL_DETERMINISTIC = True         # Use deterministic policy for evaluation
EVAL_RENDER = False               # Render during evaluation

# Training loop configuration
RESET_NUM_TIMESTEPS = False       # Whether to reset timestep counter each iteration
TB_LOG_NAME = "PPO"               # TensorBoard log name


# ============================================================================
# MEMORY ADDRESSES (if these need to be configurable)
# ============================================================================

# Memory addresses for game state (commented out - may not need to be configurable)
# MEMORY_ADDRESS_STUNNED = 50219
# MEMORY_ADDRESS_ORIENTATION = 50866
# MEMORY_ADDRESS_ENEMY_HP = 50869
# MEMORY_ADDRESS_PLAYER_HP = 50357
# MEMORY_ADDRESS_ENERGY = 50295
# MEMORY_ADDRESS_MATCH_TIME = 53001

