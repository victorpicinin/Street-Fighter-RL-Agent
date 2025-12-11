# Memory & Training Configuration Analysis
**Hardware:** 32GB RAM, 16GB VRAM

## Current Configuration Analysis

### Current Values:
- `NUM_ENVS = 2`
- `EP_LENGTH = 8192`
- `FRAME_STACK_SIZE = 4`
- `BATCH_SIZE = 2048`
- `N_EPOCHS = 10`
- `TOTAL_TIMESTEPS = 10_000_000`
- `SAVE_REPLAY_BUFFER = True`

---

## Memory Calculations

### 1. Observation Memory (RAM)
- **Observation shape:** (144, 160, 3) = 69,120 bytes ≈ 67.5 KB per frame
- **With frame stacking (4 frames):** 4 × 67.5 KB = **270 KB per observation**
- **Per environment buffer (EP_LENGTH steps):** 8192 × 270 KB = **~2.2 GB per env**
- **With NUM_ENVS = 2:** 2 × 2.2 GB = **~4.4 GB RAM** for observations ⚠️

### 2. PPO Replay Buffer Memory (RAM)
- **Buffer size:** EP_LENGTH × NUM_ENVS = 8192 × 2 = 16,384 steps
- **Per step stores:** obs (270 KB), action (4B), reward (4B), value (4B), log_prob (4B)
- **Total buffer:** 16,384 × 270 KB ≈ **~4.4 GB RAM** ⚠️

### 3. Environment Subprocess Memory (RAM)
- **Each PyBoy instance:** ~50-100 MB
- **Python interpreter per subprocess:** ~200-500 MB
- **Per subprocess total:** ~500 MB
- **NUM_ENVS = 2:** 2 × 500 MB = **~1 GB RAM**

### 4. VRAM Usage (GPU)
- **Model weights (CNN):** ~5-10 MB
- **Batch size 2048:** Each batch = 2048 × 270 KB = **~540 MB per batch**
- **With gradients:** ~1-1.5 GB per batch
- **During training:** Model + batch + gradients = **~2-3 GB VRAM** ✅

### 5. Python/PyTorch Overhead
- **PyTorch base:** ~1-2 GB RAM
- **Python overhead:** ~500 MB RAM

---

## Total Memory Estimate

### RAM Usage:
- Observations: **4.4 GB**
- PPO Buffer: **4.4 GB** (if SAVE_REPLAY_BUFFER=True, may persist longer)
- Environment subprocesses: **1 GB**
- Python/PyTorch: **1.5 GB**
- **TOTAL: ~11.3 GB RAM** ✅ (Well within 32GB)

### VRAM Usage:
- Model + Batch + Gradients: **~2-3 GB VRAM** ✅ (Well within 16GB)

---

## Issues Found

### ⚠️ CRITICAL: BATCH_SIZE > BUFFER SIZE
**Problem:**
- `BATCH_SIZE = 2048`
- `EP_LENGTH = 8192` → Total collected samples = 8192 × 2 = 16,384
- **Batches per update:** 16,384 / 2048 = **8 batches**
- **Samples per epoch:** 16,384
- **With 10 epochs:** 10 × 8 = **80 batch updates per PPO update**

**Impact:** This is actually fine! With 16,384 samples, you get 8 batches per epoch.

### ⚠️ WARNING: EP_LENGTH Mismatch with Episode Length
**Problem:**
- `EP_LENGTH = 8192` steps
- Typical fighting game episode: ~100-500 steps (one fight)
- **8192 steps ≈ 16-80 fights per episode**

**Impact:** Very long episodes. May need adjustment based on actual episode length.

### ⚠️ WARNING: NUM_ENVS Too Low for Hardware
**Problem:**
- Only using 2 parallel environments
- With 32GB RAM, could easily handle 4-8 environments
- Current usage: ~11.3 GB / 32 GB = **35% RAM utilization**
- **Underutilizing hardware!**

**Recommendation:** Increase to 4-6 environments for better parallelization.

### ⚠️ WARNING: SAVE_REPLAY_BUFFER Memory Impact
**Problem:**
- `SAVE_REPLAY_BUFFER = True` keeps buffer in memory longer
- May prevent garbage collection
- Adds ~4.4 GB memory overhead

**Recommendation:** Set to False unless you need it for debugging.

### ⚠️ POTENTIAL: Frame Stacking Memory
**Problem:**
- 4 frames × 270 KB = large observations
- Each environment stores EP_LENGTH × 270 KB

**Impact:** Acceptable but could be reduced if needed.

---

## Logic Checks

### ✅ Batch Size vs Collected Samples
- **Collected:** EP_LENGTH × NUM_ENVS = 8192 × 2 = 16,384 samples
- **Batch size:** 2048
- **Batches per epoch:** 16,384 / 2048 = 8 batches ✅
- **Epochs:** 10
- **Total updates per PPO step:** 10 × 8 = 80 batch updates ✅

### ✅ Training Loop Logic
- **Timesteps per iteration:** EP_LENGTH × NUM_ENVS = 16,384
- **Total iterations:** 10,000,000 / 16,384 ≈ 610 iterations ✅
- **Reset happens correctly:** `env.reset()` called each iteration ✅

### ⚠️ POTENTIAL ISSUE: Evaluation Callback
- `EvalCallback` is created but **NOT added to callback_list**!
- Line 116: `callback_list = CallbackList([checkpoint_callback])`
- **Missing:** `eval_callback`

### ✅ Checkpoint Frequency
- `SAVE_FREQ = EP_LENGTH × 10 = 81,920 steps`
- Saves every ~5 iterations ✅

---

## Recommendations

### Priority 1: Fix Critical Issues
1. **Add EvalCallback to callback list** (currently missing!)
2. **Increase NUM_ENVS to 4-6** (better hardware utilization)
3. **Set SAVE_REPLAY_BUFFER = False** (unless needed)

### Priority 2: Optimize Configuration
4. **Reduce EP_LENGTH** to match actual episode length (~1000-2000)
5. **Consider reducing BATCH_SIZE** to 1024 if memory issues occur
6. **Monitor actual memory usage** during training

### Priority 3: Fine-tuning
7. **Adjust N_EPOCHS** based on training stability (10 may be high)
8. **Consider gradient accumulation** if batch size is too small

