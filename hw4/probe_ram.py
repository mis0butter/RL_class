"""Probe Breakout RAM to find correct addresses for ball/paddle positions.

Strategy: 
  1. Play game with deliberate actions (LEFT, RIGHT) and observe which 
     RAM bytes correlate with paddle movement.
  2. Launch the ball (FIRE) and observe which bytes change rapidly 
     (those are likely ball coordinates).
"""
import gymnasium as gym
import numpy as np
import ale_py
gym.register_envs(ale_py)

# Use BOTH ram and rgb observations so we can cross-reference
env = gym.make('ALE/Breakout-v5', obs_type='ram')

ram, info = env.reset()

# Step 1: Press FIRE to start the game and launch the ball
print("=== Pressing FIRE to launch ball ===")
ram_before_fire = ram.copy()
ram, _, _, _, _ = env.step(1)  # FIRE
ram_after_fire = ram.copy()

# Show which bytes changed after FIRE
changed = np.where(ram_before_fire != ram_after_fire)[0]
print(f"RAM bytes that changed after FIRE: {changed}")
for idx in changed:
    print(f"  RAM[{idx}]: {ram_before_fire[idx]} -> {ram_after_fire[idx]}")

# Step 2: Move RIGHT for 30 steps and track which bytes change monotonically
print("\n=== Moving RIGHT for 30 steps ===")
ram_history = [ram.copy()]
for _ in range(30):
    ram, _, term, trunc, _ = env.step(2)  # RIGHT
    ram_history.append(ram.copy())
    if term or trunc:
        ram, _ = env.reset()
        ram, _, _, _, _ = env.step(1)

ram_stack = np.array(ram_history)  # shape: (31, 128)

# Find bytes that changed and show their trajectories
print("\nRAM bytes that changed during RIGHT movement:")
for idx in range(128):
    vals = ram_stack[:, idx]
    if vals.max() - vals.min() > 2:  # changed meaningfully
        # Check if monotonically increasing (paddle going right?)
        diffs = np.diff(vals.astype(np.int16))
        direction = "INCREASING" if np.sum(diffs > 0) > 20 else \
                    "DECREASING" if np.sum(diffs < 0) > 20 else "oscillating"
        print(f"  RAM[{idx:3d}]: min={vals.min():3d} max={vals.max():3d} "
              f"range={vals.max()-vals.min():3d} {direction} "
              f"first5={list(vals[:5])} last5={list(vals[-5:])}")

# Step 3: Now move LEFT for 30 steps
print("\n=== Moving LEFT for 30 steps ===")
ram_history_left = [ram.copy()]
for _ in range(30):
    ram, _, term, trunc, _ = env.step(3)  # LEFT
    ram_history_left.append(ram.copy())
    if term or trunc:
        ram, _ = env.reset()
        ram, _, _, _, _ = env.step(1)

ram_stack_left = np.array(ram_history_left)

print("\nRAM bytes that changed during LEFT movement:")
for idx in range(128):
    vals = ram_stack_left[:, idx]
    if vals.max() - vals.min() > 2:
        diffs = np.diff(vals.astype(np.int16))
        direction = "INCREASING" if np.sum(diffs > 0) > 20 else \
                    "DECREASING" if np.sum(diffs < 0) > 20 else "oscillating"
        print(f"  RAM[{idx:3d}]: min={vals.min():3d} max={vals.max():3d} "
              f"range={vals.max()-vals.min():3d} {direction} "
              f"first5={list(vals[:5])} last5={list(vals[-5:])}")

# Step 4: Play a longer game with NOOP to watch ball bounce
print("\n=== Playing 200 steps with NOOP (watching ball) ===")
env2 = gym.make('ALE/Breakout-v5', obs_type='ram')
ram, _ = env2.reset()
ram, _, _, _, _ = env2.step(1)  # FIRE

ram_history_play = [ram.copy()]
for _ in range(200):
    ram, _, term, trunc, _ = env2.step(0)  # NOOP
    ram_history_play.append(ram.copy())
    if term or trunc:
        break

ram_stack_play = np.array(ram_history_play)

print("\nRAM bytes that oscillate during gameplay (likely ball coords):")
for idx in range(128):
    vals = ram_stack_play[:, idx]
    if vals.max() - vals.min() > 5:
        # Count direction changes (oscillation = ball bouncing)
        diffs = np.diff(vals.astype(np.int16))
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) > 5:
            sign_changes = np.sum(np.diff(np.sign(nonzero_diffs)) != 0)
        else:
            sign_changes = 0
        print(f"  RAM[{idx:3d}]: min={vals.min():3d} max={vals.max():3d} "
              f"range={vals.max()-vals.min():3d} sign_changes={sign_changes:2d} "
              f"first10={list(vals[:10])}")

env.close()
env2.close()
