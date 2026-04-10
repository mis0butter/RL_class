# In this homework, you will implement two reinforcement learning algorithms with linear function
# approximation using the Atari Breakout environment from the Arcade Learning Environment
# (ALE). The environment can be found at: https://ale.farama.org/getting-started/.
# You will work with linear function approximation for both methods. Assume that the state
# is represented by suitable features extracted from the game screen (e.g., raw pixels flattened, tile
# coding, or other feature engineering techniques). Sections 9.5 and 13.7 in this book present some
# ideas of choosing feature vectors: Reinforcement Learning: An Introduction
# http://incompleteideas.net/book/RLbook2020.pdf

# ========================================================= 
# Task 1: Q-Learning with Linear Function Approximation [50 points]
# ========================================================= 

# ----------------------------- 
# Objective: Implement the Q-learning algorithm to learn an approximate action-value function
# ----------------------------- 

# Q(s, a; θ) using a linear function approximation.
# • Represent the Q-function as:
#       Q(s, a; θ) = ϕ(s, a)^T θ
# where ϕ(s, a) is the feature vector corresponding to state-action pair (s, a) and θ is the weight
# vector.
# • Implement the Q-learning update:
#       θ_k+1 = θ_k + α(r_k + γ max_a' Q(s'_k, a'; θ_k) - Q(s_k, a_k; θ_k)) ∇_θ Q(s_k, a_k; θ_k)
# • Train your agent for multiple episodes and report the average reward over time.

# ----------------------------- 
# Deliverables:
# ----------------------------- 

# 1. Submit your code.
# 2. Plot showing learning progress (average reward per episode).
# 3. Explain the choice of feature representation and behavior policies for data collection. 
#    Implement the outcome of your learning and report the results.

# ========================================================= 
# import packages 
# ========================================================= 

import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt 

import ale_py 
gym.register_envs(ale_py) 

rng = np.random.default_rng(0) 

# ========================================================= 
# create breakout environment (RAM observations!) 
# ========================================================= 

# KEY CHANGE: obs_type='ram' gives us the 128-byte Atari 2600 RAM 
# instead of the 210x160x3 pixel image. This lets us extract 
# semantic features directly.
env = gym.make('ALE/Breakout-v5', obs_type='ram')  

# get observation space and action space 
N_states  = env.observation_space.shape       # states: (128,) RAM bytes  
N_actions = env.action_space.n                # 4 actions: NOOP, FIRE, RIGHT, LEFT     

print('observation space: \n', N_states) 
print('action space: \n', N_actions) 

# ========================================================= 
# function approximation: TILE CODING over RAM features  
# ========================================================= 
# 
# RAM address mapping for Breakout (Atari 2600): 
# (Verified empirically by probing RAM during gameplay — see probe_ram.py)
#   - RAM[101] = ball X position 
#   - RAM[99]  = ball Y position 
#   - RAM[72]  = paddle X position 
# 
# Strategy: extract semantic features from RAM, then apply TILE CODING 
# to create binary feature vectors. Tile coding maps continuous values 
# into overlapping binary indicators, allowing a linear model to 
# represent nonlinear decision boundaries. 
# 
# We tile code over these key variable groups:
#   1. (paddle_ball_dx, ball_y) — 2D tiling, 8 tilings × 10×10 bins = 800 
#      This is the CORE feature: "where is the ball relative to paddle?"
#   2. (ball_dx) — 1D tiling, 4 tilings × 6 bins = 24
#   3. (ball_dy) — 1D tiling, 4 tilings × 6 bins = 24 
#   4. (paddle_x) — 1D tiling, 4 tilings × 8 bins = 32 
#   5. bias — 1 feature 
# 
# Total: 881 binary features (still small, but MUCH more expressive 
# than 7 raw features for a linear model)
# ========================================================= 

def extract_raw_features(ram, prev_ram=None):
    """Extract raw continuous features from RAM.
    
    Returns dict of named features, each normalized to [0, 1].
    """
    r = ram.astype(np.int16)
    
    ball_x   = r[101] / 207.0  
    ball_y   = r[99]  / 199.0  
    paddle_x = r[72]  / 191.0  
    
    # paddle - ball offset, shifted to [0, 1] range 
    paddle_ball_dx = (r[72] - r[101] + 191) / 382.0 
    
    if prev_ram is not None:
        pr = prev_ram.astype(np.int16)
        ball_dx = ((r[101] - pr[101]) + 20) / 40.0  # shift to [0,1]
        ball_dy = ((r[99]  - pr[99])  + 20) / 40.0  
    else:
        ball_dx = 0.5  # centered = no velocity 
        ball_dy = 0.5 
    
    return {
        'ball_x': np.clip(ball_x, 0, 1),
        'ball_y': np.clip(ball_y, 0, 1),
        'paddle_x': np.clip(paddle_x, 0, 1),
        'paddle_ball_dx': np.clip(paddle_ball_dx, 0, 1),
        'ball_dx': np.clip(ball_dx, 0, 1),
        'ball_dy': np.clip(ball_dy, 0, 1),
    }

# ----------------------------- 
# Tile coding implementation 
# ----------------------------- 

class TileCoder:
    """Simple tile coding for 1D and 2D continuous variables.
    
    Each tiling partitions the [0,1]^d space into bins. Multiple 
    tilings are offset from each other so that nearby points share 
    some (but not all) active tiles — this is what gives tile coding 
    its generalization power.
    """
    
    def __init__(self):
        # Define tiling groups: (name, dimensions, n_tilings, bins_per_dim)
        self.groups = [
            # 2D: paddle-ball alignment × ball height — the CORE feature
            ('pdx_by', ['paddle_ball_dx', 'ball_y'], 8, [10, 10]),
            # 1D: ball velocities  
            ('bdx',    ['ball_dx'],                  4, [6]),
            ('bdy',    ['ball_dy'],                  4, [6]),
            # 1D: paddle position  
            ('px',     ['paddle_x'],                 4, [8]),
        ]
        
        # compute total number of features and offsets 
        self.n_features = 1  # start with 1 for bias
        self.group_offsets = []
        for name, dims, n_tilings, bins_per_dim in self.groups:
            n_tiles_per_tiling = 1
            for b in bins_per_dim:
                n_tiles_per_tiling *= b 
            offset = self.n_features 
            self.group_offsets.append(offset)
            self.n_features += n_tilings * n_tiles_per_tiling 
        
        # pre-generate random offsets for each tiling 
        rng_tc = np.random.default_rng(42)
        self.offsets = []
        for name, dims, n_tilings, bins_per_dim in self.groups:
            # offset each tiling by a random fraction of a bin width 
            group_offsets = []
            for t in range(n_tilings):
                off = [rng_tc.uniform(0, 1.0/b) for b in bins_per_dim]
                group_offsets.append(off)
            self.offsets.append(group_offsets)
    
    # ----------------------------- 

    def encode(self, features_dict):
        """Convert continuous features to sparse binary tile-coded vector.
        
        Returns np.array of shape (n_features,) with mostly 0s and 
        a few 1s (one per tiling per group).
        """
        phi = np.zeros(self.n_features)
        phi[0] = 1.0  # bias 
        
        for g, (name, dims, n_tilings, bins_per_dim) in enumerate(self.groups):
            # get the continuous values for this group 
            vals = [features_dict[d] for d in dims]
            
            n_tiles_per_tiling = 1 
            for b in bins_per_dim:
                n_tiles_per_tiling *= b 
            
            for t in range(n_tilings):
                # apply offset for this tiling 
                shifted = [v + self.offsets[g][t][i] for i, v in enumerate(vals)]
                
                # compute bin indices 
                bin_idx = []
                for i, (v, b) in enumerate(zip(shifted, bins_per_dim)):
                    idx = int(v * b)
                    idx = max(0, min(b - 1, idx))  # clamp 
                    bin_idx.append(idx)
                
                # flatten multi-dim bin index 
                flat_idx = 0 
                for i, idx in enumerate(bin_idx):
                    stride = 1 
                    for j in range(i + 1, len(bin_idx)):
                        stride *= bins_per_dim[j]
                    flat_idx += idx * stride 
                
                # set the feature 
                feature_idx = self.group_offsets[g] + t * n_tiles_per_tiling + flat_idx 
                phi[feature_idx] = 1.0 
        
        return phi 

tile_coder = TileCoder()
N_features = tile_coder.n_features 

# ----------------------------- 
# get state from tile coding 
# ----------------------------- 

def get_state(ram, prev_ram=None):
    """Extract tile-coded features from RAM.
    
    Returns sparse binary vector of shape (N_features,).
    """
    raw = extract_raw_features(ram, prev_ram)
    return tile_coder.encode(raw)

# ----------------------------- 

print(f"\nTile coder: {N_features} features")
for g, (name, dims, n_tilings, bins_per_dim) in enumerate(tile_coder.groups):
    n_per = 1
    for b in bins_per_dim:
        n_per *= b
    print(f"  {name}: {dims} -> {n_tilings} tilings × {n_per} tiles = {n_tilings * n_per}")
print(f"  + 1 bias")

# ========================================================= 
# test environment and features 
# ========================================================= 

# reset env and see initial observation 
obs, info = env.reset() 
print(f"\nRAM observation shape: {obs.shape}")
print(f"Ball X (RAM[101]): {obs[101]}, Ball Y (RAM[99]): {obs[99]}, Paddle X (RAM[72]): {obs[72]}")

# run a few random steps to verify RAM addresses change sensibly 
obs, _, _, _, _ = env.step(1)  # FIRE to launch ball 
num_steps = 50 
for step in range(num_steps): 
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action) 
    if terminated or truncated: 
        obs, info = env.reset() 
        obs, _, _, _, _ = env.step(1)  # FIRE again 

print(f"After {num_steps} steps — Ball X: {obs[101]}, Ball Y: {obs[99]}, Paddle X: {obs[72]}")

env.close() 

# ========================================================= 
# Q-learning set up!!! 
# ========================================================= 
# Algorithm: 
#  
# Initialize Q(s,a), for all s in S and a in A, arbitrarily except Q(terminal, a) = 0 for all a 
# Loop for each episode: 
#   Initialize S 
#   Loop for each step of episode: 
#       Choose A from S using policy derived from Q (epsilon-greedy) 
#       Take action A, observe R and S' 
#       Q(S, A) <-- Q(S, A) + α[ R + γ max_a Q(S', a) - Q(S,A) ]
#       S <-- S' 
#   until S is terminal 
# ========================================================= 

# re-open env for training 
env = gym.make('ALE/Breakout-v5', obs_type='ram')

# initialize: take one step to get two consecutive RAM frames 
ram1, _ = env.reset() 
ram2, _, _, _, _ = env.step(1)  # action 1 = FIRE to launch ball 

# initialize weight matrix!! shape: [N_actions, N_features]
theta = np.zeros((N_actions, N_features))

# parameters 
# alpha is divided by the number of active tiles per step (~21) 
# so that the effective step size per feature is reasonable 
epsilon0 = 1.0       # initial exploration rate  
alpha    = 0.1 / 21  # step size (normalized by ~active tiles)
gamma    = 0.99      # discount factor 

ep_reward_hist = [] 
reward_ep   = 0 
max_iter    = 500000 

print(f"\nTraining Q-learning with RAM features...")
print(f"  Feature dimension: {N_features}")
print(f"  Theta shape: {theta.shape}")
print(f"  Max iterations: {max_iter}")

# ========================================================= 
# Q-learning loop!!! 
# ========================================================= 

for k in range(max_iter): 
    
    # decay epsilon 
    epsilon = max(0.05, epsilon0 / (1.0 + 0.0001 * k))

    # state feature representation from RAM  
    state = get_state(ram2, ram1)

    # compute all 4 q_vals using matrix dot product 
    # Q(s, a; θ) = θ[a] · ϕ(s)  for each action a 
    q_vals = np.dot(theta, state) 

    # Choose A from S using policy derived from Q (epsilon-greedy) 
    if rng.random() < epsilon: 
        action = env.action_space.sample() 
    else: 
        action = np.argmax(q_vals) 

    # Take action A, observe R and transition to new RAM state 
    ram3, reward, terminated, truncated, info = env.step(action) 

    # clip reward to [-1, 1] for stability 
    reward_clipped = np.clip(reward, -1.0, 1.0)

    # track raw reward 
    reward_ep += reward 

    # ----------------------------- 
    # Q(S, A) <-- Q(S, A) + α[ R + γ max_a Q(S', a) - Q(S,A) ]
    # ----------------------------- 

    # compute td target based on if ep terminated or not 
    if terminated or truncated: 
        td_target = reward_clipped 
    else: 
        # R + γ max_a Q(S', a) 
        next_state  = get_state(ram3, ram2)
        next_q_vals = np.dot(theta, next_state)
        td_target   = reward_clipped + gamma * np.max(next_q_vals) 

    # TD error: R + γ max_a Q(S', a) - Q(S, A) 
    td_err = td_target - q_vals[action]  
    
    # update ONLY the weights for the action we took 
    # ∇_θ Q(s, a; θ) = ϕ(s) for the linear case 
    theta[action, :] += alpha * td_err * state 

    # ----------------------------- 
    # advance to next state or reset episode 
    # ----------------------------- 
    
    if terminated or truncated: 

        # save episode total reward 
        ep_reward_hist.append(reward_ep) 

        # print progress every 50 episodes 
        if len(ep_reward_hist) % 50 == 0:
            last50 = ep_reward_hist[-50:]
            print(f"  Episode {len(ep_reward_hist):4d} | "
                  f"Last 50 avg: {np.mean(last50):6.2f} | "
                  f"Max: {np.max(last50):5.1f} | "
                  f"Epsilon: {epsilon:.3f}")

        # reset episode 
        reward_ep = 0 
        ram1, _ = env.reset()
        ram2, _, _, _, _ = env.step(1)  # FIRE to launch ball 

    else: 
        ram1 = ram2 
        ram2 = ram3 

env.close()

# ========================================================= 
# plot rewards 
# ========================================================= 

ep_reward_hist = np.array(ep_reward_hist) 

# compute rolling average for smoother plot 
window = 50 
if len(ep_reward_hist) >= window:
    rolling_avg = np.convolve(ep_reward_hist, np.ones(window)/window, mode='valid')
else:
    rolling_avg = ep_reward_hist

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ----------------------------- 
# left: raw episode rewards 
# ----------------------------- 

axes[0].plot(ep_reward_hist, alpha=0.3, color='steelblue', label='Episode reward')
if len(ep_reward_hist) >= window:
    axes[0].plot(np.arange(window-1, len(ep_reward_hist)), rolling_avg, 
                 color='darkblue', linewidth=2, label=f'{window}-episode avg')
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].set_title("Q-learning with RAM Features: Atari Breakout")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ----------------------------- 
# right: log scale  
# ----------------------------- 

axes[1].semilogy(np.arange(len(ep_reward_hist)), np.maximum(ep_reward_hist, 0.5), 
                 alpha=0.3, color='steelblue')
if len(ep_reward_hist) >= window:
    axes[1].semilogy(np.arange(window-1, len(ep_reward_hist)), 
                     np.maximum(rolling_avg, 0.5),
                     color='darkblue', linewidth=2)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Reward (log scale)")
axes[1].set_title("Log Scale")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hw4/Q_learning_breakout_ram.png", dpi=300)
print(f"\nPlot saved to hw4/Q_learning_breakout_ram.png")
print(f"Total episodes: {len(ep_reward_hist)}")
print(f"Final {window}-episode average reward: {np.mean(ep_reward_hist[-window:]):.2f}")

# ----------------------------- 
# print learned weight summary (tile-coded weights are too many to print individually)
# ----------------------------- 

print("\n--- Learned weight summary (θ) ---")
action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
print(f"{'Action':>10s} {'||θ||':>8s} {'mean':>8s} {'min':>8s} {'max':>8s} {'nonzero':>8s}")
for a in range(N_actions):
    w = theta[a]
    print(f"{action_names[a]:>10s} {np.linalg.norm(w):8.4f} {w.mean():8.4f} "
          f"{w.min():8.4f} {w.max():8.4f} {np.sum(np.abs(w) > 1e-6):8d}")
