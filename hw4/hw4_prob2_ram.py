# In this homework, you will implement two reinforcement learning algorithms with linear function
# approximation using the Atari Breakout environment from the Arcade Learning Environment
# (ALE). The environment can be found at: https://ale.farama.org/getting-started/.
# You will work with linear function approximation for both methods. Assume that the state
# is represented by suitable features extracted from the game screen (e.g., raw pixels flattened, tile
# coding, or other feature engineering techniques). Sections 9.5 and 13.7 in this book present some
# ideas of choosing feature vectors: Reinforcement Learning: An Introduction
# http://incompleteideas.net/book/RLbook2020.pdf

# ========================================================= 
# Task 2: Policy Gradient with Linear Function Approximation [50 points]
# ========================================================= 

# ----------------------------- 
# Objective:  Implement a policy gradient method to learn a parameterized stochastic policy π(a|s; θ) with linear function approximation.
# ----------------------------- 

# Represent the policy as a softmax over linear features: 
#                 exp(ϕ(s, a)^T θ)
#   π(a|s; θ) = --------------------
#               ∑_b exp(ϕ(s, b)^T θ)

# Implement the update rule: 
#   θ_k+1 ← θ_k + α ∇_θ f(π_k)

# Train your agent for multiple episodes and report the average reward over time 

# ----------------------------- 
# Deliverables:
# ----------------------------- 

# 1. Submit your code.
# 2. Plot showing learning progress (average reward per episode).
# 3. Explain the choice of feature representation and how do you estimate the gradients. Implement the outcome of your learning and report the results.

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
# Policy gradient set up!!! 
# ========================================================= 
# Algorithm: one-step actor-critic (episodic) 
#  
# Input: a differentiable policy parameterization π(a|s, θ) 
# Input: a differentiable state-value function parameterization vhat(s,w) 
# Parameters: step sizes α^θ > 0, α^w > 0 
# Initialize policy parameter θ in R^d' and state-value weights w in R^d 
# Loop forever (for each episode): 
#   Initialize S (first state of episode) 
#   I <-- 1 
#   Loop while S is not terminal (for each time step): 
#       A ~ π(.|S, θ) 
#       Take action A, observe S', R 
#       delta <-- R + gamma vhat(S', w) - vhat(S, w) 
#       w <-- w + alpha^w delta grad vhat(S, w) 
#       theta <-- theta + alpha^theta I delta grad ln pi(A|S, theta) 
#       I <-- gamma I 
#       S <-- S' 
# ========================================================= 
# Represent the policy as a softmax over linear features: 
#                 exp(ϕ(s, a)^T θ)
#   π(a|s; θ) = --------------------
#               ∑_b exp(ϕ(s, b)^T θ)

# Implement the update rule: 
#   θ_k+1 ← θ_k + α ∇_θ f(π_k)
# ========================================================= 

# re-open env for training 
env = gym.make('ALE/Breakout-v5', obs_type='ram')

# initialize: take one step to get two consecutive RAM frames 
ram1, _ = env.reset() 
ram2, _, _, _, _ = env.step(1)  # action 1 = FIRE to launch ball 

# ----------------------------- 
# Initialize  
# ----------------------------- 
# Input: differentiable policy π(a|s, θ) — softmax linear policy
# Input: a differentiable state-value function parameterization vhat(s,w)
# Parameters: step sizes α^θ > 0, α^w > 0 
# Initialize policy parameter θ in R^(N_actions × N_features) 
# Initialize state-value weights w in R^N_features 
# ----------------------------- 

# policy parameters: θ[a, :] gives weights for action a 
theta = np.zeros((N_actions, N_features))

# state-value (critic/baseline) weights: vhat(s, w) = w^T ϕ(s)
w = np.zeros(N_features)

# step sizes (divided by ~21 active tiles per step) 
alpha_theta = 0.01 / 21   # actor step size  
alpha_w     = 0.1  / 21   # critic step size (larger — critic should learn faster)
gamma       = 0.99         # discount factor 

ep_reward_hist = [] 
reward_ep   = 0 
max_iter    = 500000 

# discount factor accumulator — resets to 1 each episode 
I = 1.0 

print(f"\nTraining one-step actor-critic with RAM features...")
print(f"  Feature dimension: {N_features}")
print(f"  Theta shape: {theta.shape}")
print(f"  w shape:     {w.shape}")
print(f"  Max iterations: {max_iter}")

# ========================================================= 
# helper: softmax policy π(a|s, θ)
# ========================================================= 

def softmax_policy(state, theta):
    """Compute π(a|s; θ) for all actions using softmax over linear features.
    
        π(a|s; θ) = exp(ϕ(s)^T θ_a) / Σ_b exp(ϕ(s)^T θ_b)
    
    Args:
        state: tile-coded feature vector ϕ(s), shape (N_features,)
        theta: policy weights, shape (N_actions, N_features)
    
    Returns:
        probs: action probabilities, shape (N_actions,)
    """
    # logits[a] = ϕ(s)^T θ_a = θ[a, :] · state
    logits = theta @ state                       # shape (N_actions,)
    logits -= np.max(logits)                     # subtract max for numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    return probs

def grad_log_policy(state, action, theta):
    """Compute ∇_θ ln π(a|s; θ) for the softmax linear policy.
    
    For softmax linear:
        ∇_{θ_a} ln π(a|s; θ) = ϕ(s) - Σ_b π(b|s; θ) ϕ(s) 
                              = ϕ(s) (1 - π(a|s; θ))           for the taken action a
        ∇_{θ_b} ln π(a|s; θ) = -π(b|s; θ) ϕ(s)               for b ≠ a 
    
    Equivalently, the full gradient w.r.t. θ (as a matrix) is:
        ∇_θ ln π(a|s; θ)[b, :] = (1_{b=a} - π(b|s; θ)) ϕ(s)
    
    Args:
        state:  feature vector ϕ(s), shape (N_features,)
        action: the action taken (int)
        theta:  policy weights, shape (N_actions, N_features)
    
    Returns:
        grad: gradient matrix, shape (N_actions, N_features)
    """
    probs = softmax_policy(state, theta)
    
    # grad[b, :] = (1_{b=a} - π(b|s;θ)) * ϕ(s)
    grad = np.zeros_like(theta)
    for b in range(theta.shape[0]):
        indicator = 1.0 if b == action else 0.0 
        grad[b, :] = (indicator - probs[b]) * state 
    
    return grad 

# ========================================================= 
# Policy gradient loop: one-step actor-critic (episodic) 
# ========================================================= 
# Loop forever (for each episode): 
#   Initialize S (first state of episode) 
#   I <-- 1 
#   Loop while S is not terminal (for each time step): 
#       A ~ π(.|S, θ) 
#       Take action A, observe S', R 
#       δ <-- R + γ v̂(S', w) - v̂(S, w) 
#       w <-- w + α^w · δ · ∇_w v̂(S, w) 
#       θ <-- θ + α^θ · I · δ · ∇_θ ln π(A|S, θ) 
#       I <-- γ · I 
#       S <-- S' 
# ========================================================= 

for k in range(max_iter): 

    # state feature representation from RAM  
    state = get_state(ram2, ram1)

    # ----------------------------- 
    # A ~ π(·|S, θ)  — sample action from softmax policy 
    # ----------------------------- 
    probs = softmax_policy(state, theta) 
    action = rng.choice(N_actions, p=probs) 

    # Take action A, observe R and transition to new RAM state 
    ram3, reward, terminated, truncated, info = env.step(action) 

    # clip reward to [-1, 1] for stability 
    reward_clipped = np.clip(reward, -1.0, 1.0)

    # track raw reward 
    reward_ep += reward 

    # ----------------------------- 
    # One-step actor-critic update 
    # ----------------------------- 

    # v̂(S, w) = w^T ϕ(S) 
    v_s = np.dot(w, state)

    # δ <-- R + γ v̂(S', w) - v̂(S, w)
    if terminated or truncated: 
        delta = reward_clipped - v_s 
    else: 
        next_state = get_state(ram3, ram2)
        v_sp = np.dot(w, next_state)
        delta = reward_clipped + gamma * v_sp - v_s 

    # Critic update: w <-- w + α^w · δ · ∇_w v̂(S, w)
    # For linear v̂(s, w) = w^T ϕ(s), ∇_w v̂ = ϕ(s)
    w += alpha_w * delta * state 

    # Actor update: θ <-- θ + α^θ · I · δ · ∇_θ ln π(A|S, θ)
    grad = grad_log_policy(state, action, theta) 
    theta += alpha_theta * I * delta * grad 

    # I <-- γ · I 
    I *= gamma 

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
                  f"Max: {np.max(last50):5.1f}")

        # reset episode 
        reward_ep = 0 
        I = 1.0   # reset discount factor accumulator for new episode 
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
axes[0].set_title("Policy Gradient (Actor-Critic) with RAM Features: Atari Breakout")
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
plt.savefig("hw4/policy_gradient_breakout_ram.png", dpi=300)
print(f"\nPlot saved to hw4/policy_gradient_breakout_ram.png")
print(f"Total episodes: {len(ep_reward_hist)}")
print(f"Final {window}-episode average reward: {np.mean(ep_reward_hist[-window:]):.2f}")

# ----------------------------- 
# print learned weight summary (tile-coded weights are too many to print individually)
# ----------------------------- 

print("\n--- Learned weight summary (θ — actor) ---")
action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
print(f"{'Action':>10s} {'||θ||':>8s} {'mean':>8s} {'min':>8s} {'max':>8s} {'nonzero':>8s}")
for a in range(N_actions):
    wt = theta[a]
    print(f"{action_names[a]:>10s} {np.linalg.norm(wt):8.4f} {wt.mean():8.4f} "
          f"{wt.min():8.4f} {wt.max():8.4f} {np.sum(np.abs(wt) > 1e-6):8d}")

print(f"\n--- Learned weight summary (w — critic) ---")
print(f"  ||w|| = {np.linalg.norm(w):.4f}, mean = {w.mean():.4f}, "
      f"min = {w.min():.4f}, max = {w.max():.4f}, "
      f"nonzero = {np.sum(np.abs(w) > 1e-6)}")
