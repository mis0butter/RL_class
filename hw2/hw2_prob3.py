# Consider the cartpole environment from OpenAI Gym, which can be found here:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# In this question, we will implement tabular Q-learning to solve this Cartpole problem, where
# we have the discount factor γ = 0.9.
# 
# Questions: Write a simulation program to implement Q-learning. In your simulation, consider a
# number of episodes (e.g., 100), where each episode runs until the program terminates. For each
# episode, choose your own behavior policy to generate data, e.g., the behavior policy can be a
# randomized policy or the policy of the previous episode.
# 
# Submit your simulation outputs showing the average of ∥dk∥2 over the number of episodes,
# where
#   dk = R(sk, ak) + γ max_b Q(s'_k, b) - Q(s_k, a_k).
# 
# Does your returned policy solve the problem, i.e., how long your program last by running the 
# policy of the last episode? Make comments on the choice of behavior policies and the number of episodes.

# ========================================================= 
# Cartpole Description 
# ========================================================= 

# This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
# ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
# A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
# The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
#  in the left and right direction on the cart.

# ### Action Space

# The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
#  of the fixed force the cart is pushed with.

# | Num | Action                 |
# |-----|------------------------|
# | 0   | Push cart to the left  |
# | 1   | Push cart to the right |

# **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
#  the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

# ### Observation Space

# The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

# | Num | Observation           | Min                 | Max               |
# |-----|-----------------------|---------------------|-------------------|
# | 0   | Cart Position         | -4.8                | 4.8               |
# | 1   | Cart Velocity         | -Inf                | Inf               |
# | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
# | 3   | Pole Angular Velocity | -Inf                | Inf               |

# **Note:** While the ranges above denote the possible values for observation space of each element,
#     it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
# -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
#    if the cart leaves the `(-2.4, 2.4)` range.
# -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
#    if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

# ### Rewards

# Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
# including the termination step, is allotted. The threshold for rewards is 475 for v1.

# ### Starting State

# All observations are assigned a uniformly random value in `(-0.05, 0.05)`

# ### Episode End

# The episode ends if any one of the following occurs:

# 1. Termination: Pole Angle is greater than ±12°
# 2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
# 3. Truncation: Episode length is greater than 500 (200 for v0)

# ========================================================= 
# Problem 3 
# ========================================================= 

import numpy as np 
import gymnasium as gym 
import matplotlib.pyplot as plt 

np.random.seed(0) 

# ----------------------------- 
# Parameters 
# ----------------------------- 

n_bins = 12             # bins per obs dimension
A = 2                   # 2 actions: left, right
gamma = 0.9             # discount factor
alpha = 0.1             # learning rate
episodes = 500
epsilon_start = 1.0     # exploration rate
epsilon_end = 0.01      # minimum exploration rate
epsilon_decay = 0.995   # multiplicative decay per episode

# ----------------------------- 
# State discretization 
# ----------------------------- 

obs_low  = np.array([-2.4, -3.0, -0.2095, -3.0])
obs_high = np.array([ 2.4,  3.0,  0.2095,  3.0])

S = n_bins ** 4

def discretize(obs):
    clipped = np.clip(obs, obs_low, obs_high)
    scaled  = (clipped - obs_low) / (obs_high - obs_low)
    bins    = np.minimum((scaled * n_bins).astype(int), n_bins - 1)
    idx = 0
    for b in bins:
        idx = idx * n_bins + b
    return idx

# ----------------------------- 
# Environment 
# ----------------------------- 

env = gym.make('CartPole-v1')

# =========================================================
# Q-learning with epsilon-greedy behavior policy
# =========================================================

Q = np.zeros((S, A))
epsilon = epsilon_start
avg_dk_sq = np.zeros(episodes)
episode_rewards = np.zeros(episodes)

for ep in range(episodes):

    obs, _ = env.reset()
    s = discretize(obs)
    done = False
    ep_td_errors_sq = []
    total_reward = 0

    # for each episode, run until the episode terminates
    while not done:
        
        # epsilon-greedy behavior policy
        if np.random.rand() < epsilon:
            a = np.random.randint(A)
        else:
            a = np.argmax(Q[s])

        # sample next state and reward from the MDP
        obs_next, r, term, trunc, _ = env.step(a)
        
        done  = term or trunc
        s_next = discretize(obs_next)

        # TD error: dk = R + gamma * max_b Q(s', b) - Q(s, a)
        dk = r + gamma * (0.0 if done else np.max(Q[s_next])) - Q[s, a]
        ep_td_errors_sq.append(dk ** 2)

        # Q-learning update
        Q[s, a] += alpha * dk

        s = s_next
        total_reward += r

    avg_dk_sq[ep] = np.mean(ep_td_errors_sq)
    episode_rewards[ep] = total_reward
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1:4d}  |  avg ||dk||^2 = {avg_dk_sq[ep]:.4f}"
              f"  |  reward = {total_reward:.0f}  |  eps = {epsilon:.3f}")

# ----------------------------- 
# Plot 
# ----------------------------- 

plt.figure(figsize=(6.5, 4.0), dpi=150) 
plt.semilogy(np.arange(1, episodes + 1),
                 np.maximum(avg_dk_sq, 1e-300), linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel(r"Average $\|\delta_k\|^2$ (log scale)")
plt.title("Prob 3: Q-learning on Cartpole-v1")
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig("Q_learning_cartpole.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure: Q_learning_cartpole.png")
