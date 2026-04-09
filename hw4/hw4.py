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
# 3. Explain the choice of feature representation and behavior policies for data collection. Implement the outcome of your learning and report the results.

# ========================================================= 
# import packages 
# ========================================================= 

import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt 
import time 

import ale_py 
gym.register_envs(ale_py) 

np.random.seed(0) 

# ========================================================= 
# create breakout environment 
# ========================================================= 

env = gym.make('ALE/Breakout-v5', render_mode='human')  # render_mode='human' requires pygame 

# get observation space and action space 
S = env.observation_space.shape       # states: (210, 160, 3)  
A = env.action_space.n                # 4 actions: left, right, nothing, fire     

print('observation space: \n', S) 
print('action space: \n', A) 

# reset env and see initial observation 
obs, info = env.reset() 

# run episode with random actions so we can watch the render 
num_steps = 500 
for step in range(num_steps): 

    # sample random action 
    # action = my_intelligent_policy(obs) 
    action = env.action_space.sample() 

    # take action and get new observation 
    obs, reward, terminated, truncated, info = env.step(action) 

    # render the env and wait a bit 
    env.render() 
    time.sleep(0.001) 

    # import pdb; pdb.set_trace() 
    
    if terminated or truncated: 
        obs, info = env.reset() 

env.close() 

# ========================================================= 
# define a stochastic stationary policy 
# ========================================================= 

# pi = np.random.rand(S, A)
# pi = pi / pi.sum(axis=1, keepdims=True)


    

# =========================================================
