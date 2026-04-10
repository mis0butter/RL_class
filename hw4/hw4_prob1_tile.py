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
import cv2 

import ale_py 
gym.register_envs(ale_py) 

rng = np.random.default_rng(0) 

# ========================================================= 
# create breakout environment 
# ========================================================= 

# env = gym.make('ALE/Breakout-v5', render_mode='human')  # render_mode='human' requires pygame 
env = gym.make('ALE/Breakout-v5')  

# get observation space and action space 
N_states  = env.observation_space.shape       # states: (210, 160, 3)  
N_actions = env.action_space.n                # 4 actions: left, right, nothing, fire     

print('observation space: \n', N_states) 
print('action space: \n', N_actions) 

# ========================================================= 
# function approximation: feature vector 
# ========================================================= 

def make_grayscale(obs): 

    # obs is (210, 160, 3)
    rgb_weight = np.array([0.299, 0.587, 0.114])

    # convert to grayscale 
    grayscale  = np.dot(obs, rgb_weight) 

    # reduce size 
    obs_reduce = cv2.resize(grayscale, (42, 42), interpolation = cv2.INTER_AREA) 

    return obs_reduce.flatten() 

# ----------------------------- 

def get_state(frame1, frame2): 
    
    # make obs grayscale and reduce  
    frame1_reduce   = make_grayscale(frame1) 
    frame2_reduce   = make_grayscale(frame2) 

    # position AND velocity 
    position = frame2_reduce.flatten()  # use frame2 as it's the more recent frame
    velocity = frame2_reduce.flatten() - frame1_reduce.flatten() 
    
    return np.concatenate([position, velocity]) / 255.0  

# ----------------------------- 

# reduce obs 
obs, info = env.reset() 
obs_reduce = make_grayscale(obs) 

# get feature size 
N_states_reduce = obs_reduce.size 
N_features      = 2 * N_states_reduce 

# ========================================================= 
# test environment and features 
# ========================================================= 

# reset env and see initial observation 
obs, info = env.reset() 

# run episode with random actions so we can watch the render 
num_steps = 50 
for step in range(num_steps): 

    # sample random action 
    # action = my_intelligent_policy(obs) 
    action = env.action_space.sample() 

    # take action and get new observation 
    obs, reward, terminated, truncated, info = env.step(action) 

    # (Testing loop just to render)

    # # render the env and wait a bit 
    # env.render() 
    # time.sleep(0.001) 
    
    if terminated or truncated: 
        obs, info = env.reset() 

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

# initialize feature vector ϕ(s, a) 
frame1, _ = env.reset() 
frame2, _, _, _, _ = env.step(env.action_space.sample())

# initialize weight matrix!! shape: [N_actions, N_features]
theta = np.ones((N_actions, N_features))
theta = theta / theta.sum()  

# parameters 
epsilon0 = 1.0       # initial exploration rate  
alpha    = 0.001     # step size (reduced for stability)
gamma    = 0.95      # discount factor 

ep_reward_hist = [] 
reward_ep   = 0 
max_iter    = 100000 

# ========================================================= 
# Q-learning loop!!! 
# ========================================================= 

# ----------------------------- 
# loop!!! 
# ----------------------------- 

for k in range(max_iter): 
    
    # decay epsilon slightly faster so it drops by 100k steps
    epsilon = max(0.1, epsilon0 / (1.0 + 0.00005 * k))

    # state feature representation
    state = get_state(frame1, frame2)

    # MAGIC: compute all 4 q_vals INSTANTLY using matrix dot product!
    q_vals = np.dot(theta, state) 

    # Choose A from S using policy derived from Q (epsilon-greedy) 
    if rng.random() < epsilon: 
        action = env.action_space.sample() 
    else: 
        action = np.argmax(q_vals) 

    # Take action A, observe R and transition to new frame (frame3)
    frame3, reward, terminated, truncated, info = env.step(action) 

    # track reward
    reward_ep += reward 

    # ----------------------------- 
    # Q(S, A) <-- Q(S, A) + α[ R + γ max_a Q(S', a) - Q(S,A) ]
    # ----------------------------- 

    # compute td target based on if ep terminated or not 
    if terminated or truncated: 
        td_target = reward 
    else: 
        # R + γ max_a Q(S', a) (State S' = frame2, frame3)
        next_state = get_state(frame2, frame3)
        next_q_vals = np.dot(theta, next_state)
        td_target   = reward + gamma * np.max(next_q_vals) 

    # R + γ max_a Q(S', a) - Q(S,A)
    td_err = td_target - q_vals[action]  
    
    # update ONLY the weights for the action we took! 
    # (equivalent to phi[action, :] update from before, but 100x faster)
    theta[action, :] += alpha * td_err * state 

    # ----------------------------- 
    # advance to next state or reset episode 
    # ----------------------------- 
    
    if terminated or truncated: 

        # save episode total reward 
        ep_reward_hist.append(reward_ep) 

        # reset episode 
        reward_ep = 0 
        frame1, _ = env.reset()
        frame2, _, _, _, _ = env.step(env.action_space.sample())

    else: 
        frame1 = frame2 
        frame2 = frame3 

# ========================================================= 
# plot rewards 
# ========================================================= 

# import pdb; pdb.set_trace() 

ep_reward_hist = np.array(ep_reward_hist) 

plt.figure() 
plt.semilogy(np.arange(len(ep_reward_hist)), ep_reward_hist) 
plt.xlabel("Episode") 
plt.ylabel("Reward") 
plt.title("Q-learning: Atari Breakout")

# plt.show() 
plt.savefig("Q_learning_atari_breakout.png", dpi=300)

    

