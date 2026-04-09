# Consider the Acrobot environment from OpenAI Gym, which can be found here:
# https://mgoulao.github.io/gym-docs/environments/classic_control/acrobot/
# Questions: Pick your own stochastic stationary policy π for this problem. Write a computer
# program to simulate TD learning to evaluate V π with a fixed chosen initial state. You should run
# TD learning in a number of episodes (e.g., 20) where each episode runs a number of iterations
# (e.g., 1000). Simulate your program for different λ, i.e., λ = 0, 0.3, 0.5, 0.7, 1. The output of your
# simulations should be a plot which shows 5 curves of the average of TD errors ∥dk∥2 over the
# number of episodes. Explain the impacts of λ on the performance of TD learning. You have to
# submit your code and specify in your solution what is your policy.

# ====================================================================== 
# import packages 
# ====================================================================== 

import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt 
import time 

np.random.seed(0) 

# ====================================================================== 
# create acrobot environment 
# ====================================================================== 

# env = gym.make('Acrobot-v1') 
env = gym.make('Acrobot-v1', render_mode='human')  # render_mode='human' requires pygame 

# get observation space and action space 
S = env.observation_space.shape[0]       # number of states 
A = env.action_space.n                   # number of actions     

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
    
    if terminated or truncated: 
        obs, info = env.reset() 

env.close() 

# ====================================================================== 
# define a stochastic stationary policy 
# ====================================================================== 

pi = np.random.rand(S, A)
pi = pi / pi.sum(axis=1, keepdims=True)


    

