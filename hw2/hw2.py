
# Consider the Acrobot environment from OpenAI Gym, which can be found here:
# https://mgoulao.github.io/gym-docs/environments/classic_control/acrobot/
# Questions: Pick your own stochastic stationary policy π for this problem. Write a computer
# program to simulate TD learning to evaluate V π with a fixed chosen initial state. You should run
# TD learning in a number of episodes (e.g., 20) where each episode runs a number of iterations
# (e.g., 1000). Simulate your program for different λ, i.e., λ = 0, 0.3, 0.5, 0.7, 1. The output of your
# simulations should be a plot which shows 5 curves of the average of TD errors ∥dk∥2 over the
# number of episodes. Explain the impacts of λ on the performance of TD learning. You have to
# submit your code and specify in your solution what is your policy.

# ========================================================= 
# Acrobot Description 
# ========================================================= 

# This environment is part of the Classic Control environments. Please read that page first for general information.
# 
# Action Space      | Discrete(3)
# ------------------|-------------|
# Observation Shape | (6,)
# Observation High  | [ 1. 1. 1. 1. 12.57 28.27]
# Observation Low   | [ -1. -1. -1. -1. -12.57 -28.27]
# Import            | gym.make("Acrobot-v1") 

# ========================================================= 
# Problem 2 
# ========================================================= 

import numpy as np 
import gymnasium as gym 
import matplotlib.pyplot as plt 
from tile_coding import * 

np.random.seed(0) 

# ----------------------------- 
# Parameters 
# ----------------------------- 

S = 4 
A = 3 
gamma = 0.95 
tol = 1e-12 
max_iter = 1000 

# ----------------------------- 
# Environment creation 
# ----------------------------- 

env = gym.make('Acrobot-v1') 

lows  = np.array([ -1. , -1. , -1. , -1. , -12.57 , -28.27])
highs = np.array([ 1. , 1. , 1. , 1. , 12.57 , 28.27]) 

bins_per_dim = (4, 4, 4, 4, 4, 4) 

#                 bins,         offsets
tiling_specs = [( bins_per_dim, (-0.167, -0.167, -0.167, -0.167, -2.1, -4.71 ) ), 
                ( bins_per_dim, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) ), 
                ( bins_per_dim, (0.167, 0.167, 0.167, 0.167, 2.1, 4.71 ) ), 
                ]
tilings = create_tilings(lows, highs, tiling_specs)

visualize_tilings(tilings)

import pdb; pdb.set_trace() 


# # ========================================================= 
# # TD(lambda) for lambda = 0, 0.3,0.5,0.7,1
# # =========================================================

# lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
# alpha0 = 0.2
# rng = np.random.default_rng(0)

# # V_k = np.zeros((S,len(lambdas)))
# errTD = np.zeros((max_iter,len(lambdas)))

# for i in range(len(lambdas)): 
#     lam = lambdas[i] 
#     V_k = np.zeros(S)
#     e = np.zeros(S)  # eligibility trace
#     s = rng.integers(S)  # initial state (fixed per lambda run)

#     for k in range(max_iter):

#         # sample action from pi(s,:)
#         a = rng.choice(A, p=pi[s])
        

#         # sample next state from P(s,:,a)