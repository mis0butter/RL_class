import gymnasium as gym 

# ====================================================================== 
# Question 2 
# ====================================================================== 

# Questions: 
# Pick your own stochastic stationary policy π for this problem. Write a computer
# program to simulate TD learning to evaluate V
# π with a fixed chosen initial state. You should run
# TD learning in a number of episodes (e.g., 20) where each episode runs a number of iterations
# (e.g., 1000). Simulate your program for different λ, i.e., λ = 0, 0.3, 0.5, 0.7, 1. The output of your
# simulations should be a plot which shows 5 curves of the average of TD errors ∥dk∥
# 2 over the
# number of episodes. Explain the impacts of λ on the performance of TD learning. You have to
# submit your code and specify in your solution what is your policy.

env = gym.make('Acrobot-v1') 

# ====================================================================== 
# Question 3 
# ====================================================================== 



