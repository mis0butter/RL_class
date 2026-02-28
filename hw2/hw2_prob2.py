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

np.random.seed(0)

# -----------------------------
# Parameters (shared)
# -----------------------------

n_bins = 4              # bins per obs dimension
S = n_bins ** 6         # 729 discrete states
A = 3                   # 3 possible actions
gamma = 0.95            # discount factor 

episodes = 100
iters = 1000

# -----------------------------
# State discretization
# -----------------------------

obs_low  = np.array([-1.0, -1.0, -1.0, -1.0, -4*np.pi, -9*np.pi])
obs_high = np.array([ 1.0,  1.0,  1.0,  1.0,  4*np.pi,  9*np.pi])

def discretize(obs):
    clipped = np.clip(obs, obs_low, obs_high)
    scaled  = (clipped - obs_low) / (obs_high - obs_low)
    bins    = np.minimum((scaled * n_bins).astype(int), n_bins - 1)
    idx = 0
    for b in bins:
        idx = idx * n_bins + b
    return idx

# -----------------------------
# Stochastic stationary policy π(a|s)
# -----------------------------
# Pumping policy based on angular velocity of second joint (obs[5]):
#   dθ₂ >  1 → P(a=0,1,2) = (0.1, 0.2, 0.7)   push right
#   dθ₂ < −1 → P(a=0,1,2) = (0.7, 0.2, 0.1)   push left
#   else      → P(a=0,1,2) = (0.3, 0.4, 0.3)   mostly idle

_probs = {1: np.array([0.1, 0.2, 0.7]),
          -1: np.array([0.7, 0.2, 0.1]),
          0: np.array([0.3, 0.4, 0.3])}

def pi_probs(obs):
    d = obs[5]
    if d > 1.0:   return _probs[1]
    if d < -1.0:  return _probs[-1]
    return _probs[0]

# -----------------------------
# Environment + fixed initial state
# -----------------------------

env = gym.make("Acrobot-v1")

# both links down, zero velocity
INIT_STATE = np.array([0.0, 0.0, 0.0, 0.0])   

def reset_env():
        
    # set the initial state of the environment (env.reset randomizes the state)
    env.reset()
    env.unwrapped.state = INIT_STATE.copy()
    
    th1, th2, dth1, dth2 = INIT_STATE
    
    return np.array([np.cos(th1), np.sin(th1),
                     np.cos(th2), np.sin(th2),
                     dth1, dth2])

# =========================================================
# TD(lambda) for lambda = 0, 0.3, 0.5, 0.7, 1
# =========================================================

lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
alpha0 = 0.2
avg_dk_sq = np.zeros((episodes, len(lambdas)))

for i in range(len(lambdas)):

    lam = lambdas[i]
    V_k = np.zeros(S)
    rng_td = np.random.default_rng(0)

    for ep in range(episodes):

        # reset the environment and get the initial state
        obs = reset_env()
        s = discretize(obs)
        e = np.zeros(S)
        ep_td_errors_sq = []

        # for each episode, run for a fixed number of iters
        for iter in range(iters):

            # sample action from pi(s,:)
            a = rng_td.choice(A, p=pi_probs(obs)) 
            
            import pdb; pdb.set_trace()

            # sample next state and reward from the MDP
            obs_next, r, term, trunc, _ = env.step(a)
            s_next = discretize(obs_next)
            
            done = term or trunc

            # TD error 
            dk = r + gamma * (0.0 if done else V_k[s_next]) - V_k[s]
            ep_td_errors_sq.append(dk ** 2)

            # update eligibility trace
            e *= gamma * lam
            e[s] = 1.0
            V_k += alpha0 * dk * e

            if done:
                obs = reset_env()
                s = discretize(obs)
                e = np.zeros(S)
            else:
                s = s_next
                obs = obs_next

        avg_dk_sq[ep, i] = np.mean(ep_td_errors_sq)

    print(f"λ={lam}  done  (final avg |dk|^2 = {avg_dk_sq[-1, i]:.4f})")

env.close()

# ----------------------------- 
# Plot all lambdas
# ----------------------------- 

plt.figure(figsize=(6.5, 4.0), dpi=150)
for i in range(len(lambdas)):
    y = np.maximum(avg_dk_sq[:, i], 1e-300)
    plt.semilogy(np.arange(1, episodes + 1), y, linewidth=2, label=f"lambda={lambdas[i]}")

plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.xlabel("Episode")
plt.ylabel(r"Average $\|\delta_k\|^2$ (log scale)")
plt.title("Prob 2: TD($\\lambda$) on Acrobot-v1")
plt.legend()
plt.tight_layout()
plt.savefig("TD_lambda_acrobot.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure: TD_lambda_acrobot.png")
