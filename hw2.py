# Consider the Acrobot environment from OpenAI Gym, which can be found here:
# https://mgoulao.github.io/gym-docs/environments/classic_control/acrobot/
# Questions: Pick your own stochastic stationary policy π for this problem. Write a computer
# program to simulate TD learning to evaluate V π with a fixed chosen initial state. You should run
# TD learning in a number of episodes (e.g., 20) where each episode runs a number of iterations
# (e.g., 1000). Simulate your program for different λ, i.e., λ = 0, 0.3, 0.5, 0.7, 1. The output of your
# simulations should be a plot which shows 5 curves of the average of TD errors ∥dk∥2 over the
# number of episodes. Explain the impacts of λ on the performance of TD learning. You have to
# submit your code and specify in your solution what is your policy.

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

np.random.seed(0)

# -----------------------------
# Parameters (shared)
# -----------------------------

n_bins = 3              # bins per obs dimension
S = n_bins ** 6         # 729 discrete states
A = 3                   # 3 possible actions
gamma = 0.95            # discount factor
max_iter = 50000        # total TD iterations (~100 episodes × 500 steps)

# -----------------------------
# State discretization
# -----------------------------

obs_low  = np.array([-1.0, -1.0, -1.0, -1.0, -4.0, -9.0])
obs_high = np.array([ 1.0,  1.0,  1.0,  1.0,  4.0,  9.0])

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
FIXED_STATE = np.array([0.0, 0.0, 0.0, 0.0])   # both links down, zero velocity

def obs_from_state(st):
    return np.array([np.cos(st[0]), np.sin(st[0]),
                     np.cos(st[1]), np.sin(st[1]),
                     st[2], st[3]])

def reset_env():
    env.reset()
    env.unwrapped.state = FIXED_STATE.copy()
    return obs_from_state(FIXED_STATE)

# =========================================================
# Estimate V_pi via Monte Carlo
# (replaces the exact Bellman solve in TD.py — we have no
#  transition model, so we average discounted returns instead)
# =========================================================

n_mc = 500
rng_mc = np.random.default_rng(123)
state_returns = {}

for _ in range(n_mc):
    obs = reset_env()
    traj = []
    for _ in range(500):
        s = discretize(obs)
        a = rng_mc.choice(A, p=pi_probs(obs))
        obs, r, term, trunc, _ = env.step(a)
        traj.append((s, r))
        if term or trunc:
            break
    G = 0.0
    for s, r in reversed(traj):
        G = r + gamma * G
        state_returns.setdefault(s, []).append(G)

V_pi = np.zeros(S)
for s, rets in state_returns.items():
    V_pi[s] = np.mean(rets)

print(f"MC done — {len(state_returns)} unique states visited")

# =========================================================
# TD(lambda) for lambda = 0, 0.3, 0.5, 0.7, 1
# =========================================================

lambdas = [0.0, 0.3, 0.5, 0.7]
alpha0 = 0.2
rng = np.random.default_rng(0)
V_k = np.zeros((S, len(lambdas)))
errTD = np.zeros((max_iter, len(lambdas)))

for i in range(len(lambdas)):

    lam = lambdas[i]
    V_k = np.zeros(S)
    e = np.zeros(S)               # eligibility trace
    rng_td = np.random.default_rng(0)

    obs = reset_env()
    s = discretize(obs)           # initial state (fixed)

    for k in range(max_iter):

        # sample action from π(s)
        a = rng_td.choice(A, p=pi_probs(obs))

        # take action in environment
        obs_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        s_next = discretize(obs_next)

        # TD error
        delta = r + gamma * (0.0 if done else V_k[s_next]) - V_k[s]

        # stepsize
        alpha = alpha0

        # update eligibility traces (accumulating traces)
        e *= gamma * lam
        e[s] += 1.0

        # update value function
        V_k += alpha * delta * e

        # error to true V_pi
        errTD[k, i] = np.linalg.norm(V_k - V_pi, 2)

        # advance (reset on episode end)
        if done:
            obs = reset_env()
            s = discretize(obs)
            e = np.zeros(S)       # traces reset between episodes
        else:
            s = s_next
            obs = obs_next

    print(f"λ={lam}  done  (final error = {errTD[-1, i]:.4f})")

env.close()

# Plot all lambdas
plt.figure(figsize=(6.5, 4.0), dpi=150)
for i in range(len(lambdas)):
    y = np.maximum(errTD[:, i], 1e-300)   # avoid log(0)
    plt.semilogy(np.arange(max_iter), y, linewidth=2, label=f"lambda={lambdas[i]}")

plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.xlabel("Iteration k")
plt.ylabel(r"$\|V_k - V_\pi\|_2$ (log scale)")
plt.title("TD($\\lambda$) Policy Evaluation on Acrobot-v1")
plt.legend()
plt.tight_layout()
plt.savefig("TD_lambda_acrobot.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure: TD_lambda_acrobot.png")
