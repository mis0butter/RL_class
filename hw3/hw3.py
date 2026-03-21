# Problem 2 [40 points]
# Consider the cartpole environment from OpenAI Gym, which can be found here:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# In this question, we will implement tabular policy gradient method to solve this Cartpole
# problem, where we have the discount factor γ = 0.9. Specify your choice of initial policy.
# Questions: Write a simulation program to implement policy gradient method. You can use
# any variant of policy gradients and TD learning, e.g., the online actor-critic in the lecture notes
# or natural actor-critic method (policy updates are natural policy gradient). Explain how you
# obtain the gradient at each iteration, i.e., using only one sample or multiple samples (the so-called
# minibatch). Submit your simulation outputs showing the progress of your value function at each
# iteration, i.e., f(πk). Does the policy obtained by your simulation solves the problem?

# ========================================================= 
# Method: Tabular Online Actor-Critic (Policy Gradient + TD)
# =========================================================
# We use a softmax tabular policy:
#   π(a|s) = exp(θ[s,a]) / Σ_a' exp(θ[s,a'])
# and a tabular value function V(s) for the critic.
#
# At each step the TD error is:
#   δ = r + γ V(s') - V(s)
# Critic update:  V(s) ← V(s) + α_v * δ
# Actor update:   θ(s,a) ← θ(s,a) + α_θ * δ * (1 - π(a|s))   (for the chosen a)
#                 θ(s,a') ← θ(s,a') - α_θ * δ * π(a'|s)        (for a' ≠ a)
# which is equivalent to θ(s,:) += α_θ * δ * (e_a - π(s,:)),
# where e_a is the one-hot vector for action a.
#
# This uses a single sample (one transition) per gradient update (fully online).
#
# Initial policy: θ = 0 everywhere → uniform random policy π(a|s) = 0.5

# ========================================================= 
# import packages 
# ========================================================= 

import gymnasium as gym 
# gym.logger.set_level(40)  # suppress warnings
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ========================================================= 
# parameters 
# ========================================================= 

n_bins = 10             # bins per observation dimension
S = n_bins ** 4         # total number of discrete states (CartPole has 4-dim obs)
A = 2                   # 2 actions: push left or push right
gamma = 0.9             # discount factor (as specified)
alpha_theta = 0.1       # actor learning rate
alpha_v = 0.1           # critic learning rate
n_episodes = 3000       # number of training episodes
max_t = 500             # max steps per episode (CartPole-v0 cap is 200)
print_every = 100

# ========================================================= 
# state discretization (similar to hw2)
# ========================================================= 

# CartPole observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
# We clip to reasonable ranges to handle outliers
obs_low  = np.array([-2.4, -3.0, -0.21, -3.0])
obs_high = np.array([ 2.4,  3.0,  0.21,  3.0])

def discretize(obs):
    clipped = np.clip(obs, obs_low, obs_high)
    scaled  = (clipped - obs_low) / (obs_high - obs_low)
    bins    = np.minimum((scaled * n_bins).astype(int), n_bins - 1)
    idx = 0
    for b in bins:
        idx = idx * n_bins + b
    return idx

# ========================================================= 
# softmax tabular policy 
# ========================================================= 

# θ[s, a]: policy parameters — initialized to 0 (uniform random initial policy)
theta = np.zeros((S, A))

# V[s]: value function (critic) — initialized to 0
V = np.zeros(S)

def softmax_policy(s):
    """Return action probabilities π(a|s) using softmax over θ[s, :]."""
    logits = theta[s] - np.max(theta[s])  # subtract max for numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def choose_action(s):
    """Sample action from π(·|s)."""
    probs = softmax_policy(s)
    return np.random.choice(A, p=probs)

# ========================================================= 
# train agent: online actor-critic
# ========================================================= 

env = gym.make('CartPole-v1')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

scores = []
value_at_start = []  # f(π_k) = V(s_0) tracked over episodes

for i_episode in range(1, n_episodes + 1):

    obs, _ = env.reset()
    s = discretize(obs)
    s0 = s  # remember start state for tracking V(s_0)
    episode_reward = 0

    for t in range(max_t):
        
        # choose action from softmax policy
        a = choose_action(s)
        probs = softmax_policy(s)

        # take action
        obs_next, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s_next = discretize(obs_next)

        # TD error (critic)
        delta = reward + gamma * (0.0 if done else V[s_next]) - V[s]

        # critic update
        V[s] += alpha_v * delta

        # actor update: θ(s,:) += α_θ * δ * (e_a - π(s,:))
        grad = -probs.copy()       # -π(a'|s) for all a'
        grad[a] += 1.0             # +1 for the chosen action  → (e_a - π)
        theta[s] += alpha_theta * delta * grad

        episode_reward += reward
        s = s_next

        if done:
            break

    scores.append(episode_reward)
    value_at_start.append(V[s0])

    if i_episode % print_every == 0:
        avg = np.mean(scores[-100:])
        print(f'Episode {i_episode}\tAverage Score (last 100): {avg:.2f}\tV(s0): {V[s0]:.2f}')
    if np.mean(scores[-100:]) >= 195.0:
        print(f'Environment solved in {i_episode - 100} episodes!\t'
              f'Average Score: {np.mean(scores[-100:]):.2f}')
        break

env.close()

# ========================================================= 
# plot scores 
# ========================================================= 

fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=150)

# Plot 1: Episode scores
axes[0].plot(np.arange(1, len(scores)+1), scores, alpha=0.3, label='Score')
# running average
window = 100
if len(scores) >= window:
    running_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    axes[0].plot(np.arange(window, window + len(running_avg)), running_avg,
                 color='red', linewidth=2, label=f'{window}-episode avg')
axes[0].set_ylabel('Score')
axes[0].set_xlabel('Episode #')
axes[0].set_title('Tabular Actor-Critic on CartPole-v0')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Value function at initial state f(π_k)
axes[1].plot(np.arange(1, len(value_at_start)+1), value_at_start, color='green', linewidth=1)
axes[1].set_ylabel('V(s₀)')
axes[1].set_xlabel('Episode #')
axes[1].set_title('Progress of Value Function f(πₖ) = V(s₀)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hw3_prob2_tabular_actor_critic.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved figure: hw3_prob2_tabular_actor_critic.png')

# ========================================================= 
# animate agent (greedy policy) — save as GIF
# ========================================================= 

from matplotlib.animation import FuncAnimation

env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, _ = env.reset()
s = discretize(obs)

frames = []
total_reward = 0
for t in range(200):
    frames.append(env.render())  # capture RGB frame
    probs = softmax_policy(s)
    a = np.argmax(probs)  # greedy action
    obs, reward, terminated, truncated, _ = env.step(a)
    done = terminated or truncated
    s = discretize(obs)
    total_reward += reward
    if done:
        break

env.close()
print(f'\nGreedy policy test: survived {t+1} steps, total reward = {total_reward}')

# create animation from frames
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
img = ax.imshow(frames[0])

def update(frame_idx):
    img.set_data(frames[frame_idx])
    return [img]

anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
anim.save('hw3_cartpole_animation.gif', writer='pillow', fps=20)
plt.close()
print(f'Saved animation: hw3_cartpole_animation.gif  ({len(frames)} frames)')
