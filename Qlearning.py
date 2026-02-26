import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Random MDP + Value Iteration + Policy Iteration (clean)
# =========================================================

np.random.seed(0)

# -----------------------------
# Parameters (shared)
# -----------------------------
S = 4
A = 3
gamma = 0.95
tol = 1e-12
maxIter = 100000

# -----------------------------
# Environment creation
# -----------------------------
# P[s, s', a]
P = np.random.rand(S, S, A)
for a in range(A):
    for s in range(S):
        P[s, :, a] /= P[s, :, a].sum()   # normalize over s'

# R[s, a]
R = np.random.rand(S, A)

# Initial stochastic policy pi[s, a]
pi = np.random.rand(S, A)
pi = pi / pi.sum(axis=1, keepdims=True)

# =========================================================
# Compute V_pi by enumerating all deterministic policies
# =========================================================
# Build P_pi(s,s') = sum_a pi(s,a) P(s,s',a)
P_pi = np.einsum("sja,sa->sj", P, pi)   # (S,S), j is s'

# Build R_pi(s) = sum_a pi(s,a) R(s,a)
R_pi = (pi * R).sum(axis=1)             # (S,)

# Exact policy evaluation: (I - gamma P_pi) V_pi = R_pi
V_pi = np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)   # (S,)

# =========================================================
# Compute V_star and Q_star by enumerating all deterministic policies
# =========================================================
numPolicies = A ** S
bestJ = -np.inf
V_star = np.zeros(S)
pi_star_actions = np.ones(S, dtype=int)

for idx in range(numPolicies):
    # Decode idx into base-A digits -> actions per state (0..A-1)
    tmp = idx
    actions = np.zeros(S, dtype=int)
    for s in range(S):
        actions[s] = tmp % A
        tmp //= A

    # Build P_pi and R_pi
    temp_P_pi = np.zeros((S, S))
    temp_R_pi = np.zeros(S)
    for s in range(S):
        a = actions[s]
        temp_P_pi[s, :] = P[s, :, a]
        temp_R_pi[s] = R[s, a]

    # Exact policy evaluation
    temp_V_pi = np.linalg.solve(
        np.eye(S) - gamma * temp_P_pi,
        temp_R_pi
    )

    J = temp_V_pi.sum()

    if J > bestJ:
        bestJ = J
        V_star = temp_V_pi
        pi_star_actions = actions.copy()

# given V_star (shape S)
Q_star = R + gamma * np.einsum("ija,j->ia", P, V_star)   # (S,A)


# =========================================================
# Q-learning (learn Q_star)
# =========================================================
Q = np.zeros((S, A))
errQ = np.zeros(maxIter)

rng = np.random.default_rng(0)

epsilon0 = 1   # exploration rate (epsilon greedy)
alpha0 = 0.01     # stepsize

s = rng.integers(S)  # initial state

for k in range(maxIter):
    # # epsilon schedule (optional)
    epsilon = max(0.01, epsilon0 / (1.0 + 0.0001 * k))
    # alpha = alpha0 / (1.0 + 0.0001 * k)
    # epsilon = epsilon0
    alpha = alpha0

    # epsilon-greedy action selection
    if rng.random() < epsilon:
        a = rng.integers(A)
    else:
        a = int(np.argmax(Q[s, :]))

    # sample next state and reward from the MDP
    s_next = rng.choice(S, p=P[s, :, a])
    r = R[s, a]

    # Q-learning TD target
    td_target = r + gamma * np.max(Q[s_next, :])
    td_err = td_target - Q[s, a]

    # update
    Q[s, a] += alpha * td_err

    # log error to Q_star
    errQ[k] = np.linalg.norm(Q - Q_star, ord="fro")

    # advance
    s = s_next

# Plot Q-learning error (log scale)
plt.figure()
plt.semilogy(np.arange(maxIter), np.maximum(errQ, 1e-300), linewidth=2)
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.xlabel("Iteration k")
plt.ylabel(r"$\|Q_k - Q_\star\|_F$ (log scale)")
plt.title("Q-learning: Convergence to $Q_\\star$")
plt.tight_layout()
plt.savefig("Q_learning.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure: Q_learning.png")
