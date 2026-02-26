import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Random MDP + Value Iteration + Policy Iteration (clean)
# =========================================================

np.random.seed(0)

# -----------------------------
# Parameters (shared)
# -----------------------------
S = 4                  # number of states
A = 3                  # number of actions
gamma = 0.95           # discount factor
tol = 1e-12            # accuracy
maxIter = 500          # used for BOTH VI and PI
evalSweeps = 5         # partial evaluation sweeps per PI step

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
# Compute V_star by enumerating all deterministic policies
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

# =========================================================
# Value Iteration
# =========================================================
V_k = np.zeros(S)
errVI = np.zeros(maxIter)

for k in range(maxIter):
    V_k_next = -np.inf * np.ones(S)

    for a in range(A):
        Q_a = R[:, a] + gamma * (P[:, :, a] @ V_k)
        V_k_next = np.maximum(V_k_next, Q_a)

    errVI[k] = np.linalg.norm(V_star - V_k_next, 2)

    if np.linalg.norm(V_k_next - V_k, np.inf) < tol:
        errVI = errVI[:k+1]
        break

    V_k = V_k_next

# Plot VI
fig = plt.figure()
plt.plot(errVI,label=r"$\|\|V_k - V_\star\|\|$", linewidth=2)
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Log Scale")
plt.title("Value Iteration")
plt.yscale('log')
plt.legend()
plt.show()
fig.savefig("VI.png", dpi=300)

# =========================================================
# Policy Iteration (Modified PI)
# =========================================================
# Deterministic initialization from pi
pi_actions = np.argmax(pi, axis=1)   # shape (S,)
V_eval = np.zeros(S)
errPI = np.zeros(maxIter)

for k in range(maxIter):

    # Build P_pi_k and R_pi_k
    P_pi_k = np.zeros((S, S))
    R_pi_k = np.zeros(S)
    for s in range(S):
        a = pi_actions[s]
        P_pi_k[s, :] = P[s, :, a]
        R_pi_k[s] = R[s, a]

    # Partial policy evaluation
    for _ in range(evalSweeps):
        V_eval = R_pi_k + gamma * (P_pi_k @ V_eval)

    V_pi_k = V_eval.copy()

    # Optional exact evaluation (uncomment if desired)
    # V_pi_k = np.linalg.solve(np.eye(S) - gamma * P_pi_k, R_pi_k)

    errPI[k] = np.linalg.norm(V_pi_k - V_star, 2)

    # Policy improvement
    Q = np.zeros((S, A))
    for a in range(A):
        Q[:, a] = R[:, a] + gamma * (P[:, :, a] @ V_pi_k)

    new_actions = np.argmax(Q, axis=1)

    # Stop if error is small
    if errPI[k] < tol:
        errPI = errPI[:k+1]
        break

    pi_actions = new_actions

# Plot PI
fig = plt.figure()
plt.plot(errPI,label=r"$\|\|V_{\pi_k} - V_\star\|\|$", linewidth=2)
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Log Scale")
plt.title("Policy Iteration")
plt.yscale('log')
plt.legend()
plt.show()
fig.savefig("PI.png", dpi=300, bbox_inches='tight')

print("Finished. Saved figures: VI.png, PI.png")
