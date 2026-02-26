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
maxIter = 1000

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
# TD(lambda) for lambda = 0, 0.3,0.5,0.7,1
# =========================================================
lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
alpha0 = 0.2
rng = np.random.default_rng(0)
V_k = np.zeros((S,len(lambdas)))
errTD = np.zeros((maxIter,len(lambdas)))

for i in range(len(lambdas)):
    lam = lambdas[i]
    V_k = np.zeros(S)
    e = np.zeros(S)  # eligibility trace
    s = rng.integers(S)  # initial state (fixed per lambda run)
    for k in range(maxIter):
        # sample action from pi(s,:)
        a = rng.choice(A, p=pi[s])

        # sample next state from P(s,:,a)
        s_next = rng.choice(S, p=P[s, :, a])

        # reward
        r = R[s, a]

        # TD error
        delta = r + gamma * V_k[s_next] - V_k[s]

        # stepsize schedule (stable default; adjust if needed)
        # alpha = alpha0 / (alpha0 + k)
        alpha = alpha0

        # update eligibility traces (accumulating traces)
        e *= gamma * lam
        e[s] += 1.0

        # update value function
        V_k += alpha * delta * e

        # error to true V_pi
        errTD[k,i] = np.linalg.norm(V_k - V_pi, 2)

        # advance
        s = s_next


# Plot all lambdas
plt.figure(figsize=(6.5, 4.0), dpi=150)
for i in range(len(lambdas)):
    y = np.maximum(errTD[:,i], 1e-300)   # avoid log(0)
    plt.semilogy(np.arange(maxIter), y, linewidth=2, label=f"lambda={lambdas[i]}")

plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.xlabel("Iteration k")
plt.ylabel(r"$\|V_k - V_\pi\|_2$ (log scale)")
plt.title("TD($\\lambda$) Policy Evaluation")
plt.legend()
plt.tight_layout()
plt.savefig("TD_lambda.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure: TD_lambda.png")