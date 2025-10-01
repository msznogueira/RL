import numpy as np
from scipy.stats import poisson

# --- Problem constants ---
K = 20                          # capacity per lot
gamma = 0.9
move_cost = 2.0
rent_reward = 10.0

# Poisson rates: requests then returns
lam_r1, lam_r2 = 3, 4
lam_b1, lam_b2 = 3, 2

# Truncation tail beyond capacity (safe & fast)
TAIL = 15
R1_MAX = K + TAIL
R2_MAX = K + TAIL
B1_MAX = K + TAIL
B2_MAX = K + TAIL
EPS = 1e-12  # ignore tiny probabilities

# --- Helpers ---
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def feasible_actions(i, j):
    lo = -min(5, j)  # negative: move from loc2->loc1
    hi =  min(5, i)  # positive: move from loc1->loc2
    return range(lo, hi + 1)

def E_min_poisson_table(K, dist):
    # Table t[n] = E[min(R, n)] for n in 0..K
    pmf = dist.pmf(np.arange(0, K+1))  # we only need up to K for this expectation
    cdf = dist.cdf(np.arange(0, K+1))
    t = np.zeros(K+1, dtype=float)
    # E[min(R,n)] = sum_{k=0}^{n-1} k*P(R=k) + n * P(R >= n)
    # compute prefix sums of k*pmf for speed
    ks = np.arange(0, K+1, dtype=float)
    kpmf_prefix = np.cumsum(ks * pmf)  # length K+1, at index n gives sum_{k<=n} k*P(k)
    for n in range(1, K+1):
        sum_k = kpmf_prefix[n-1]          # sum_{k=0}^{n-1} k*P(k)
        tail = 1.0 - cdf[n-1]             # P(R >= n)
        t[n] = sum_k + n * tail
    return t

def truncated_pmf(dist, max_k):
    arr = dist.pmf(np.arange(0, max_k+1))
    arr[arr < EPS] = 0.0
    # No renormalization: any leftover tail probability beyond max_k will be absorbed by capacity clamp later
    return arr

# --- Precompute expectations for rewards ---
R1 = poisson(mu=lam_r1)
R2 = poisson(mu=lam_r2)
Emin_R1 = E_min_poisson_table(K, R1)  # E[min(requests1, n)]
Emin_R2 = E_min_poisson_table(K, R2)  # E[min(requests2, n)]

# --- Precompute PMFs (truncated) for requests and returns ---
pmf_r1 = truncated_pmf(R1, R1_MAX)  # shape (R1_MAX+1,)
pmf_r2 = truncated_pmf(R2, R2_MAX)
pmf_b1 = truncated_pmf(poisson(mu=lam_b1), B1_MAX)
pmf_b2 = truncated_pmf(poisson(mu=lam_b2), B2_MAX)

# Precompute index lists where pmf > 0 to skip zeros
idx_r1 = np.flatnonzero(pmf_r1)
idx_r2 = np.flatnonzero(pmf_r2)
idx_b1 = np.flatnonzero(pmf_b1)
idx_b2 = np.flatnonzero(pmf_b2)

# --- Precompute 1D transition kernels T1, T2 ---
# T1[i_prime, x] = P(i_next = x | start i_prime)
# T2[j_prime, y] = P(j_next = y | start j_prime)
T1 = np.zeros((K+1, K+1), dtype=float)
T2 = np.zeros((K+1, K+1), dtype=float)

# Single-location kernel builder (requests ~ pmf_r, returns ~ pmf_b)
def build_kernel_single(K, pmf_r, idx_r, pmf_b, idx_b):
    T = np.zeros((K+1, K+1), dtype=float)
    for n in range(K+1):  # morning stock n
        row = np.zeros(K+1, dtype=float)
        for r in idx_r:
            pr = pmf_r[r]
            f = r if r <= n else n                # fulfilled rentals = min(r, n)
            remain = n - f
            for b in idx_b:
                pb = pmf_b[b]
                nxt = remain + b
                nxt = K if nxt > K else nxt       # clamp to capacity
                row[nxt] += pr * pb
        # tiny numerical drift fix: normalize row to sum <= 1 (remaining tail mass goes to K anyway)
        s = row.sum()
        if s > 0:
            row /= s
        T[n, :] = row
    return T

T1 = build_kernel_single(K, pmf_r1, idx_r1, pmf_b1, idx_b1)
T2 = build_kernel_single(K, pmf_r2, idx_r2, pmf_b2, idx_b2)

# --- Value Iteration (fast) ---
V = np.zeros((K+1, K+1), dtype=float)  # V[i, j]
theta = 1e-3

while True:
    delta = 0.0
    V_new = V.copy()

    # precompute right multiplies to reuse: for each j' we’ll need V @ T2[j'] (or its transpose)
    # We'll compute EV_next(i', j') = (T1[i'] @ V @ T2[j']^T)
    # Code uses: tmp = V @ T2[j_prime], then EV_next = T1[i_prime] @ tmp
    VT2 = V @ T2.T   # shape (K+1, K+1) where column j' gives sum_y V[:,y]*P2(y|j')

    for i in range(K+1):
        for j in range(K+1):

            best = -np.inf
            for a in feasible_actions(i, j):
                i_prime = clamp(i - a, 0, K)
                j_prime = clamp(j + a, 0, K)

                # expected immediate reward
                immediate = -move_cost * abs(a)
                immediate += rent_reward * (Emin_R1[i_prime] + Emin_R2[j_prime])

                # expected next value via bilinear form:
                # EV_next = T1[i_prime,:] @ (V @ T2[j_prime,:]^T)
                EV_next = T1[i_prime, :].dot(VT2[:, j_prime])

                val = immediate + gamma * EV_next
                if val > best:
                    best = val

            V_new[i, j] = best
            delta = max(delta, abs(best - V[i, j]))

    V = V_new
    if delta < theta:
        break

# (Optional) derive greedy policy arrows
def greedy_action(i, j, V, T1, T2):
    best_a, best_val = 0, -np.inf
    VT2 = V @ T2.T
    for a in feasible_actions(i, j):
        i_prime = clamp(i - a, 0, K)
        j_prime = clamp(j + a, 0, K)
        immediate = -move_cost * abs(a) + rent_reward * (Emin_R1[i_prime] + Emin_R2[j_prime])
        EV_next = T1[i_prime, :].dot(VT2[:, j_prime])
        val = immediate + gamma * EV_next
        if val > best_val:
            best_val, best_a = val, a
    return best_a

# Example: print a small slice of V and a few greedy actions
print(np.round(V[:6, :6], 1))
print([greedy_action(10, j, V, T1, T2) for j in range(0, 21, 5)])
