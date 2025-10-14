import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------
# Parameters (edit freely)
# -------------------------
K = 20                 # capacity per location (0..K)
MAX_MOVE = 5           # nightly transfer cap |a| <= MAX_MOVE
RENTAL_REWARD = 10.0   # revenue per rental
MOVE_COST = 2.0        # cost per car moved
GAMMA = 0.9            # discount factor

# Poisson means (your variant)
LAMBDA_RENT = (3.0, 4.0)  # (loc1, loc2) rentals
LAMBDA_RET  = (3.0, 2.0)  # (loc1, loc2) returns

# Poisson truncation (for tails)
# Big enough that residual tail mass ≪ 1e-8. Using K + margin is typically fine.
PMF_MARGIN = 10
CUTS = (
    int(max(K + PMF_MARGIN, LAMBDA_RENT[0] + 10 * math.sqrt(LAMBDA_RENT[0]))),
    int(max(K + PMF_MARGIN, LAMBDA_RENT[1] + 10 * math.sqrt(LAMBDA_RENT[1]))),
    int(max(K + PMF_MARGIN, LAMBDA_RET[0]  + 10 * math.sqrt(LAMBDA_RET[0]))),
    int(max(K + PMF_MARGIN, LAMBDA_RET[1]  + 10 * math.sqrt(LAMBDA_RET[1]))),
)

THETA_EVAL = 1e-3      # policy evaluation stopping criterion
MAX_EVAL_ITERS = 10_000
MAX_POLICY_ITERS = 200

# -------------------------
# Utilities
# -------------------------

def poisson_pmf(lam, n_max):
    """ Poisson pmf truncated to 0..n_max, last bin absorbs the tail. """
    ns = np.arange(n_max + 1, dtype=int)
    # stable recursive pmf build
    pmf = np.zeros(n_max + 1, dtype=float)
    pmf[0] = math.exp(-lam)
    for k in range(1, n_max):
        pmf[k] = pmf[k-1] * lam / k
    tail = 1.0 - pmf[:n_max].sum()
    pmf[n_max] = max(tail, 0.0)
    # guard tiny negatives from numerical noise
    pmf = np.clip(pmf, 0.0, 1.0)
    pmf /= pmf.sum()
    return pmf

def precompute_single_loc(K, lam_d, lam_r, cut_d, cut_r):
    """
    Precompute, for each starting inventory n in 0..K:
      - P(n' | n): (K+1)-length vector over next inventory after rentals D ~ Pois(lam_d),
        returns R ~ Pois(lam_r), with capping at K.
      - E[min(D, n)]: expected rentals from starting inventory n.
    Transition logic per location:
      rentals = min(D, n)
      next = min(n - rentals + R, K)
    """
    pmf_d = poisson_pmf(lam_d, cut_d)
    pmf_r = poisson_pmf(lam_r, cut_r)

    # Precompute E[min(D,n)] efficiently
    # E[min(D, n)] = sum_{d=0}^{n-1} d * P(D=d) + n * P(D>=n)
    cdf_d = np.cumsum(pmf_d)
    # partial sums of d*P(D=d)
    d_vals = np.arange(len(pmf_d))
    s1 = np.cumsum(d_vals * pmf_d)

    E_min = np.zeros(K + 1, dtype=float)
    for n in range(K + 1):
        if n == 0:
            E_min[n] = 0.0
        else:
            n_clip = min(n, len(pmf_d)-1)
            sum_d_less = s1[n_clip-1] if n_clip-1 >= 0 else 0.0
            tail = 1.0 - cdf_d[n_clip-1] if n_clip-1 >= 0 else 1.0
            E_min[n] = sum_d_less + n * tail

    # Transition matrix T[n, n_next]
    T = np.zeros((K + 1, K + 1), dtype=float)
    for n in range(K + 1):
        # iterate over truncated supports
        for d, p_d in enumerate(pmf_d):
            rentals = min(d, n)
            remaining = n - rentals  # >= 0
            for r, p_r in enumerate(pmf_r):
                nxt = remaining + r
                if nxt > K:
                    nxt = K
                T[n, nxt] += p_d * p_r

        # normalization guard (should already be 1)
        if not np.isclose(T[n].sum(), 1.0, rtol=1e-10, atol=1e-12):
            # renormalize very slightly if needed due to tail clipping/float
            T[n] /= T[n].sum()

    return T, E_min

# -------------------------
# Build single-location tables
# -------------------------

T1, Emin1 = precompute_single_loc(K, LAMBDA_RENT[0], LAMBDA_RET[0], CUTS[0], CUTS[2])
T2, Emin2 = precompute_single_loc(K, LAMBDA_RENT[1], LAMBDA_RET[1], CUTS[1], CUTS[3])

# -------------------------
# State indexing helpers
# -------------------------

# States: (i, j) for i,j in 0..K
all_states = [(i, j) for i in range(K+1) for j in range(K+1)]
state_to_idx = {s: idx for idx, s in enumerate(all_states)}
N_STATES = len(all_states)

def feasible_actions(i, j):
    """ Actions a: cars moved 1->2 (negative = 2->1). """
    a_min = -min(j, MAX_MOVE)
    a_max =  min(i, MAX_MOVE)
    return range(a_min, a_max + 1)

# -------------------------
# Expected immediate reward and next-state distribution
# -------------------------

def expected_reward_and_next(i, j, a, V=None):
    """
    Given state (i, j) and action a, compute:
      - expected immediate reward: RENTAL_REWARD * (E[min(D1, i')] + E[min(D2, j')]) - MOVE_COST*|a|
      - (if V is None) return the (s', prob) vector for next states;
        else return expected next value = sum_{s'} P * V[s']
    Uses precomputed single-location transitions and outer product.
    """
    i_post = i - a
    j_post = j + a
    # bounds safety
    i_post = max(0, min(K, i_post))
    j_post = max(0, min(K, j_post))

    # Expected rentals
    r_exp = RENTAL_REWARD * (Emin1[i_post] + Emin2[j_post])
    move_penalty = MOVE_COST * abs(a)
    immediate = r_exp - move_penalty

    # Next-state distribution = outer product of T1[i_post] and T2[j_post]
    p_i = T1[i_post]  # shape (K+1,)
    p_j = T2[j_post]  # shape (K+1,)
    # If V is not provided, return the distribution explicitly
    if V is None:
        probs = np.outer(p_i, p_j)  # shape (K+1, K+1)
        # flatten to match state index order (i' major, then j')
        probs_flat = probs.reshape(-1)
        return immediate, probs_flat
    else:
        # compute expected next value quickly without forming full probs matrix
        # sum_{i'} sum_{j'} p_i[i'] * p_j[j'] * V[i', j']
        # reshape V to (K+1, K+1)
        V_grid = V.reshape((K+1, K+1))
        # (p_i @ V_grid @ p_j)
        exp_next = p_i @ V_grid @ p_j
        return immediate, exp_next

# -------------------------
# Policy iteration
# -------------------------

# Initialize policy to "do nothing" (a=0) where feasible, else nearest feasible
policy = np.zeros(N_STATES, dtype=int)
for idx, (i, j) in enumerate(all_states):
    a0 = 0
    if a0 < -min(j, MAX_MOVE):
        a0 = -min(j, MAX_MOVE)
    if a0 >  min(i, MAX_MOVE):
        a0 =  min(i, MAX_MOVE)
    policy[idx] = a0

V = np.zeros(N_STATES, dtype=float)

stable = False
for it_pol in range(1, MAX_POLICY_ITERS + 1):
    # --- Policy evaluation ---
    for it_eval in range(MAX_EVAL_ITERS):
        delta = 0.0
        for s_idx, (i, j) in enumerate(all_states):
            a = policy[s_idx]
            imm, exp_next = expected_reward_and_next(i, j, a, V=V)
            v_new = imm + GAMMA * exp_next
            delta = max(delta, abs(v_new - V[s_idx]))
            V[s_idx] = v_new
        if delta < THETA_EVAL:
            break

    # --- Policy improvement ---
    policy_stable = True
    for s_idx, (i, j) in enumerate(all_states):
        old_a = policy[s_idx]
        # evaluate all feasible actions
        best_a = None
        best_q = -1e100
        for a in feasible_actions(i, j):
            imm, exp_next = expected_reward_and_next(i, j, a, V=V)
            q = imm + GAMMA * exp_next
            if q > best_q:
                best_q = q
                best_a = a
        policy[s_idx] = best_a
        if best_a != old_a:
            policy_stable = False

    print(f"Policy iter {it_pol}: policy_stable={policy_stable}")
    if policy_stable:
        break

# -------------------------
# Visualization
# -------------------------

# reshape to grids
V_grid = V.reshape((K+1, K+1))
Pi_grid = policy.reshape((K+1, K+1))  # a*(i,j)

# Plot Value function heatmap
plt.figure(figsize=(7, 6))
plt.title("Optimal Value Function V*(i,j)")
plt.xlabel("Cars at Location 2 (j)")
plt.ylabel("Cars at Location 1 (i)")
plt.imshow(V_grid, origin='lower', aspect='auto')
plt.colorbar(label='Value')
plt.tight_layout()
plt.show()

# Plot Policy heatmap (action = cars moved 1->2; negative means 2->1)
plt.figure(figsize=(7, 6))
plt.title("Optimal Policy π*(i,j): cars moved 1→2")
plt.xlabel("Cars at Location 2 (j)")
plt.ylabel("Cars at Location 1 (i)")
plt.imshow(Pi_grid, origin='lower', aspect='auto')
plt.colorbar(label='cars moved 1→2')
plt.tight_layout()
plt.show()

# Print a small slice for sanity
print("\nSample of optimal moves (i from 0..K, j from 0..K):")
for i in range(0, K+1, 5):
    row = [f"{Pi_grid[i,j]:+d}" for j in range(0, K+1, 5)]
    print(f"i={i:2d}: " + "  ".join(row))
