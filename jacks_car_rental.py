from scipy.stats import poisson
import math

K = 20
states = [(i, j) for i in range(K+1) for j in range(K+1)]
state_to_idx = {s: idx for idx, s in enumerate(states)}
N_STATES = len(states)

gamma = 0.9
move_cost = 2.0
rent_reward = 10.0

# Poisson distributions
R1 = poisson(mu=3)  # requests loc1
R2 = poisson(mu=4)  # requests loc2
B1 = poisson(mu=3)  # returns  loc1
B2 = poisson(mu=2)  # returns  loc2

# truncate tails for loops
TAIL = 15  # safe; you can tune
R1_MAX = K + TAIL
R2_MAX = K + TAIL
B1_MAX = K + TAIL
B2_MAX = K + TAIL

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def feasible_actions(i, j):
    # Max 5 moved; also cannot move more than available at origin
    # We’ll clamp after applying the move, so this bound suffices:
    lo = -min(5, j)   # negative = move from loc2 -> loc1
    hi =  min(5, i)   # positive = move from loc1 -> loc2
    return range(lo, hi+1)

def E_min_poisson(n, pmf, cdf):
    # E[min(R, n)]  (works for n >= 0)
    if n <= 0: 
        return 0.0
    s = 0.0
    # sum_{k=0}^{n-1} k * P(R=k)
    for k in range(n):
        s += k * pmf(k)
    # + n * P(R >= n)
    s += n * (1.0 - cdf(n-1))
    return s

# precompute E[min(R, n)] tables
Emin_R1 = [E_min_poisson(n, R1.pmf, R1.cdf) for n in range(K+1)]
Emin_R2 = [E_min_poisson(n, R2.pmf, R2.cdf) for n in range(K+1)]

# deterministic policy per state (e.g., start with "move 0")
policy = [0 for _ in range(N_STATES)]

V = [0.0 for _ in range(N_STATES)]
theta = 1e-4

while True:
    delta = 0.0
    for idx, (i, j) in enumerate(states):
        a = policy[idx]

        # clamp inventories after moving overnight
        i_prime = clamp(i - a, 0, K)
        j_prime = clamp(j + a, 0, K)

        # expected immediate reward (movement cost + expected rental revenue)
        immediate = -move_cost * abs(a)
        immediate += rent_reward * (Emin_R1[i_prime] + Emin_R2[j_prime])

        # expected V of next state via requests & returns
        EV_next = 0.0
        # loop over requests
        for r1 in range(R1_MAX + 1):
            p_r1 = R1.pmf(r1)
            if p_r1 < 1e-12: 
                continue
            for r2 in range(R2_MAX + 1):
                p_r2 = R2.pmf(r2)
                if p_r2 < 1e-12:
                    continue

                f1 = min(r1, i_prime)  # fulfilled at loc1
                f2 = min(r2, j_prime)  # fulfilled at loc2

                # loop over returns
                for b1 in range(B1_MAX + 1):
                    p_b1 = B1.pmf(b1)
                    if p_b1 < 1e-12:
                        continue
                    for b2 in range(B2_MAX + 1):
                        p_b2 = B2.pmf(b2)
                        if p_b2 < 1e-12:
                            continue

                        i_next = clamp(i_prime - f1 + b1, 0, K)
                        j_next = clamp(j_prime - f2 + b2, 0, K)
                        p = p_r1 * p_r2 * p_b1 * p_b2
                        EV_next += p * V[state_to_idx[(i_next, j_next)]]

        v_old = V[idx]
        V[idx] = immediate + gamma * EV_next
        delta = max(delta, abs(v_old - V[idx]))
        print(delta)

    if delta < theta:
        break
