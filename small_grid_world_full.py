from dataclasses import dataclass
from time import sleep

@dataclass
class Coordinate:
    x: int
    y: int

def step(state, action):
    x, y = S[state].x, S[state].y
    if action == 'U' and x > 0: x -= 1
    elif action == 'D' and x < DIM-1: x += 1
    elif action == 'L' and y > 0: y -= 1
    elif action == 'R' and y < DIM-1: y += 1
    return x*DIM + y

def state_is_terminal(state):
    if state in (0, 15):
        return True
    return False

DIM = 4

# Number of states
N = DIM**2

# States
S = {}

# Map state to coordinates
s = 0
for i in range(DIM):
    for j in range(DIM):
        S[s] = Coordinate(i, j)
        s += 1          

V = [0.0]*N

A = ('R', 'L', 'U', 'D')
reward = -1

gamma = 1


theta = 1e-4

policy = {s: {a: 0.0 if state_is_terminal(s) else 1/4 for a in A} for s in range(N)}

while True:

    # Policy evaluation
    while True:
        delta = 0.0
        for s in S:
            if (state_is_terminal(s)):
                continue
            v = V[s]
            expected_return = 0.0
            for a, pi_s in policy[s].items():
                s_prime = step(s, a)
                expected_return += pi_s * 1 * (reward + gamma*V[s_prime]) # Deterministic transition probability and reward
            V[s] = expected_return
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # Policy improvement
    policy_stable = True
    for s in S:
        if state_is_terminal(s):
            continue
        old_a = max(policy[s], key=policy[s].get) if sum(policy[s].values()) > 0 else None
        qvals = {a: 1 * (reward + gamma * V[step(s, a)]) for a in A}
        best_a = max(qvals, key=qvals.get)

        if (best_a != old_a):
            policy_stable = False
        
        for a in A:
            policy[s][a] = 1.0 if a == best_a else 0.0
    if policy_stable:
        break

print(policy)
print(V)
