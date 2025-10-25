from dataclasses import dataclass

@dataclass
class Coordinate:
    x: int
    y: int

DIM = 4
N = DIM * DIM
A = ['U', 'D', 'L', 'R']
gamma = 1.0
reward = -1.0
theta = 1e-4

S = {s: Coordinate(*divmod(s, DIM)) for s in range(N)}

def step(state, action):
    x, y = S[state].x, S[state].y
    if action == 'U' and x > 0: x -= 1
    elif action == 'D' and x < DIM-1: x += 1
    elif action == 'L' and y > 0: y -= 1
    elif action == 'R' and y < DIM-1: y += 1
    return x*DIM + y

def is_terminal(s):
    return s in (0, N-1)

# start with equiprobable random policy
pi = {s: {a: (0.0 if is_terminal(s) else 1/4) for a in A} for s in range(N)}
V = [0.0]*N

while True:
    # --- policy evaluation ---
    while True:
        delta = 0.0
        for s in range(N):
            if is_terminal(s): 
                continue
            v_old = V[s]
            expected = 0.0
            for a, p in pi[s].items():
                s_prime = step(s, a)
                expected += p * (reward + gamma * V[s_prime])
            V[s] = expected
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break

    # --- policy improvement ---
    policy_stable = True
    for s in range(N):
        if is_terminal(s): 
            continue
        # best action under current V
        qvals = {a: (reward + gamma * V[step(s, a)]) for a in A}
        best_a = max(qvals, key=qvals.get)

        # did the action distribution change?
        old_best = max(pi[s], key=pi[s].get) if sum(pi[s].values()) > 0 else None
        if old_best != best_a:
            policy_stable = False

        # make policy deterministic greedy
        for a in A:
            pi[s][a] = 1.0 if a == best_a else 0.0

    if policy_stable:
        break

# pretty-print
for i in range(DIM):
    print([round(V[i*DIM + j], 3) for j in range(DIM)])

arrow = {'U':'↑','D':'↓','L':'←','R':'→'}
for i in range(DIM):
    row = []
    for j in range(DIM):
        s = i*DIM + j
        if is_terminal(s):
            row.append('T')
        else:
            row.append(arrow[max(pi[s], key=pi[s].get)])
    print(row)
