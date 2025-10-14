import numpy as np
import matplotlib.pyplot as plt

# ----- Parameters -----
n_bandits = 2000
n_actions = 10
n_steps = 1000
epsilons = [0, 0.1, 0.01]  # greedy, eps=0.1, eps=0.01

# ----- Helper function: run epsilon-greedy -----
def run_bandit(epsilon, n_bandits, n_actions, n_steps):
    # True action values (q*): one set per bandit
    q_true = np.random.randn(n_bandits, n_actions)
    
    # Estimated action values (Q) and counts (N)
    Q = np.zeros((n_bandits, n_actions))
    N = np.zeros((n_bandits, n_actions))
    
    rewards = np.zeros((n_bandits, n_steps))
    optimal_action_counts = np.zeros((n_bandits, n_steps))
    
    for t in range(n_steps):
        # Epsilon-greedy action selection
        explore = np.random.rand(n_bandits) < epsilon
        greedy_actions = np.argmax(Q, axis=1)
        random_actions = np.random.randint(0, n_actions, n_bandits)
        actions = np.where(explore, random_actions, greedy_actions)
        
        # Compute rewards: true mean + Gaussian noise
        rewards_t = np.random.randn(n_bandits) + q_true[np.arange(n_bandits), actions]
        rewards[:, t] = rewards_t
        
        # Track optimal action selection
        optimal_actions = np.argmax(q_true, axis=1)
        optimal_action_counts[:, t] = (actions == optimal_actions)
        
        # Incremental update of Q estimates
        N[np.arange(n_bandits), actions] += 1
        alpha = 1 / N[np.arange(n_bandits), actions]
        Q[np.arange(n_bandits), actions] += alpha * (rewards_t - Q[np.arange(n_bandits), actions])
    
    # Average results over all bandits
    return rewards.mean(axis=0), optimal_action_counts.mean(axis=0)

# ----- Run experiments -----
results = {}
for eps in epsilons:
    avg_reward, optimal_action_rate = run_bandit(eps, n_bandits, n_actions, n_steps)
    results[eps] = (avg_reward, optimal_action_rate)

# ----- Plot results -----
plt.figure(figsize=(12, 5))
for eps, (avg_reward, _) in results.items():
    plt.plot(avg_reward, label=f"ε={eps}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average reward over time (10-armed testbed)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
for eps, (_, opt_rate) in results.items():
    plt.plot(100 * opt_rate, label=f"ε={eps}")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Percentage of optimal action (10-armed testbed)")
plt.legend()
plt.grid(True)
plt.show()
