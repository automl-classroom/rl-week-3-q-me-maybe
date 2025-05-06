import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_3 import EpsilonGreedyPolicy
from sarsa import SARSAAgent

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
env.metadata["render_fps"] = 120
policy = EpsilonGreedyPolicy(env, epsilon=0.4)
agent = SARSAAgent(env, policy, alpha=0.4, gamma=0.99)

n_episodes = 1000
max_steps = 100
rewards = []

# Training
for episode in range(n_episodes):
    # print(f"Episode: {episode}, Q: {dict(agent.Q)}")
    state, _ = env.reset()
    action = agent.predict_action(state)
    total_reward = 0
    print(f"Episode: {episode}")
    for _ in range(max_steps):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        next_action = agent.predict_action(next_state)
        agent.update_agent(state, action, reward, next_state, next_action, done)

        if done:
            break

        state, action = next_state, next_action
    if total_reward > 0:
        print(f"Episode {episode}: Ziel erreicht!")

    rewards.append(total_reward)

print(dict(agent.Q))
window = 50
avg_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")

plt.figure(figsize=(10, 5))
plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel(f"Gleitender Durchschnitt der Rewards (über {window} Episoden)")
plt.title("SARSA-Lernen auf FrozenLake-v1")
plt.grid(True)
plt.tight_layout()
plt.show()
