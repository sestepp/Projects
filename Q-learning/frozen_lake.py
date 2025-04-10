import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def create_or_load_map(is_training, map_file='frozen_lake_map8x8.pkl'):
    if is_training or not os.path.exists(map_file):
        random_map = generate_random_map(size=8)
        with open(map_file, 'wb') as f:
            pickle.dump(random_map, f)
    else:
        with open(map_file, 'rb') as f:
            random_map = pickle.load(f)
    return random_map

def initialize_q_table(env, is_training):
    if is_training:
        return np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.pkl', 'rb') as f:
            return pickle.load(f)
        
def train_agent(env, q, episodes, is_training, learning_rate_a, discount_factor_g, epsilon, epsilon_decay_rate, rng, rewards_per_episode):
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

def save_q_table(q, filename="frozen_lake8x8.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(q, f)

def run(episodes, is_training=True, render=False):
    random_map = create_or_load_map(is_training)
    env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=False, render_mode='human' if render else None)
    q = initialize_q_table(env, is_training)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    train_agent(env, q, episodes, is_training, learning_rate_a, discount_factor_g, epsilon, epsilon_decay_rate, rng, rewards_per_episode)
    env.close()
    plot_rewards(episodes, rewards_per_episode)

    if is_training:
        save_q_table(q)

def plot_rewards(episodes, rewards_per_episode):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards in Last 100 Episodes')
    plt.title('Performance of Q-Learning on FrozenLake 8x8')
    plt.savefig('frozen_lake8x8.png')

if __name__ == '__main__':
    # run(15000)
    run(1, is_training=False, render=True)
