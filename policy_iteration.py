import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def policy_eval(policy, env, discount_factor=0.9, theta=1e-6):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=0.9):
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        policy_stable = True
        for s in range(env.observation_space.n):
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.action_space.n)[best_a]
        if policy_stable:
            return policy, V

def run(episodes, render=False):

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    policy, V = policy_improvement(env)
    
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = np.argmax(policy[state])
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake_policy_iteration.png')

    f = open("frozen_lake_policy_iteration_policy.pkl", "wb")
    pickle.dump(policy, f)
    f.close()

if __name__ == '__main__':
    run(1000, render=True)
