#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if (np.random.uniform(0, 1) > epsilon):
                a = argmax(self.Q_sa[s])
            else:
                a = np.random.choice(4)

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            # TO DO: Add own code
            # Replace this with correct action selection
            a = np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))

        return a

    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        Tep = len(states) - 1  # 101
        G = np.zeros(Tep + 1)  # 100

        for t in range(Tep - 1, 0, -1):

            # print(states[t])
            G[t] = self.gamma*G[t + 1] + rewards[t]

            self.Q_sa[states[t]][actions[t]] = self.Q_sa[states[t]][actions[t]] \
                + self.learning_rate * \
                (G[t] - self.Q_sa[states[t]][actions[t]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)

    rewards = []
    index = 0
    while index < n_timesteps:
        rewards_per_episode = []
        states = []
        actions = []

        s = env.reset()
        states.append(s)
        
        for t in range(max_episode_length):
            index += 1
            a = pi.select_action(s, policy, epsilon, temp)
            actions.append(a)

            s_next, r, done = env.step(a)
            rewards_per_episode.append(r)
            # rewards[_] += r
            rewards.append(r)
            states.append(s_next)
            if done or index >= n_timesteps:
                break

            s = s_next

        # rewards[_] /= t + 1

        pi.update(states, actions, rewards_per_episode)

        # if plot:
        #     # Plot the Q-value estimates during n-step Q-learning execution
        #     env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
        #                step_pause=0.001)
    return rewards


def test():
    n_timesteps = 50000
    max_episode_length = 50
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax'  # 'egreedy' or 'softmax'
    epsilon = 0.01
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                          policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards[49999]))


if __name__ == '__main__':
    test()
