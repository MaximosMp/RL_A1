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
from Helper import softmax


class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.uniform(0, 1) > epsilon:
                # NOTE: this is np.argmax, not Helper.argmax, so ties are broken
                # towards the lowest action index instead of at random. See
                # "Known issues" in the README.
                a = np.argmax(self.Q_sa[s])
            else:
                a = np.random.choice(self.n_actions)

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            a = np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))

        return a

    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # states holds one more entry than actions/rewards: the state the
        # episode ended in. T_ep is the number of steps actually taken.
        T_ep = len(states) - 1

        for t in range(T_ep):
            G = 0
            m = min(self.n, T_ep - t)

            for i in range(m):
                G += self.gamma**i * rewards[t + i]

            # Bootstrap unless the n-step window lands on a terminal state.
            # Behaviour-preserving: terminal rows of Q_sa are never updated and
            # so stay zero, making the old wider condition add 0 anyway.
            if not (done and t + m == T_ep):
                G += self.gamma**m * np.max(self.Q_sa[states[t + m]][:])

            self.Q_sa[states[t]][actions[t]] = self.Q_sa[states[t]][actions[t]] \
                + self.learning_rate * (G - self.Q_sa[states[t]][actions[t]])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(
        env.n_states, env.n_actions, learning_rate, gamma, n)

    rewards = []
    index = 0
    while index < n_timesteps:
        # for _ in range(n_timesteps):
        rewards_per_episode = []
        states = []
        actions = []

        s = env.reset()
        states.append(s)
        done = False
        for t in range(max_episode_length):
            index += 1
            a = pi.select_action(s, policy, epsilon, temp)
            actions.append(a)

            s_next, r, done = env.step(a)
            states.append(s_next)
            rewards_per_episode.append(r)
            rewards.append(r)
            s = s_next
            if done or index >= n_timesteps:
                break

        if plot:
            # Plot the Q-value estimates during n-step Q-learning execution
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                       step_pause=0.01)

        pi.update(states, actions, rewards_per_episode, done)
    return rewards


def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                       policy, epsilon, temp, plot, n=n)
    print("Mean reward per timestep: {:.3f}".format(np.mean(rewards)))


if __name__ == '__main__':
    test()
