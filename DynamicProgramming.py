#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):  # , p_sas=None, r_sas=None
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''

        a = np.argmax(self.Q_sa[s, :])

        return a

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        sum_ = 0

        for s_prime in range(self.n_states):

            sum_ += (p_sas[s, a, s_prime]*(r_sas[s, a, s_prime] +
                     self.gamma * np.max(self.Q_sa[s_prime])))

        self.Q_sa[s, a] = sum_


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE

    while True:

        max_error = 0  # Reset max error

        for s in range(env.n_states):
            for a in range(env.n_actions):

                x = QIagent.Q_sa[s, a]

                QIagent.update(s, a, env.p_sas, env.r_sas)

                max_error = max(max_error, abs(x - QIagent.Q_sa[s, a]))

        # print("Max error is getting lower and lower: ", max_error)
        if max_error < threshold:
            break

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    # env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)

    # View optimal policy
    V_s = np.max(QIagent.Q_sa[:], axis=1)

    optimal_value_start_state = V_s[3]

    done = False
    s = env.reset()

    rewards = []

    while not done:
        a = QIagent.select_action(s)

        s_next, r, done = env.step(a)
        # env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        rewards.append(r)
        s = s_next

    mean_reward_per_timestep = np.sum(rewards) / len(rewards)

    # print("Mean reward per timestep under optimal policy: {}".format(
    #     mean_reward_per_timestep))

    return mean_reward_per_timestep


if __name__ == '__main__':
    mean_rewards_per_run = []
    for i in range(50):
        mean_reward_per_timestep = experiment()
        mean_rewards_per_run.append(mean_reward_per_timestep)

    mean_rewards_per_run = np.sum(
        mean_rewards_per_run) / len(mean_rewards_per_run)
    print(mean_rewards_per_run)
