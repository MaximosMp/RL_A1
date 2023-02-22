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
        # Q-value table initialized to zeros
        self.Q_sa = np.zeros((n_states, n_actions))

       # self.p_sas = p_sas
       # self.r_sas = r_sas

      #  print("p_sas probability distribution function", self.p_sas)
      #  print("r_sas reward function", self.r_sas)
      #  print("p_sas shape", self.p_sas.shape)
        # print("r_sas shape", self.r_sas.shape)

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''
        # TO DO: Add own code
        # Returns the greedy best action in state s

        # print("those are the options", self.Q_sa[s, :])
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a = np.argmax(self.Q_sa[s, :])
        # print('s: {}, a: {}'.format(s,a))
        # print("this is the best action", a)
        return a

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        sum_ = 0

        # print(p_sas.shape, r_sas.shape)
        # print(self.Q_sa[s, a])
        for s_prime in range(self.n_states):
            # print('PRIME', self.Q_sa[s_prime])
            # print(p_sas[s, a, s_prime], r_sas[s, a, s_prime])

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

        print("Max error is getting lower and lower: ", max_error)
        if max_error < threshold:
            break

        # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=2.0)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)

    # View optimal policy
    V_s = np.max(QIagent.Q_sa[:], axis=1)

    print(V_s.shape)
    optimal_value_start_state = V_s[3]

    done = False
    s = env.reset()
    
    total_reward = 0
    num_steps = 0
    while not done:
        a = QIagent.select_action(s)

        s_next, r, done = env.step(a)
        #env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        s = s_next

        num_steps += 1
        total_reward += r
        print("total reward", total_reward)
    
    print('Total reward: {}'.format(total_reward))
    print("optimal_value_start_state", optimal_value_start_state)
    print("last reward", r)
    print("num_steps", num_steps)
    print("total_reward", total_reward)

    mean_reward_per_timestep = (40 - total_reward) / num_steps
    # print("Optimal policy:", QIagent.Q_sa)

    # TO DO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))


if __name__ == '__main__':
    experiment()
