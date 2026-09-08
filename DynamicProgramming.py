#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q-value iteration on the Stochastic Windy Gridworld (assignment 1).

usage: python DynamicProgramming.py [--repetitions N] [--save-figures] [--render]

Solves the MDP exactly using the environment's transition and reward model,
then rolls out the greedy policy to measure the average reward per timestep.
That figure is the "DP optimum" line in the learning-curve plots produced by
Experiment.py.

  --repetitions N   average the greedy rollout over N episodes (default 1)
  --save-figures    write step_<n>.png at iterations 0, 8 and convergence
  --render          show the value estimates during iteration (slow)

Skeleton for the course 'Reinforcement Learning', Leiden University,
by Thomas Moerland; the Q-value-iteration implementation is our own.
"""
import sys

import numpy as np

from Environment import StochasticWindyGridworld

# Iterations at which the report's intermediate figures were captured.
FIGURE_ITERATIONS = (0, 8)

# The greedy rollout is bounded so a non-terminating policy cannot hang the
# script; the converged policy reaches the goal in far fewer steps.
MAX_ROLLOUT_STEPS = 10000


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma):
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


def Q_value_iteration(env, gamma=1.0, threshold=0.001, save_figures=False,
                      render=False):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    count = 0
    while True:

        max_error = 0  # Reset max error

        for s in range(env.n_states):
            for a in range(env.n_actions):

                x = QIagent.Q_sa[s, a]

                QIagent.update(s, a, env.p_sas, env.r_sas)

                max_error = max(max_error, abs(x - QIagent.Q_sa[s, a]))

        if (render or save_figures) and count in FIGURE_ITERATIONS:
            env.render(Q_sa=QIagent.Q_sa,
                       plot_optimal_policy=True, step_pause=0.001)
            if save_figures:
                env.saveFig('step_' + str(count))

        count += 1
        if max_error < threshold:
            break

    if render or save_figures:
        env.render(Q_sa=QIagent.Q_sa,
                   plot_optimal_policy=True, step_pause=0.001)
        if save_figures:
            env.saveFig('step_' + str(count))

    print(f"Q-value iteration converged after {count} iterations "
          f"(threshold {threshold})")
    return QIagent


def greedy_rollout(env, QIagent):
    """Runs one episode of the greedy policy; returns its mean reward per step."""
    rewards = []
    s = env.reset()

    for _ in range(MAX_ROLLOUT_STEPS):
        a = QIagent.select_action(s)
        s, r, done = env.step(a)
        rewards.append(r)
        if done:
            break
    else:
        print(f"warning: rollout did not terminate within "
              f"{MAX_ROLLOUT_STEPS} steps")

    return np.sum(rewards) / len(rewards)


def experiment(repetitions=1, save_figures=False, render=False):
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    if render or save_figures:
        env.render()

    QIagent = Q_value_iteration(env, gamma, threshold,
                                save_figures=save_figures, render=render)

    # Optimal value of the start state, straight out of the converged Q table.
    # Ask the environment which state that is rather than hardcoding the index.
    V_s = np.max(QIagent.Q_sa, axis=1)
    start_state = env.reset()
    print(f"V*(start state {start_state}) = {V_s[start_state]:.3f}")

    # The environment's wind is stochastic, so average the greedy rollout to
    # get a stable estimate. This is the number Experiment.py plots as the
    # "DP optimum" horizontal line.
    means = [greedy_rollout(env, QIagent) for _ in range(repetitions)]
    mean_reward_per_timestep = float(np.mean(means))

    print(f"Mean reward per timestep under the optimal policy "
          f"({repetitions} episode{'s' if repetitions != 1 else ''}): "
          f"{mean_reward_per_timestep:.3f}")

    return mean_reward_per_timestep


def main(args):
    usage = __doc__.split("usage: ", 1)[1].split("\n", 1)[0]

    repetitions, save_figures, render = 1, False, False
    i = 0
    while i < len(args):
        if args[i] == '--repetitions':
            if i + 1 >= len(args):
                raise SystemExit("--repetitions needs a value")
            repetitions = int(args[i + 1])
            i += 2
        elif args[i] == '--save-figures':
            save_figures = True
            i += 1
        elif args[i] == '--render':
            render = True
            i += 1
        elif args[i] in ('-h', '--help'):
            raise SystemExit(__doc__)
        else:
            raise SystemExit(f"unknown flag {args[i]!r}\n\nusage: {usage}")

    experiment(repetitions=repetitions, save_figures=save_figures,
               render=render)


if __name__ == '__main__':
    main(sys.argv[1:])
