#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment driver for the report.

usage: python Experiment.py [exploration | on-off-policy | depth | all]

Each subcommand reproduces one figure:

  exploration     e-greedy vs softmax exploration      -> exploration.png
  on-off-policy   Q-learning vs SARSA, three alphas    -> on_off_policy.png
  depth           n-step Q-learning vs Monte Carlo     -> depth.png
  all             all three, in order

Skeleton for the course 'Reinforcement Learning', Leiden University,
by Thomas Moerland; the experiment bodies are our own.
"""
import sys
import time

import numpy as np

from Helper import LearningCurvePlot, smooth
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Q_learning import q_learning
from SARSA import sarsa

# Optimal average reward per timestep under the DP solution, taken from the
# Q-value-iteration run in DynamicProgramming.py. Plotted as the ceiling that
# the learning agents are working towards.
OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP = 1.3

# Nice labels for plotting
BACKUP_LABELS = {'q': 'Q-learning',
                 'sarsa': 'SARSA',
                 'mc': 'Monte Carlo',
                 'nstep': 'n-step Q-learning'}

# Shared settings. The report averaged 50 repetitions of 50k timesteps per
# setting, which takes a long while; override with --repetitions to sanity
# check a figure quickly.
N_REPETITIONS = 50
N_TIMESTEPS = 50000
SMOOTHING_WINDOW = 1001
MAX_EPISODE_LENGTH = 150
GAMMA = 1.0
PLOT = False  # per-step rendering is very slow, keep it off across repetitions


def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy',
                             epsilon=None, temp=None, smoothing_window=51, plot=False, n=5):

    reward_results = np.empty([n_repetitions, n_timesteps])  # Result array
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        print('Repetition {}'.format(rep))
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate,
                                 gamma, policy, epsilon, temp, plot)
        elif backup == 'sarsa':
            rewards = sarsa(n_timesteps, learning_rate, gamma,
                            policy, epsilon, temp, plot)
        elif backup == 'mc':
            rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                                  policy, epsilon, temp, plot)
        elif backup == 'nstep':
            rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                               policy, epsilon, temp, plot, n=n)
        else:
            raise ValueError(
                f"unknown backup {backup!r}, expected one of {sorted(BACKUP_LABELS)}")

        reward_results[rep] = rewards

    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    # average over repetitions
    learning_curve = np.mean(reward_results, axis=0)
    # additional smoothing
    learning_curve = smooth(learning_curve, smoothing_window)
    return learning_curve


def curve(backup, learning_rate, policy='egreedy', epsilon=None, temp=None,
          n=5, n_repetitions=N_REPETITIONS):
    """average_over_repetitions with the settings shared by every experiment."""
    return average_over_repetitions(
        backup, n_repetitions, N_TIMESTEPS, MAX_EPISODE_LENGTH, learning_rate,
        GAMMA, policy, epsilon, temp, SMOOTHING_WINDOW, PLOT, n)


def exploration(n_repetitions=N_REPETITIONS):
    """Assignment 2: effect of exploration."""
    plot = LearningCurvePlot(
        title=r'Exploration: $\epsilon$-greedy versus softmax exploration')

    for epsilon in [0.02, 0.1, 0.3]:
        plot.add_curve(
            curve('q', 0.25, 'egreedy', epsilon=epsilon, n_repetitions=n_repetitions),
            label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))

    for temp in [0.01, 0.1, 1.0]:
        plot.add_curve(
            curve('q', 0.25, 'softmax', temp=temp, n_repetitions=n_repetitions),
            label=r'softmax, $ \tau $ = {}'.format(temp))

    plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")
    plot.save('exploration.png')


def on_off_policy(n_repetitions=N_REPETITIONS):
    """Assignment 3: Q-learning versus SARSA."""
    plot = LearningCurvePlot(title='Back-up: on-policy versus off-policy')

    for backup in ['q', 'sarsa']:
        for learning_rate in [0.02, 0.1, 0.4]:
            plot.add_curve(
                curve(backup, learning_rate, 'egreedy', epsilon=0.1,
                      n_repetitions=n_repetitions),
                label=r'{}, $\alpha$ = {} '.format(
                    BACKUP_LABELS[backup], learning_rate))

    plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")
    plot.save('on_off_policy.png')


def depth(n_repetitions=N_REPETITIONS):
    """Assignment 4: back-up depth."""
    plot = LearningCurvePlot(title='Back-up: depth')

    for n in [1, 3, 10, 30]:
        plot.add_curve(
            curve('nstep', 0.25, 'egreedy', epsilon=0.1, n=n,
                  n_repetitions=n_repetitions),
            label=r'{}-step Q-learning'.format(n))

    plot.add_curve(
        curve('mc', 0.25, 'egreedy', epsilon=0.1, n_repetitions=n_repetitions),
        label='Monte Carlo')

    plot.add_hline(OPTIMAL_AVERAGE_REWARD_PER_TIMESTEP, label="DP optimum")
    plot.save('depth.png')


EXPERIMENTS = {'exploration': exploration,
               'on-off-policy': on_off_policy,
               'depth': depth}


def main(args):
    usage = __doc__.split("usage: ", 1)[1].split("\n", 1)[0]

    reps = N_REPETITIONS
    if '--repetitions' in args:
        i = args.index('--repetitions')
        if i + 1 >= len(args):
            raise SystemExit("--repetitions needs a value")
        reps = int(args[i + 1])
        args = args[:i] + args[i + 2:]

    if len(args) != 1:
        raise SystemExit(f"usage: {usage}")

    name = args[0]
    if name == 'all':
        chosen = list(EXPERIMENTS.values())
    elif name in EXPERIMENTS:
        chosen = [EXPERIMENTS[name]]
    else:
        raise SystemExit(f"unknown experiment {name!r}\n\nusage: {usage}")

    for experiment in chosen:
        print(f"\n=== {experiment.__name__} ({reps} repetitions) ===")
        experiment(n_repetitions=reps)


if __name__ == '__main__':
    main(sys.argv[1:])
