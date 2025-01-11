# ---------
# - Setup -
# ---------

import time
import numpy as np
import numba as nb
import random
import torch
import copy
import itertools
import matplotlib.pyplot as plt

# Set seed
seed = 1998
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Imports for environment
from base_env import BaseEnv
from buffer_stock_env import BufferStockEnv

# Imports for plotting
from history import History
from show_results import *
import plot_settings

# Imports for RL
from agent import Agent
from dp_agent import DPAgent
from td3_agent import TD3Agent
from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer
from networks import Actor, Critic

# Imports for DP
from DP_solution import BufferStockModelClass
from consav.misc import elapsed

# General plot layout
plt.rcParams.update({
    "axes.grid": True,
    "grid.color": "black",
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    "font.size": 20,
    "font.family": "sans-serif",
    "pgf.texsystem": "pdflatex"
})

# ------------------------- 
# - Parameter experiments -
# -------------------------

T = 20
rho = 1.0
beta = 0.96
testing_episodes = 5000  # Number of episodes to test
training_episodes = 1000  # Number of episodes to train
shock_period = 5

# Choose which RL agent to use
# agent_class = TD3Agent
agent_class = DDPGAgent

model_params = {
    'pi': [[0.2, 0.4], '$\pi$']
}

hyper_params = {
#    'tau' : [[0.005, 0.006], '$\tau$'],
#    'batch_size' : [[512, 1028], 'Batch size'],
#    'buffer_size' : [[int(10*1e6), int(15*1e6)], 'Buffer size'],
     'ini_explore_noise' : [[0.35], 'Explore noise'],
#    'policy_noise' : [[0.2, 0.3, 0.1], 'Policy noise'],
#    'policy_delay' : [[2, 3, 1], 'Policy delay'],
#    'actor_lr' : [[1.1*1e-4, 1e-4, 0.9*1e-4], 'actor_lr']  # Vary with episodes
}

for model_key, (model_values, model_name) in model_params.items():
    agent_dicts_fig1 = []

    for hyper_key, (hyper_values, hyper_name) in hyper_params.items():

        for i, model_value in enumerate(model_values):
            agent_dicts_fig3 = []

            # Solve with DP
            par = {
                'solmethod':'egm',
                'T':T,
                'rho':rho,
                'beta':beta,
                'do_print':False
            }
            model = BufferStockModelClass(name='baseline', par=par)
            model.solve()

            for j, hyper_value in enumerate(hyper_values):
                agent_dicts_fig2 = []

                # --- BASELINE (NO SHOCK) ---

                # Initialize environment
                env = BufferStockEnv(T=T, rho=rho, beta=beta)
                env.seed(1998)

                # Initialize agents
                action_dim = env.action_space.shape[0]
                state_dim = env.observation_space.shape[0]

                # Use agent_class instead of a hard-coded TD3Agent
                rl_agent = agent_class(
                    state_dim,
                    action_dim,
                    beta=beta,
                    **{hyper_key: hyper_value}
                )
                dp_agent = DPAgent(model, env)

                # Train RL agent
                rl_agent.interact(
                    env,
                    training_episodes,
                    train=True,
                    do_print=False
                )

                # Plot policy functions
                plot_policy(rl_agent, dp_agent)

                # Simulate RL agent and DP agent
                rl_agent.interact(env, testing_episodes, train=False)
                dp_agent.interact(env, testing_episodes, train=False)

                # Save histories
                rl_agent.history.save('rl_agent.pkl')
                dp_agent.history.save('dp_agent.pkl')

                # --- APPLY SHOCK ---

                # Create new environment with shock
                ini_m_rl = np.mean(rl_agent.history.m[:,shock_period], axis=0)
                ini_p_rl = np.mean(rl_agent.history.p[:,shock_period], axis=0)

                ini_m_dp = np.mean(dp_agent.history.m[:,shock_period], axis=0)
                ini_p_dp = np.mean(dp_agent.history.p[:,shock_period], axis=0)

                state_vars_rl = [
                    {'name': 'p', 'ini': ini_p_rl, 'low': 0.0, 'high': 100},
                    {'name': 'm', 'ini': ini_m_rl, 'low': 0.0, 'high': 100},
                    {'name': 't', 'ini': shock_period/env.T, 'low': 0.0, 'high': 1.0}
                ]

                state_vars_dp = [
                    {'name': 'p', 'ini': ini_p_dp, 'low': 0.0, 'high': 100},
                    {'name': 'm', 'ini': ini_m_dp, 'low': 0.0, 'high': 100},
                    {'name': 't', 'ini': shock_period/env.T, 'low': 0.0, 'high': 1.0}
                ]

                env_change_rl = BufferStockEnv(
                    state_vars=state_vars_rl,
                    reset_period=shock_period,
                    T=T,
                    **{model_key: model_value}
                )
                env_change_dp = BufferStockEnv(
                    state_vars=state_vars_dp,
                    reset_period=shock_period,
                    T=T,
                    **{model_key: model_value}
                )

                # Solve with DP under shock
                par_shock = par.copy()
                par_shock['T'] = T - shock_period
                par_shock[model_key] = model_value
                model_shock = BufferStockModelClass(name='shock', par=par_shock)
                model_shock.solve()

                # Deep copy the RL agent
                rl_agent_shock = copy.deepcopy(rl_agent)
                dp_agent_shock = DPAgent(model_shock, env_change_dp)

                # Train RL agent in shocked environment
                rl_agent_shock.interact(env_change_rl, training_episodes, train=True, do_print=False)

                # Load histories from baseline
                dp_agent_shock.history = History.load('dp_agent.pkl')
                rl_agent_shock.history = History.load('rl_agent.pkl')

                # Simulate
                dp_agent_shock.interact(env_change_dp, testing_episodes, train=False, keep_history=True, shift=shock_period)
                rl_agent_shock.interact(env_change_rl, testing_episodes, train=False, keep_history=True)

                # --- PLOTS ---

                # Figure 1 (example)
                if j == 0:
                    plot_settings.td3_fig1['label'] = f'{model_name} = {model_value}'
                    plot_settings.td3_fig1['linestyle'] = plot_settings.linestyles_fig1[i]
                    plot_settings.td3_fig1['history'] = rl_agent_shock.history
                    agent_dicts_fig1.append(plot_settings.td3_fig1.copy())

                # Figure 2
                plot_settings.td3_shock_fig2['history'] = rl_agent_shock.history
                plot_settings.dp_shock_fig2['history'] = dp_agent_shock.history
                plot_settings.td3_fig2['history'] = rl_agent.history
                plot_settings.dp_fig2['history'] = dp_agent.history

                plot_settings.td3_shock_fig2['cf_start'] = shock_period
                plot_settings.dp_shock_fig2['cf_start'] = shock_period

                # Figure 3
                plot_settings.td3_shock_fig3['color'] = plot_settings.colors_fig3[j]
                plot_settings.td3_shock_fig3['label'] = f'RL agent ({hyper_name} = {hyper_value})'
                plot_settings.td3_shock_fig3['history'] = rl_agent_shock.history

                if j == 0:
                    plot_settings.dp_shock_fig3['history'] = dp_agent_shock.history
                    plot_settings.dp_shock_fig3['cf_start'] = shock_period
                    agent_dicts_fig3.append(plot_settings.dp_shock_fig3.copy())

                # Append to lists
                agent_dicts_fig2.append(plot_settings.td3_shock_fig2.copy())
                agent_dicts_fig2.append(plot_settings.dp_shock_fig2.copy())
                agent_dicts_fig2.append(plot_settings.td3_fig2.copy())
                agent_dicts_fig2.append(plot_settings.dp_fig2.copy())

                agent_dicts_fig3.append(plot_settings.td3_shock_fig3.copy())

                # Plot Figure 2
                var_list = ['m','c','a','s']
                plot_avg_trajectories_separate2(*agent_dicts_fig2,
                    var_list=var_list,
                    shock_line=shock_period,
                    save=(f'plots/fig2/fig2_'
                          f'{model_key}{model_value}_'
                          f'{hyper_key}{hyper_value}'))

            # Plot Figure 3
            var_list = ['m','c','a','s']
            plot_avg_trajectories_separate2(*agent_dicts_fig3,
                var_list=var_list,
                shock_line=shock_period,
                save=(f'plots/fig3/fig3_'
                      f'{model_key}{model_value}_'
                      f'{hyper_key}'))

        # Plot Figure 1
        var_list = ['m','c','a','s']
        plot_avg_trajectories_separate2(*agent_dicts_fig1,
            var_list=var_list,
            shock_line=shock_period,
            save=(f'plots/fig1/fig1_{model_key}'))
