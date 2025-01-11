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
training_episodes = 2000  # Number of episodes to train
shock_period = 5

# Choose which RL agent to use
# agent_class = TD3Agent
agent_class = DDPGAgent

params = {
    'sigma_xi': [[0.5], '$\sigma_xi$']
}

for param, (param_values, param_name) in params.items():
    agent_dicts_fig1 = []

    for i, param_value in enumerate(param_values):
        agent_dicts_fig2 = []

        # --- DP SOLUTIONS ---

        # Baseline - excl. shock
        par = {
            'solmethod':'egm',
            'T':T,
            'rho':rho,
            'beta':beta,
            'do_print':False
        }
        model = BufferStockModelClass(name='baseline', par=par)
        model.solve()

        # Incl. shock
        par_shock = par.copy()
        par_shock[param] = param_value
        par_shock['T'] = T - shock_period
        model_shock = BufferStockModelClass(name='shock', par=par_shock)
        model_shock.solve()

        # --- ENVIRONMENTS ---

        # Baseline - excl. shock
        env = BufferStockEnv(
            T=T, 
            rho=rho, 
            beta=beta
        )

        # Incl. shock
        env_shock = BufferStockEnv(
            reset_period=shock_period,
            T=T,
            rho=rho,
            beta=beta,
            **{param: param_value}
        )

        env.seed(1998)
        env_shock.seed(1998)

        # --- AGENTS ---

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        rl_agent = agent_class(
            state_dim,
            action_dim,
            beta=beta  
        )

        dp_agent = DPAgent(model, env)

        # --- RL LEARNING (BASELINE) ---

        rl_agent.interact(
            env,
            train=True,
            n_episodes=training_episodes,
            do_print=False
        )

        # plot_policy(rl_agent, dp_agent) # does not work because self.env.period is not reset in dp_agent

        # --- BASELINE SIMULATION ---

        # Use the same env seed for DP and RL
        random.seed(1999)
        env_shock.seed(1999)
        np.random.seed(1999)
        rl_agent.interact(env, testing_episodes, train=False)

        random.seed(1999)
        env_shock.seed(1999)
        np.random.seed(1999)
        dp_agent.interact(env, testing_episodes, train=False)

        # Save histories for later
        rl_agent.history.save('rl_agent.pkl')
        dp_agent.history.save('dp_agent.pkl')

        # --- PREP SHOCK ---
        
        # Want to start from states reached in shock_period
        inis_m_rl = rl_agent.history.m[:,shock_period]
        inis_p_rl = rl_agent.history.p[:,shock_period]

        inis_m_dp = dp_agent.history.m[:,shock_period]
        inis_p_dp = dp_agent.history.p[:,shock_period]

        inis_t = np.ones_like(inis_m_rl)*shock_period/env.T

        inis_rl = [{"m": m_val, "p": p_val, "t": t_val} 
                   for m_val, p_val, t_val in zip(inis_m_rl, inis_p_rl, inis_t)]
    
        inis_dp = [{"m": m_val, "p": p_val, "t": t_val} 
                   for m_val, p_val, t_val in zip(inis_m_dp, inis_p_dp, inis_t)]
        
        # Deep copy agents
        rl_agent_shock = copy.deepcopy(rl_agent)
        dp_agent_shock = copy.deepcopy(dp_agent)
        dp_agent_shock.model = model_shock
        dp_agent_shock.env = env_shock

        # --- RL LEARNING (SHOCK) ---

        rl_agent_shock.interact(
            env_shock, 
            train=True,
            n_episodes=training_episodes,
            do_print=False, 
            inis=inis_rl
        )

        # --- SHOCK SIMULATION ---

        # Load histories from baseline (before shock_period)
        dp_agent_shock.history = History.load('dp_agent.pkl')
        rl_agent_shock.history = History.load('rl_agent.pkl')

        # Simulate
        random.seed(2000)
        env_shock.seed(2000)
        np.random.seed(2000)
        dp_agent_shock.interact(
            env_shock, 
            testing_episodes, 
            keep_history=True, 
            train=False, 
            inis=inis_dp,
            shift=shock_period  # To extract the correct policy in dp_agent.py
        )
        
        random.seed(2000)
        env_shock.seed(2000)
        np.random.seed(2000)
        rl_agent_shock.interact(
            env_shock, 
            testing_episodes, 
            keep_history=True,
            train=False,
            inis=inis_rl
        )

        # --- PREP PLOTS ---

        # Figure 1
        plot_settings.rl_fig1['history'] = rl_agent.history
        plot_settings.dp_fig1['history'] = dp_agent.history

        # Figure 2
        plot_settings.rl_shock_fig2['history'] = rl_agent_shock.history
        plot_settings.dp_shock_fig2['history'] = dp_agent_shock.history
        
        plot_settings.rl_fig2['history'] = rl_agent.history
        plot_settings.dp_fig2['history'] = dp_agent.history

        plot_settings.rl_shock_fig2['cf_start'] = shock_period-1
        plot_settings.dp_shock_fig2['cf_start'] = shock_period-1

        # --- DO PLOTS ---

        # Plot Figure 1 - baseline
        plot_fig1(
            plot_settings.rl_fig1,
            plot_settings.dp_fig1,
            save=(f'plots/baseline')
        )

        # Plot Figure 2
        plot_fig2(
            plot_settings.rl_shock_fig2,
            plot_settings.rl_fig2,
            plot_settings.dp_shock_fig2,
            plot_settings.dp_fig2,
            var_list=['a','m','x'],
            shock_line=shock_period-1,
            save=(f'plots/fig2/fig2_'
                    f'{param}{param_value}_')
        )

        # Plot Figure 3
        plot_fig3(
            plot_settings.rl_shock_fig2,
            plot_settings.rl_fig2,
            plot_settings.dp_shock_fig2,
            plot_settings.dp_fig2,
            var_list=['a','x','c'],
            shock_line=shock_period-1,
            save=(f'plots/fig3/fig3_'
                    f'{param}{param_value}_')
        )

        #plot_consumption_functions_subplots(
        #    t=7,
        #    agent_dicts = [plot_settings.rl_fig2, plot_settings.dp_fig2], 
        #    save=f'plots/consumption_functions_{param}{param_value}'
        # )

        # --- TABLES ---

        print_avg_values(
            rl_agent_shock, 
            dp_agent_shock, 
            rl_agent, 
            dp_agent, 
            var_list=['c','a','value'])
