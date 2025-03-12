import warnings
import traceback

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
from htm_agent import HTMAgent
from td3_agent import TD3Agent
from dp_agent import DPAgent
from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer
from networks import Actor, Critic

# Imports for DP
from DP_solution_normalized import BufferStockModelClass
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

# Settings
T = 20
rho = 1.0
beta = 0.96
testing_episodes = 10000  # Number of episodes to test
training_episodes = 2000  # Number of episodes to train
shock_period = 5

params = {
    'sigma_xi': [[0.3], r'$\sigma_{\xi}$'],
    'sigma_psi': [[0.3], r'$\sigma_{\psi}$'],
    'R': [[1.04, 1.02], r'$R$'],
    'pi': [[0.5], r'$\pi$'],
    'mu': [[0.25], r'$\mu$'],
}

baseline_only = False

agent_class = DDPGAgent

# -----------------------
# - BASELINE (NO SHOCK) -
# -----------------------

# DP solution
par = {
    'solmethod':'vfi',
    'T':T,
    'rho':rho,
    'beta':beta,
    'do_print':False
}
model = BufferStockModelClass(name='baseline', par=par)

dp_baseline_tik = time.time()
model.solve()
print(f'DP baseline solution found in {elapsed(dp_baseline_tik)}')

# Environment
env = BufferStockEnv(
    T=T, 
    rho=rho, 
    beta=beta
)

env.seed(1998)

# Agents
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
rl_agent = agent_class(
    state_dim,
    action_dim,
    beta=beta  
)

dp_agent = DPAgent(model, env)
htm_agent = HTMAgent(env)

# RL learning
rl_baseline_tik = time.time()
rl_agent.history = History('test', env, training_episodes)
rl_agent.interact(
    env,
    train=True,
    n_episodes=training_episodes,
    keep_history=True,  # True -> Does not record training history
    do_print=False
)
print(f'RL baseline training done in {elapsed(rl_baseline_tik)}') 

rl_agent.history.save('rl_agent_base_training.pkl')

# Life-cycle simulations
baseline_sim_tik = time.time()

env.seed(1999)
random.seed(1999)
np.random.seed(1999)  # Use the same env seed for DP and RL
rl_agent.interact(env, testing_episodes, train=False)

env.seed(1999)
random.seed(1999)
np.random.seed(1999)
dp_agent.interact(env, testing_episodes, train=False)

env.seed(1999)
random.seed(1999)
np.random.seed(1999)
htm_agent.interact(env, testing_episodes, train=False)

# Save histories for later
rl_agent.history.save('rl_agent.pkl')
dp_agent.history.save('dp_agent.pkl')

print(f'Baseline life-cycle simulation done in {elapsed(baseline_sim_tik)}')

# Plot settings
plot_settings.rl_fig1['history'] = rl_agent.history
plot_settings.dp_fig1['history'] = dp_agent.history

plot_settings.rl_policy_fig['agent'] = rl_agent
plot_settings.dp_policy_fig['agent'] = dp_agent

plot_fig1_policy(plot_settings.rl_fig1, plot_settings.dp_fig1, 
                 plot_settings.rl_policy_fig, plot_settings.dp_policy_fig, 
                 T, env.R, save='plots/baseline_combined', use_master_legend=True, exp_m=True)

print_avg_values(
    rl_agent, 
    dp_agent, 
    var_list=['c','a','value'])

# ------------------------- 
# - Parameter experiments -
# -------------------------

if baseline_only is False:

    # Dictionary for big table
    results = {'baseline': {'RL': None, 'DP': None, 'HTM': None}}

    for param, (param_values, param_name) in params.items():
        for i, param_value in enumerate(param_values):
            shock_name = f'{param}={param_value}'
            agent_dicts_fig2 = []

            # DP solution
            par_shock = par.copy()
            par_shock[param] = param_value
            par_shock['T'] = T - shock_period
            model_shock = BufferStockModelClass(name='shock', par=par_shock)

            dp_shock_tik = time.time()
            model_shock.solve()
            print(f'DP solution for {shock_name} shock found in {elapsed(dp_shock_tik)}')

            # Environment
            env_shock = BufferStockEnv(
                reset_period=shock_period,
                T=T,
                rho=rho,
                beta=beta,
                **{param: param_value}
            )

            env_shock.seed(1998)

            # Want to start simulation from states reached in shock_period
            inis_m_rl = rl_agent.history.m[:,shock_period]
            inis_m_dp = dp_agent.history.m[:,shock_period]

            inis_t = np.ones_like(inis_m_rl)*shock_period/(env.T-1)

            inis_rl = [{"m": m_val, "t": t_val} 
                    for m_val, t_val in zip(inis_m_rl, inis_t)]
        
            inis_dp = [{"m": m_val, "t": t_val} 
                    for m_val, t_val in zip(inis_m_dp, inis_t)]
            
            # Deep copy agents
            rl_agent_shock = copy.deepcopy(rl_agent)
            dp_agent_shock = copy.deepcopy(dp_agent)
            dp_agent_shock.model = model_shock
            dp_agent_shock.env = env_shock

            # RL learning
            rl_shock_tik = time.time()
            rl_agent_shock.history = History('test', env_shock, training_episodes)
            rl_agent_shock.interact(
                env_shock, 
                train=True,
                n_episodes=training_episodes,
                keep_history=True,  # Does not record training history
                do_print=False, 
                inis=inis_rl
            )
            print(f'RL training for {shock_name} shock done in {elapsed(rl_shock_tik)}')

            # Load histories from baseline (before shock_period)
            dp_agent_shock.history = History.load('dp_agent.pkl')
            rl_agent_shock.history = History.load('rl_agent.pkl')

            # Simulate
            shock_sim = time.time()

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

            print(f'Life-cycle simulation for {shock_name} shock done in {elapsed(shock_sim)}')

            # Figure 3
            plot_settings.rl_shock_fig2['history'] = rl_agent_shock.history
            plot_settings.dp_shock_fig2['history'] = dp_agent_shock.history
            
            plot_settings.rl_fig2['history'] = rl_agent.history
            plot_settings.dp_fig2['history'] = dp_agent.history

            plot_settings.rl_shock_fig2['cf_start'] = shock_period
            plot_settings.dp_shock_fig2['cf_start'] = shock_period

            plot_settings.rl_policy_fig['agent'] = rl_agent_shock
            plot_settings.dp_policy_fig['agent'] = dp_agent_shock
            plot_settings.dp_policy_base_fig['agent'] = dp_agent

            plot_fig3_policy(
                plot_settings.rl_shock_fig2,
                plot_settings.dp_shock_fig2,
                plot_settings.dp_fig2,
                plot_settings.rl_policy_fig,
                plot_settings.dp_policy_fig,
                shock_period,
                T,
                env_shock.R,
                shift=shock_period,
                save=f'plots/shock_combined_{param}{param_value}',
                use_master_legend=True
            )

            # Table
            stats = ['c', 'a', 'm', 'value', 'trans_euler_error', 'constrained']

            if results['baseline']['RL'] is None:  # Only store baseline once
                results['baseline']['DP'] = extract_metrics(dp_agent, stats)
                results['baseline']['RL'] = extract_metrics(rl_agent, stats)
                results['baseline']['HTM'] = extract_metrics(htm_agent, stats)
            
            results[shock_name] = {
                'RL': extract_metrics(rl_agent_shock, stats),
                'DP': extract_metrics(dp_agent_shock, stats)
            }

    table_tex = create_generalized_table(results, params)
    with open('comparison_table.tex', 'w') as f:
        f.write(table_tex)