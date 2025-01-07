# ---------
# - Setup -
# ---------

# Basic imports
import time
import numpy as np
import numba as nb
import random
import torch
import copy
import matplotlib.pyplot as plt

# Set seed
seed = 1998
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Plot setting
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({
    "axes.grid": True, 
    "grid.color": "black",
    "grid.linestyle": "--", 
    "grid.alpha": 0.25,
    "font.size": 20,
    "font.family": "sans-serif", 
    "pgf.texsystem": "pdflatex",
    "lines.linewidth": 3.0
})

# Imports for environment
from base_env import BaseEnv
from buffer_stock_env import BufferStockEnv

# Imports for plotting
from history import History
from show_results import *

# Imports for RL
from agent import Agent
from dp_agent import DPAgent
from td3_agent import TD3Agent
from replay_buffer import ReplayBuffer
from networks import Actor, Critic

# Imports for DP
from DP_solution import BufferStockModelClass
from consav.misc import elapsed

# Globals
T = 20
rho = 1.0
beta = 0.96
testing_episodes = 5000  # Number of episodes to test
training_episodes = 1000  # Number of episodes to train

# --------------------------
# - Solution without shock -
# --------------------------

# Initialize environment
env = BufferStockEnv(T=T, rho=rho, beta=beta)
env.seed(1998)

# Solve with DP
par_ = {
    'solmethod':'egm', 
    'T':T, 
    'rho':rho, 
    'beta':beta, 
    'do_print':False
}
model = BufferStockModelClass(name='baseline', par=par_)
model.solve()

# Initialize agents
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
td3_agent = TD3Agent(state_dim, action_dim, beta=beta)
dp_agent = DPAgent(model, env)

# Train TD3 agent
td3_agent.interact(env, training_episodes, train=True, 
    do_print=False)

# Simulate
td3_agent.interact(env, testing_episodes, train=False)
dp_agent.interact(env, testing_episodes, train=False)
td3_agent.history.save('td3_agent.pkl')
dp_agent.history.save('dp_agent.pkl')

# Plot settings for TD3 agent
colors = plt.cm.Blues(np.linspace(1.0, 1.0, 1))

td3_agent_dict = {
    'history':td3_agent.history,
    'label':'RL agent',
    'color':colors[0],
    'linestyle':'-',
    'linewidth':3,
    'linestart':0,
    'cf_start':0,
}

# Plot settings for DP agent
dp_agent_dict = {
    'history':dp_agent.history,
    'label':f'DP agent',
    'color':'red',
    'linestyle':'--',
    'linewidth':3,
    'linestart':0,
    'cf_start':0,
}

# Plot
var_list = ['m', 'c', 'a', 's']
agent_dicts = (td3_agent_dict, dp_agent_dict)
plot_avg_trajectories_separate2(*agent_dicts, var_list=var_list)

# ------------------------- 
# - Parameter experiments -
# -------------------------

shock_period = 10

par_exps = {
    'mu': [0.25, '$\mu$'],
#    'sigma_xi': [0.1, '$\sigma_{\\xi}$'],
#    'sigma_psi': [0.1, '$\sigma_{\\psi}$'],
#    'pi': [0.2, '$\pi$'],
#    'R': [1.1, '$R$']
}

for key, value in par_exps.items():
    par_value = value[0]
    label = value[1]
    
    # Create new environment with shock
    ini_m = np.mean(td3_agent.history.m[:,shock_period], axis=0)
    ini_p = np.mean(td3_agent.history.p[:,shock_period], axis=0)

    state_vars = [
        {'name': 'p', 'ini': ini_p, 'low': 0.0, 'high': 100},
        {'name': 'm', 'ini': ini_m, 'low': 0.0, 'high': 100},
        {'name': 't', 'ini': shock_period/env.T, 'low': 0.0, 'high': 1.0}
    ]

    env_change = BufferStockEnv(state_vars=state_vars, 
        reset_period=shock_period, T=30, **{key: value[0]})
    
    # Solve with DP
    par_shock = par_.copy()
    par_shock[key] = par_value
    par_shock['T'] = T - shock_period
    model_shock = BufferStockModelClass(name='shock', par=par_shock)
    model_shock.solve()

    # Initialize new agents
    td3_agent_shock = copy.deepcopy(td3_agent)
    dp_agent_shock = DPAgent(model_shock, env_change)
    
    # Train TD3 agent
    td3_agent_shock.interact(env_change, training_episodes, train=True, 
        do_print=False)

    # Simulate (reuse history for T < 5)
    dp_agent_shock.history = History.load('dp_agent.pkl')
    td3_agent_shock.history = History.load('td3_agent.pkl')

    dp_agent_shock.interact(env_change, testing_episodes, train=False, 
        keep_history=True, shift=shock_period)
    
    td3_agent_shock.interact(env_change, testing_episodes, train=False, 
        keep_history=True)

    # Plot settings
    dp_agent_dict['label'] = 'DP agent, without shock'
    dp_agent_dict['linestyle'] = ':'
    dp_agent_dict['linewidth'] = 1.0
    dp_agent_dict['cf_start'] = None
    dp_agent_dict['alpha'] = 0.75

    dp_agent_shock_dict = dp_agent_dict.copy()
    dp_agent_shock_dict['history'] = dp_agent_shock.history
    dp_agent_shock_dict['label'] = 'DP agent, with shock'
    dp_agent_shock_dict['linewidth'] = 1.5
    dp_agent_shock_dict['linestyle'] = '-'
    dp_agent_shock_dict['cf_start'] = shock_period
    dp_agent_shock_dict['alpha'] = 1.0

    td3_agent_dict['label'] = 'RL agent before learning'
    td3_agent_dict['linestyle'] = ':'
    td3_agent_dict['linewidth'] = 1.0
    td3_agent_dict['cf_start'] = None
    td3_agent_dict['alpha'] = 0.75

    td3_agent_shock_dict = td3_agent_dict.copy()
    td3_agent_shock_dict['history'] = td3_agent_shock.history
    td3_agent_shock_dict['label'] = 'RL agent after learning'
    td3_agent_shock_dict['linestyle'] = '-'
    td3_agent_shock_dict['linewidth'] = 1.5
    td3_agent_shock_dict['cf_start'] = shock_period
    td3_agent_shock_dict['alpha'] = 1.0

    # Plot TD3 vs DP with shock
    var_list = ['m','c','a','s']
    agents_dicts = [td3_agent_shock_dict, td3_agent_dict, dp_agent_dict, dp_agent_shock_dict]
    plot_avg_trajectories_separate2(*agents_dicts, var_list=var_list, shock_line=shock_period-1)