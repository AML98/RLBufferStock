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
import json
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

# --------------- 
# - Grid search -
# ---------------

T = 20
rho = 1.0
beta = 0.96
testing_episodes = 5000  # Number of episodes to test
training_episodes = 1000  # Number of episodes to train
var_list = ['m','c','a','s']  # Variables to plot
n_runs = 2  # Number of runs

# Initialize environment
env = BufferStockEnv(T=T, rho=rho, beta=beta)
env.seed(1998)

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

# Simulate DP interaction
dp_agent = DPAgent(model, env)
dp_agent.interact(env, testing_episodes, train=False)
plot_settings.dp_fig0['history'] = dp_agent.history

# Hyper parameters
hyper_params = {
    'hidden_dim' : [64, 32],
    'batch_size' : [512, 1028],
    'ini_explore_noise' : [0.3, 0.4],
    'explore_decay' : [0.998, 0.999],
    'actor_lr' : [1e-4, 1e-5],
    'critic_lr' : [1e-3, 1e-4]
}

param_names = list(hyper_params.keys())
param_values = list(hyper_params.values())
param_combs = itertools.product(*param_values)

best_value = float('-inf')
best_params = None
results = []

for param_set in param_combs:
    current_params = dict(zip(param_names, param_set))
    
    current_values = 0

    for run in range(n_runs):
        # Initialize TD3 agent
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        hidden_dim = current_params['hidden_dim']

        td3_agent = TD3Agent(
            state_dim, action_dim, beta=beta,
            actor_hidden_dim=hidden_dim,
            critic_hidden_dim=hidden_dim,
            **current_params
        )

        # Interact TD3 agent
        td3_agent.interact(env, training_episodes, train=True, do_print=False)  # Train
        td3_agent.interact(env, testing_episodes, train=False)  # Test
        current_values += td3_agent.history.value[0]

    current_value = current_values/n_runs  # Use average across runs

    # Save the current results
    results.append({
        'params': current_params,
        'value': current_value
    })

    if current_value > best_value:
        best_value = current_value
        best_params = current_params

    # Plot
    label_parts = []
    name_parts = []

    for key, value in current_params.items():
        label_parts.append(f"{key}={value}")
        name_parts.append(f"{key[:3]}={value}")

    label = ', '.join(label_parts)
    figure_name = '_'.join(name_parts)
    save_path = f"plots/hyperparam_tuning/{figure_name}.pgf"

    plot_settings.td3_fig0['label'] = label
    plot_settings.td3_fig0['history'] = td3_agent.history

    plot_avg_trajectories_separate2(plot_settings.dp_fig0, 
        plot_settings.td3_fig0, var_list=var_list, 
        save=save_path)

# Final results
print(f"Best Value: {best_value}")
print(f"Best Parameters: {best_params}")

with open('hyperparameter_results.json', 'w') as f:
    json.dump(results, f, indent=4)
