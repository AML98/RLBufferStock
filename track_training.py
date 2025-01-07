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
    "lines.linewidth": 2.0
})

# Imports for environment
from base_env import BaseEnv
from buffer_stock_env import BufferStockEnv

# Imports for plotting
from history_manager import HistoryManager
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
training_episodes = 100  # Number of episodes to train TD3
testing_episodes = 5000  # Number of episodes to simulate DP

# -----------------
# - Training runs -
# -----------------

n_runs = 50
td3_history_manager = HistoryManager(name="TD3_Agent")
dp_history_manager = HistoryManager(name="DP_Agent")

# Solve with DP
par_ = {'T':T, 'rho':rho, 'beta':beta, 'do_print':False}
model = BufferStockModelClass(name='baseline', par=par_)
model.solve()

for run in range(n_runs):
    # Initialize environment
    env = BufferStockEnv(T=T, rho=rho, beta=beta)
    if run == 0:
        env.seed(seed)

    # Initialize TD3 agent
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    td3_agent = TD3Agent(state_dim, action_dim, beta=beta)

    # Train the agent and save history
    td3_agent.interact(env, training_episodes, train=True, do_print=False)
    td3_history_manager.add_history(td3_agent.history)
    
    # Simulate DP agent to compare life-time utility
    dp_agent = DPAgent(model, env)
    dp_agent.interact(env, training_episodes, train=False)
    dp_history_manager.add_history(dp_agent.history)

    print(f"Completed run {run + 1}/{n_runs}")

plot_value(td3_history_manager, dp_history_manager, mean1=True,
    save="plots/test_layout.pgf")

td3_agent.interact(env, testing_episodes, train=False)

plot_avg_trajectories_separate(td3_agent, dp_agent, td3_history_manager, con_bands=False, var_list=['m','c','a','s'])
