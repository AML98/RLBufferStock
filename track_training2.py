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
training_episodes = 1000  # Number of episodes to train TD3
testing_episodes = 5000  # Number of episodes to simulate DP

# ----------------
# - Training run -
# ----------------

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

# Initialize environment
env = BufferStockEnv(T=T, rho=rho, beta=beta)
env.seed(seed)

# Initialize TD3 agent
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
td3_agent = TD3Agent(state_dim, action_dim, beta=beta)

# Train the agent and track training process
eps_to_track = [300-1, 600-1, 1000-1]
td3_agents = td3_agent.interact(env, training_episodes, train=True, 
    do_print=False, track=eps_to_track)

# Simulate DP agent
dp_agent = DPAgent(model, env)
dp_agent.interact(env, training_episodes, train=False)

# Simulate TD3 agents
for td3_agent in td3_agents:
    td3_agent.interact(env, testing_episodes, train=False)

# --------
# - Plot -
# --------

# Plot settings for DP agent
# dp_agent_dict = {
#    'history':dp_agent.history,
#    'label':'DP agent',
#    'color':'red',
#    'linestyle':'--',
#    'linewidth':3,
#    'linestart':0,
#    'cf_start':None,
# }

# Plot settings for TD3 agents
td3_agents_dicts = [] 
colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(td3_agents)))

for i, td3_agent in enumerate(td3_agents):
    td3_agent_dict = {
        'history':td3_agent.history,
        'label':f'RL agent, episode {eps_to_track[i]+1}',
        'color':colors[i],
        'linestyle':'--',
        'linewidth':2,
        'linestart':0,
        'cf_start':None,
    }

    # Agent of last episode
    if i == (len(td3_agents)-1):
        td3_agent_dict['linewidth'] = 3
        td3_agent_dict['linestyle'] = '-'
        td3_agent_dict['cf_start'] = 0

    td3_agents_dicts.append(td3_agent_dict)

agents_dicts = td3_agents_dicts

# Show/save plot
plot_avg_trajectories_separate2(*agents_dicts, 
    var_list=['m','c','a','s'])
