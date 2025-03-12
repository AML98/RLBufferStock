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

def custom_format(v, sig_figs=2, threshold=0.001):
    return f"{v}" if abs(v) >= threshold else f"{v:.{sig_figs}e}"

# Settings
T = 20
rho = 1.0
beta = 0.96
testing_episodes = 10000  # Number of episodes to test
training_episodes = 2001  # Number of episodes to train
runs = 1

params = [
    {'noise_decay': [[0.0075, 0.0125], r'$\gamma$', 0.01],
     'ini_explore_noise': [[0.3, 0.1], r'$\sigma_c$', 0.2]},
    {'actor_lr': [[0.75*1e-4, 0.25*1e-4], r'$\alpha_{\phi}$', 0.5*1e-4],
    'critic_lr': [[0.25*1e-3, 0.75*1e-4], r'$\alpha_{\theta}$', 1e-4]},
    {'actor_hidden_dim': [[128, 32], 'N neurons', 64],
    'critic_hidden_dim': [[128, 32], '', 64]}
]

labels = [
    ['Moderate exploration', 'High exploration', 'Low exploration'],
    ['Moderate learning rate', 'High learning rate', 'Low learning rate'],
    ['Moderate size NN', 'Large NN', 'Small NN']
]

plot_dicts = [
    plot_settings.rl1_learning_fig, 
    plot_settings.rl2_learning_fig,
    plot_settings.rl3_learning_fig
]
plot_dicts_lifecycle = [
    plot_settings.hyp_rl1,
    plot_settings.hyp_rl2,
    plot_settings.hyp_rl3,
    plot_settings.hyp_dp
]
plot_dicts_policy = [
    plot_settings.hyp_rl1_policy,
    plot_settings.hyp_rl2_policy,
    plot_settings.hyp_rl3_policy,
    plot_settings.hyp_dp_policy
]

agent_class = DDPGAgent

# For table
metrics = ["c", "a", "m", "value", "trans_euler_error", "constrained"]

# -----------------------
# - HYPERPARAMETER EXPS -
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
print(f'DP solution found in {elapsed(dp_baseline_tik)}')

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
    action_dim=action_dim,
    state_dim=state_dim,
    beta=beta, 
)
dp_agent = DPAgent(model, env)

# RL learning
rl_baseline_tik = time.time()
rl_agent.interact(
    env,
    train=True,
    n_episodes=training_episodes,
    keep_history=False,  # True -> Does not record training history
    do_print=False
)
print(f'RL baseline training done in {elapsed(rl_baseline_tik)}')

c_loss = rl_agent.history.c_loss
a_loss = rl_agent.history.a_loss
trans_euler_error = np.nanmean(rl_agent.history.trans_euler_error, axis=1)

plot_dicts[0]['a_loss'] = a_loss
plot_dicts[0]['c_loss'] = c_loss
plot_dicts[0]['euler_error'] = trans_euler_error

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

print(f'Baseline life-cycle simulation done in {elapsed(baseline_sim_tik)}')

plot_dicts_lifecycle[0]['history'] = rl_agent.history
plot_dicts_lifecycle[3]['history'] = dp_agent.history
plot_dicts_policy[0]['agent'] = rl_agent
plot_dicts_policy[3]['agent'] = dp_agent

hyper_results = {
    "baseline": {
        rl_agent.name: extract_metrics(rl_agent, metrics)
    }
}

for j,param_set in enumerate(params):
    for i in range(2):  # Length of param_values

        param_dict = {param_name: param_values[i] 
            for param_name, (param_values, param_label, baseline_val) in param_set.items()}
        
        param_labels = {param_name: param_label 
            for param_name, (param_values, param_label, baseline_val) in param_set.items()}
        
        baseline_vals = {param_name: baseline_val 
            for param_name, (param_values, param_label, baseline_val) in param_set.items()}

        rl_tik = time.time()
        
        # RL agent with different hyperparameter
        rl_agent = agent_class(
            action_dim=action_dim,
            state_dim=state_dim,
            beta=beta,  
            **param_dict
        )

        # RL learning
        rl_agent.interact(
            env,
            train=True,
            n_episodes=training_episodes,
            keep_history=False,
            do_print=False
        )

        print(f"RL training for {', '.join(f'{k} = {custom_format(v)}' for k, v in param_dict.items())} done in {elapsed(rl_tik)}")

        # Save for table
        hyper_results[", ".join(f"{param_labels[k]}={custom_format(v)})" 
            for k, v in param_dict.items())] = {rl_agent.name: extract_metrics(rl_agent, metrics)}

        # Collect learning metrics
        c_loss = rl_agent.history.c_loss
        a_loss = rl_agent.history.a_loss
        with np.errstate(divide='ignore', invalid='ignore'):
            trans_euler_error = np.nanmean(rl_agent.history.trans_euler_error, axis=1)

        # Average over runs
        avg_c_loss = c_loss
        avg_a_loss = a_loss
        avg_euler_error = trans_euler_error

        # Learning plot
        base_label = f"{labels[j][0]}\n({', '.join(f'{param_labels[k]} = {custom_format(baseline_vals[k])}' for k in baseline_vals)})"
        label = f"({', '.join(f'{param_labels[k]} = {custom_format(v)}' for k, v in param_dict.items())})"
        label = f'{labels[j][i+1]}\n{label}'

        plot_dicts[i+1]['a_loss'] = avg_a_loss
        plot_dicts[i+1]['c_loss'] = avg_c_loss
        plot_dicts[i+1]['euler_error'] = avg_euler_error

        plot_dicts[i+1]['label'] = label
        plot_dicts_policy[i+1]['label'] = label
        plot_dicts_lifecycle[i+1]['label'] = label
        
        plot_dicts[0]['label'] = base_label
        plot_dicts_policy[0]['label'] = base_label
        plot_dicts_lifecycle[0]['label'] = base_label

        # Policy and lifecycle plot
        rl_sim_tik = time.time()

        rl_agent.interact(env, testing_episodes, train=False)
        
        plot_dicts_policy[i+1]['agent'] = rl_agent
        plot_dicts_lifecycle[i+1]['history'] = rl_agent.history
        print(f"RL life-cycle simulation for {', '.join(f'{k} = {custom_format(v)}' for k, v in param_dict.items())} done in {elapsed(rl_tik)}")

    plot_learning(
        plot_settings.dp_learning_fig,
        [plot_settings.rl1_learning_fig, 
        plot_settings.rl2_learning_fig,
        plot_settings.rl3_learning_fig],
        n_episodes=training_episodes,
        save=f"plots/learning_{'_'.join(f'{param_name}' for param_name, v in param_dict.items())}"
    )

    plot_fig1_policy(
        plot_settings.hyp_rl1, 
        plot_settings.hyp_rl2, 
        plot_settings.hyp_rl1_policy, 
        plot_settings.hyp_rl2_policy, 
        T, 
        env.R, 
        # other_dict=plot_settings.hyp_dp, 
        # other_policy_dict=plot_settings.hyp_dp_policy, 
        another_dict=plot_settings.hyp_rl3,
        another_policy_dict=plot_settings.hyp_rl3_policy,
        save=f"plots/baseline_{'_'.join(f'{param_name}' for param_name, v in param_dict.items())}", 
        use_master_legend=True
    )