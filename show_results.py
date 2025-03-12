import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import math

import seaborn as sns

from consav.grids import nonlinspace

# --------------------
# - Helper functions -
# --------------------

def plot_avg_trajectory(agent_dict, var, ax):
    """
    Plot average trajectory for var using agent_dict on ax.
    """
    line = agent_dict['history'].plot_avg_trajectory2(
        var=var, 
        color=agent_dict['color'],
        linestyle=agent_dict['linestyle'], 
        linewidth=agent_dict['linewidth'], 
        linestart=agent_dict['linestart'], 
        cf_start=agent_dict['cf_start'], 
        label=agent_dict['label'], 
        alpha=agent_dict['alpha'],
        ax=ax
    )

    return line

def plot_sd(agent_dict, var, ax):
    """
    Plot average trajectory for var using agent_dict on ax.
    """
    line = agent_dict['history'].plot_sd2(
        var=var, 
        color=agent_dict['color'],
        linestyle=agent_dict['linestyle'], 
        linewidth=agent_dict['linewidth'], 
        linestart=agent_dict['linestart'], 
        label=agent_dict['label'], 
        alpha=agent_dict['alpha'],
        ax=ax
    )

    return line

# -----------
# - Figures -
# -----------

def plot_fig1(rl_dict, dp_dict, other_dict=None, another_dict=None, save=None, axes=None, add_legend=True):
    """
    """

    aspect_ratio = 1
    base_fig_width = 15

    n_cols = 3
    n_rows = 1

    fig_height = aspect_ratio * (base_fig_width * n_rows / n_cols)
    
    if axes is None:
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols, 
            figsize=(base_fig_width, fig_height))
        
        axes = axes.flatten()

    else:
        fig = axes[0].figure

    var_list = ['m', 'c', 'a']
    var_names = ['Cash-on-hand, $m_t$', 'Consumption, $c_t$', 'Assets, $a_t$']

    # For legend collection
    lines, labels = [], []

    for i, var in enumerate(var_list):
        ax = axes[i]
        ax.set_title(var_names[i])

        for d in [dp_dict, rl_dict, other_dict, another_dict]:    
            if d is not None:
                line = plot_avg_trajectory(d, var, ax)
                if line.get_label() not in labels:
                    labels.append(line.get_label())
                    lines.append(line)

    if add_legend:
        custom_order = [dp_dict['label'], rl_dict['label']]

        custom_lines = [lines[labels.index(lbl)] for lbl in custom_order]
        custom_labels = custom_order

        fig.legend(
            custom_lines, 
            custom_labels, 
            loc="upper center", 
            bbox_to_anchor=(0.5, 0.0), 
            ncol=2, 
            frameon=False)
    
    if axes is None:
        fig.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close(fig)

def plot_fig3(rl_shock_dict, dp_shock_dict, dp_dict, shock_line, save=None, add_legend=True, axes=None):
    """
    """
    # Keep aspect ratio of subplots!
    aspect_ratio = 1
    base_fig_width = 15

    n_cols = 3
    n_rows = 1

    fig_height = aspect_ratio * (base_fig_width * n_rows / n_cols)
    
    if axes is None:
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols, 
            figsize=(base_fig_width, fig_height))
        
        axes = axes.flatten()

    else:
        fig = axes[0].figure


    var_list = ['m', 'c', 'a']
    var_names = ['Cash-on-hand, $m_t$', 'Consumption, $c_t$', 'Assets, $a_t$']

    # For legend
    lines, labels = [], []

    for i, var in enumerate(var_list):
        title = var_names[i]
        axes[i].set_title(title)
        
        # Shock line
        axes[i].axvline(
            x=shock_line, 
            color='black',
            linewidth=1.0,
            linestyle='-',
            alpha=0.8
        )

        # Average trajectory
        for d in [dp_dict, dp_shock_dict, rl_shock_dict]:
            line = plot_avg_trajectory(d, var, axes[i])
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

    if add_legend:
        custom_order = [
            dp_shock_dict['label'],
            rl_shock_dict['label'],
            dp_dict['label']
        ]

        custom_lines = [lines[labels.index(lbl)] for lbl in custom_order]
        custom_labels = custom_order

        fig.legend(
            custom_lines, 
            custom_labels, 
            loc="upper center", 
            bbox_to_anchor=(0.5, 0.0), 
            ncol=2, 
            frameon=False)
    
    if axes is None:
        fig.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close(fig)

def plot_policy(rl_policy_dict, dp_policy_dict, T, R, dp_policy_base_dict=None, another_policy_dict=None,
                periods=[0,10,19], shift=0, exp_m=False, save=None, axes=None, add_legend=True):
    """
    """

    aspect_ratio = 1
    base_fig_width = 15

    n_cols = 3
    n_rows = 1

    fig_height = aspect_ratio * (base_fig_width * n_rows / n_cols)
    
    if axes is None:
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols, 
            figsize=(base_fig_width, fig_height))
        
        axes = axes.flatten()

    else:
        fig = axes[0].figure

    m_min = 0
    m_max = 5
    ms = nonlinspace(1e-6, 20, 600, 1.1)
    m_unchanged = (1 + ms*(R-1)) / R

    lines, labels = [], []

    # List of agent dictionaries to loop over
    agent_dicts = ([dp_policy_dict, rl_policy_dict]
                   if dp_policy_base_dict is None 
                   else [dp_policy_dict, dp_policy_base_dict, rl_policy_dict])
    
    agent_dicts = agent_dicts if another_policy_dict is None else agent_dicts + [another_policy_dict]
    
    for j, p in enumerate(periods):
        cs = np.zeros(len(ms))
        ax = axes[j]
        t = p/(T-1)

        # 45-degree line
        line, = ax.plot(ms, ms, color="black", linestyle="-", linewidth=1.0,
                        alpha=0.5)
        if line.get_label() not in labels:
            lines.append(line)

        # m distribution
        if another_policy_dict is not None:
            dicts = [dp_policy_dict, rl_policy_dict, another_policy_dict]
        else:
            dicts = [dp_policy_dict, rl_policy_dict]

        for d in dicts:
            if d == rl_policy_dict or d == another_policy_dict:
                alpha = 0.5
            else:
                alpha = 0.25

            bin_width = 0.25
            bin_edges = np.arange(d['agent'].history.m[:, p].min(), 
                                d['agent'].history.m[:, p].max() + bin_width, 
                                bin_width)

            hist, _, _ = ax.hist(
                d['agent'].history.m[:, p],
                bins=bin_edges,
                density=True,
                color=d['color'],
                edgecolor=None,
                alpha=alpha
            )

        # E(change in m) = 0 line
        if exp_m == True:
            line, = ax.plot(ms, m_unchanged, color="green", linestyle="-", linewidth=1.0,
                            alpha=0.8, label="E($\\Delta m_{t+1}$) = 0")
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

        # Plot the agent policies
        for agent_dict in agent_dicts:
            agent = agent_dict['agent']
            color = agent_dict['color']
            linestyle = agent_dict['linestyle']
            linewidth = agent_dict['linewidth']
            label = agent_dict['label']
            alpha = agent_dict['alpha']

            if agent_dict == dp_policy_base_dict:
                shift = 0

            # Compute policy for each m in ms
            for i, m in enumerate(ms):
                state = np.array([m, t])
                c_share = agent.select_action(state, noise=False, shift=shift)
                c = c_share * m
                cs[i] = c

            line, = ax.plot(ms, cs, color=color, label=label, linestyle=linestyle,
                            linewidth=linewidth, alpha=alpha)
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

        # Axis formatting for this subplot
        ax.set_title(f'Policy at $t={p}$')
        ax.set_xticks(np.arange(m_max+0.5, step=1.0))
        ax.set_ylim(bottom=m_min, top=(m_max if p==19 else m_max-2))
        ax.set_xlim(left=m_min, right=m_max)
        ax.set_xlabel('Cash-on-hand, $m_{t}$')

    if exp_m == True:
        custom_order = [
            dp_policy_dict['label'], 
            rl_policy_dict['label'],
            "E($\\Delta m_{t+1}$) = 0", 
            "c_t = m_t"]
    else:
        custom_order = [
            dp_policy_dict['label'], 
            rl_policy_dict['label'],
            "c_t = m_t"]

    if add_legend: 
        custom_lines = [lines[labels.index(lbl)] for lbl in custom_order if lbl in labels]
        custom_labels = [lbl for lbl in custom_order if lbl in labels]

        fig.legend(
            custom_lines, 
            custom_labels, 
            ncol=2, 
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0), 
            frameon=False)
    
    if axes is None:
        fig.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close(fig)

def plot_learning(dp_dict, rl_dicts, n_episodes, save=None):
    """
    """

    # Keep aspect ratio of subplots!
    aspect_ratio = 1
    base_fig_width = 15

    n_cols = 3
    n_rows = 1

    fig_height = aspect_ratio * (base_fig_width * n_rows / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(base_fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    labels = []
    lines = []

    # RL learning vars
    vars = ['c_loss', 'a_loss', 'euler_error']
    vars_names = ['Value loss', 'Policy loss', 'Relative Euler errors']
    
    for i in range(len(vars)):
        
        axes[i].set_title(vars_names[i])

        if vars[i] != 'm_dist':
            x_axis = np.arange(n_episodes)
            axes[i].set_xticks(np.arange(n_episodes, step=500))
            axes[i].set_xlim([0, n_episodes])
            axes[i].set_xlabel('Life cycles')
        if vars[i] == 'c_loss':
            axes[i].set_xlim([0, 501])
            axes[i].set_xticks(np.arange(n_episodes, step=100))
        else:
            axes[i].set_xlabel("Cash-on-hand, $m_t$")

        # DP value            
        if vars[i] == 'value':
            dp_value = dp_dict['value']
            dp_value = np.ones_like(x_axis) * dp_value.mean(axis=0)

            line_dp, = axes[i].plot(
                x_axis,
                dp_value,
                linewidth=dp_dict['linewidth'],
                linestyle=dp_dict['linestyle'],
                label=dp_dict['label'], 
                color=dp_dict['color']
            )
            
            if line_dp.get_label() not in labels:
                labels.append(line_dp.get_label())
                lines.append(line_dp)

        # RL vars
        for rl_dict in rl_dicts:
            
            var = rl_dict[vars[i]]

            # Everything else but m_dist
            if vars[i] != 'm_dist':
                line_rl, = axes[i].plot(
                    x_axis,
                    var,
                    linestyle='-',
                    linewidth=rl_dict['linewidth'],
                    color=rl_dict['color'],
                    label=rl_dict['label'],
                    alpha=0.8
                )

                if line_rl.get_label() not in labels:
                    labels.append(line_rl.get_label())
                    lines.append(line_rl)

            # m_dist
            else:
                line_rl, _, _ = axes[i].hist(
                    var, 
                    bins=30,
                    density=True,
                    color=rl_dict['color'], 
                    alpha=0.5,
                    edgecolor=None
                )

    all_labels = [dp_dict['label']] + [rl['label'] for rl in rl_dicts]
    custom_order = sorted(all_labels)

    custom_lines = [lines[labels.index(lbl)] for lbl in custom_order if lbl in labels]
    custom_labels = [lbl for lbl in custom_order if lbl in labels]

    fig.legend(
        custom_lines, 
        custom_labels, 
        ncol=4, 
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0), 
        frameon=False)

    fig.tight_layout()
    fig.savefig(save, format="pgf", bbox_inches="tight")
    plt.close(fig)

def plot_fig1_policy(rl_dict, dp_dict, rl_policy_dict, dp_policy_dict, T, R, exp_m=False,
                     other_dict=None, other_policy_dict=None, 
                     another_dict=None, another_policy_dict=None,
                     periods=[0,10,19], save=None, use_master_legend=True):
    """
    """

    subplot_width = 5    # inches
    subplot_height = 5   # inches
    total_width = 3 * subplot_width    # 15 inches wide (3 columns)
    total_height = 2 * subplot_height    # 10 inches tall (2 rows)

    fig = plt.figure(figsize=(total_width, total_height))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[subplot_height, subplot_height], hspace=0.3)

    gs_top = gs[0].subgridspec(nrows=1, ncols=3, wspace=0.25, hspace=0.4)
    axes_fig1 = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    gs_bottom = gs[1].subgridspec(nrows=1, ncols=3, wspace=0.25, hspace=0.4)
    axes_policy = [fig.add_subplot(gs_bottom[0, i]) for i in range(3)]

    plot_fig1(rl_dict, dp_dict, other_dict=other_dict, another_dict=another_dict, save=None, axes=axes_fig1, add_legend=not use_master_legend)
    plot_policy(rl_policy_dict, dp_policy_dict, T, R, exp_m=exp_m, dp_policy_base_dict=other_policy_dict, another_policy_dict=another_policy_dict, periods=periods, save=None, axes=axes_policy, add_legend=not use_master_legend)

    if use_master_legend:
        all_handles = []
        all_labels = []
        for ax in axes_fig1 + axes_policy:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        unique = {}
        for handle, label in zip(all_handles, all_labels):
            if label not in unique:
                unique[label] = handle
        master_handles = list(unique.values())
        master_labels = list(unique.keys())

        sorted_legend = sorted(zip(master_labels, master_handles), key=lambda x: x[0])
        master_labels, master_handles = zip(*sorted_legend)

        fig.legend(
            master_handles, 
            master_labels, 
            loc="lower center", 
            bbox_to_anchor=(0.5, -0.075),
            ncol=len(master_handles),
            frameon=False)

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close(fig)

    else:
        plt.show()

def plot_fig3_policy(rl_shock_dict, dp_shock_dict, dp_dict, rl_policy_dict, dp_policy_dict, shock_period,
                     T, R, periods=[5,10,19], save=None, use_master_legend=True, shift=0):
    """
    """

    subplot_width = 5    # inches
    subplot_height = 5   # inches
    total_width = 3 * subplot_width    # 15 inches wide (3 columns)
    total_height = 2 * subplot_height    # 10 inches tall (2 rows)

    fig = plt.figure(figsize=(total_width, total_height))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[subplot_height, subplot_height], hspace=0.3)

    gs_top = gs[0].subgridspec(nrows=1, ncols=3, wspace=0.25, hspace=0.4)
    axes_fig1 = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    gs_bottom = gs[1].subgridspec(nrows=1, ncols=3, wspace=0.25, hspace=0.4)
    axes_policy = [fig.add_subplot(gs_bottom[0, i]) for i in range(3)]

    plot_fig3(rl_shock_dict, dp_shock_dict, dp_dict, shock_period, save=None, axes=axes_fig1, add_legend=not use_master_legend)
    plot_policy(rl_policy_dict, dp_policy_dict, T, R, periods=periods, shift=shift, save=None, axes=axes_policy, add_legend=not use_master_legend)

    if use_master_legend:
        all_handles = []
        all_labels = []
        for ax in axes_fig1 + axes_policy:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        unique = {}
        for handle, label in zip(all_handles, all_labels):
            if label not in unique:
                unique[label] = handle
        master_handles = list(unique.values())
        master_labels = list(unique.keys())

        sorted_legend = sorted(zip(master_labels, master_handles), key=lambda x: x[0])
        master_labels, master_handles = zip(*sorted_legend)

        fig.legend(
            master_handles, 
            master_labels, 
            loc="lower center", 
            bbox_to_anchor=(0.5, -0.075),
            ncol=len(master_handles), 
            frameon=False)

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ----------
# - Prints -
# ----------

def print_avg_values(*agents, var_list):
    """
    Print average values of variables for multiple agents.
    """
    for var in var_list:
        print(f'{var}')
        print('-' * 30)

        for agent in agents:
            avg = np.mean(getattr(agent.history, var))
            std = np.std(getattr(agent.history, var))
            print(f'{agent.history.name} average: \t {avg:.2f}')
            print(f'{agent.history.name} std: \t\t {std:.2f}')

        print('')

# ----------
# - Tables -
# ----------

def extract_metrics(agent, metrics):
    """
    """
    results = {}

    for metric in metrics:
        data = getattr(agent.history, metric)
        
        results[f'avg_{metric}'] = np.nanmean(data)
        results[f'p5_{metric}'] = np.nanpercentile(data, 5)
        results[f'p95_{metric}'] = np.nanpercentile(data, 95)
        results[f'std_{metric}'] = np.nanstd(data)
        if metric == 'a':
            results[f'last_{metric}'] = np.mean(data[:,-1])

    return results

def extract_all_metrics(shocks, metrics):
    """
    """
    results = {}

    for shock_name, agents in shocks.items():
        results[shock_name] = {
            agent.name: extract_metrics(agent, metrics) 
            for agent in agents
        }

    return results

def create_generalized_table(results, params):
    """
    """
    shocks = [shock for shock in results.keys() if shock != "baseline"]
    baseline = results.get("baseline", {})
    baseline_agents = baseline.keys()
    shock_agents = {
        agent for shock in shocks 
        for agent in results[shock].keys() 
        if agent != "HTM"
    }

    def format_shock_name(shock_name):
        param, value = shock_name.split("=")
        pretty_name = params[param][1]
        if shock_name == "baseline":
            return "Baseline"
        return f"${pretty_name} = {value}$"

    lines = []
    lines.append("{\\small")

    # Header: Shock labels
    total_cols = len(baseline_agents) + len(shock_agents) * len(shocks)
    lines.append(
        "\\begin{tabular}{" + "l" + "c" * total_cols + "}"
    )
    lines.append("\\hline")
    lines.append(
        " & \\multicolumn{" + f"{len(baseline_agents)}" + 
        "}{c}{Baseline} & " +
        " & ".join(
            f"\\multicolumn{{{len(shock_agents)}}}{{c}}"
            f"{{{format_shock_name(shock)}}}" 
            for shock in shocks
        ) +
        " \\\\ \\hline"
    )

    # Header: Agent labels
    lines.append(
        " & " + " & ".join(baseline_agents) + " & " +
        " & ".join(" & ".join(shock_agents) for _ in shocks) +
        " \\\\ \\hline"
    )

    # Define metrics and their labels
    metric_labels = {
        "c": "\\textbf{Consumption}",
        "a": "\\textbf{Assets}",
        "m": "\\textbf{Cash-on-hand}",
        "value": "\\textbf{Life-time utility}",
        "trans_euler_error": "\\textbf{Relative Euler errors}",
        "constrained": "\\textbf{Share of constrained HHs}"
    }

    stat_labels = {
        "avg": "Average",
        "p5": "5th percentile",
        "p95": "95th percentile",
        "std": "Standard deviation",
        'last': "Average last period"
    }

    # Add grouped rows for metrics
    for metric, metric_label in metric_labels.items():
        
        # Add metric group header
        lines.append(f"\\text{{{metric_label}}} & " + " & ".join([""] * total_cols) + " \\\\")
        
        # Add sub-rows for statistics
        for stat, stat_label in stat_labels.items():
            
            row = f"{stat_label}"

            disregard_a = stat == "last" and metric != "a"
            disregard_con = stat == "p5" and metric == "constrained"

            if disregard_a or disregard_con:
                break
            
            for agent in baseline_agents:
                value = baseline.get(agent, {}).get(f"{stat}_{metric}", "N/A")
                row += (
                    f" & {value:.2f}" 
                    if isinstance(value, (int, float)) else 
                    f" & {value}"
                )
            
            for shock in shocks:
                for agent in shock_agents:
                    value = results[shock].get(agent, {}).get(f"{stat}_{metric}", "N/A")
                    row += (
                        f" & {value:.2f}" 
                        if isinstance(value, (int, float)) else 
                        f" & {value}"
                    )
            
            row += " \\\\"
            lines.append(row)

    # Footer
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")

    return "\n".join(lines)
