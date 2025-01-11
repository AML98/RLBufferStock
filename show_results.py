import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import math

# ---------
# - Plots -
# ---------

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

def plot_fig1(rl_dict, dp_dict, save=None):
    """
    """
    # Keep aspect ratio of subplots!
    aspect_ratio = 1
    base_fig_width = 15

    n_cols = 3
    n_rows = 2

    fig_height = aspect_ratio * (base_fig_width * n_rows / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(base_fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    var_list = ['m', 'c', 'a']
    var_names = ['$m_t$', '$c_t$', '$a_t$']

    # For legend
    lines, labels = [], []

    for i, var in enumerate(var_list):
        # --- Plot in row 1 ---
        title = var_names[i]
        axes[i].set_title(title)

        for d in [rl_dict, dp_dict]:
            line = plot_avg_trajectory(d, var, axes[i])
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)
        
        # --- Plot in row 2 ---
        title = f'SD({var_names[i]})'
        axes[i + n_cols].set_title(title)

        for d in [rl_dict, dp_dict]:
            line = plot_sd(d, var, axes[i + n_cols])
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

    # Legends
    custom_order = [
        rl_dict['label'],
        dp_dict['label']
    ]

    custom_lines = [lines[labels.index(lbl)] for lbl in custom_order]
    custom_labels = [lbl for lbl in custom_order]

    fig.legend(custom_lines, custom_labels, loc="upper center", 
            bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_fig2(rl_shock_dict, rl_dict, dp_shock_dict, dp_dict, var_list, shock_line, save=None):
    """
    Plots trajectories for the given dictionaries and variables in a
    fixed 2-row layout (shock row vs normal row), and a dynamic number
    of columns = len(var_list), while preserving the SAME subplot aspect ratio.
    """
    # Keep aspect ratio of subplots!
    aspect_ratio = 1
    base_fig_width = 15

    n_cols = len(var_list)
    n_rows = 2

    fig_height = aspect_ratio * (base_fig_width * n_rows / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(base_fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    row_1_dicts = [rl_shock_dict, dp_shock_dict, dp_dict]
    row_2_dicts = [rl_shock_dict, rl_dict]

    # For legend
    lines, labels = [], []

    for i, var in enumerate(var_list):
        # --- Plot in row 1 ---
        ax1 = axes[i]  # first row = 0, columns go i=0..n_cols-1
        ax1.axvline(x=shock_line, color='black', linewidth=1)
        ax1.set_ylim(bottom=0, top=10)
        ax1.set_title(f'${var}_t$')

        for d in row_1_dicts:
            line = plot_avg_trajectory(d, var, ax1)
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

        # --- Plot in row 2 ---
        ax2 = axes[i + n_cols]  # second row = 1, same col i => i + n_cols
        ax2.axvline(x=shock_line, color='black', linewidth=1)
        ax2.set_ylim(bottom=0, top=10)
        ax2.set_title(f'${var}_t$')

        for d in row_2_dicts:
            line = plot_avg_trajectory(d, var, ax2)
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

    # Legends
    custom_order = [
        rl_dict['label'],
        rl_shock_dict['label'], 
        dp_dict['label'],
        dp_shock_dict['label']
    ]

    custom_lines = [lines[labels.index(lbl)] for lbl in custom_order]
    custom_labels = [lbl for lbl in custom_order]

    fig.legend(custom_lines, custom_labels, loc="upper center", 
            bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_fig3(rl_shock_dict, rl_dict, dp_shock_dict, dp_dict, var_list, shock_line, save=None):
    """
    """
    # Keep aspect ratio of subplots!
    aspect_ratio = 1
    base_fig_width = 15

    n_cols = len(var_list)
    n_rows = 2

    fig_height = aspect_ratio * (base_fig_width * n_rows / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(base_fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    row_1_dicts = [rl_shock_dict, dp_shock_dict, dp_dict]
    row_2_dicts = [rl_shock_dict, rl_dict]

    # Containers to use for legend
    labels = []
    lines = []

    for i in range(len(var_list)):
        if var_list[i] == 'c':
            title = 'SD$(c_t)^2$'
            bottom = None
            top = None
            plot_func = plot_sd
        else:
            title = f'${var_list[i]}_t$'
            bottom = 0
            top = 3
            plot_func = plot_avg_trajectory

        # Plot row 1
        axes[i].axvline(x=shock_line, color='black', linewidth=1)
        axes[i].set_ylim(bottom=bottom, top=top)
        axes[i].set_title(title)

        for j in range(len(row_1_dicts)):
            line = plot_func(row_1_dicts[j], var_list[i], axes[i])
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

        # Plot row 2
        index = i + len(var_list)
        axes[index].axvline(x=shock_line, color='black', linewidth=1)
        axes[index].set_ylim(bottom=bottom, top=top)
        axes[index].set_title(title)

        for j in range(len(row_2_dicts)):
            line = plot_func(row_2_dicts[j], var_list[i], axes[index])
            if line.get_label() not in labels:
                labels.append(line.get_label())
                lines.append(line)

    fig.legend(lines, labels, loc="upper center", 
        bbox_to_anchor=(0.5, 0.0), ncol=len(var_list), frameon=False)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_avg_trajectories_separate2(*agent_dicts, var_list, shock_line=None,
    save=None):
    """
    """

    # Subplot structure
    num_cols = 2
    num_plots = len(var_list)
    num_rows = num_plots // num_cols + num_plots % num_cols
    fig, axes = plt.subplots(num_rows, num_cols, 
        figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Containers to use for legend
    lines = []
    labels = []

    for i in range(len(var_list)):
        var = var_list[i]
        ax = axes[i]

        for agent_dict in agent_dicts:
            # Data
            history = agent_dict['history']

            # Layout
            label = agent_dict['label']
            color = agent_dict['color']
            linestyle = agent_dict['linestyle']
            linewidth = agent_dict['linewidth']
            linestart = agent_dict['linestart']
            cf_start = agent_dict['cf_start']

            if 'alpha' in agent_dict:
                alpha = agent_dict['alpha']
            else:
                alpha = 1.0

            # Plot
            line = history.plot_avg_trajectory2(var, color=color,
                linestyle=linestyle, linewidth=linewidth, 
                linestart=linestart, cf_start=cf_start, 
                label=label, alpha=alpha, ax=ax)

            # Add only for 1st var to avoid duplicate legends
            if i == 0:
                lines.append(line)
                labels.append(label)

        ax.set_title(f'${var}_t$')
        ax.set_ylim(bottom=0, top=3)

        # Shock line
        if shock_line is not None:
            ax.axvline(x=shock_line, color='black', linewidth=1)

    fig.legend(lines, labels, loc="upper center", 
        bbox_to_anchor=(0.5, 0.0), ncol=len(var_list), frameon=False)

    # Hide any unused subplots
    for i in range(len(var_list), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_avg_trajectories_together(*items, var_list, con_bands=True,
    start_period1=0, start_period2=0, save=None):
    """
    Plot average trajectories of multiple variables in same plot.

    Parameters
    ----------
    items : tuple
        A collection of items containing trajectory data. Each item 
        should have a `history` attribute or be a 'history' object.
        Variables will be plotted for each item.  
    
    var_list : list of str
        A list of variable names to plot.
    
    con_bands : bool, optional
        If True, includes confidence bands in the plots.
    
    start_period : int, optional
        The period from which to start plotting confidence band of
        the second item in each subplot. The first item is always
        plotted from the beginning.
    
    save : bool, optional
        If True, saves the plot as a `.pgf` file named 
        "plot_trajectories.pgf" in the current directory.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    # Linestyles to differentiate between items
    linestyles = ['-', '--']

    # Colors to differentiate between vars
    colors = ['blue', 'red', 'black']

    # Containers to use for legend
    lines = []
    labels = []

    for i in range(len(var_list)):
        var = var_list[i]
        color = colors[i]

        for j in range(len(items)):
            linestyle = linestyles[j]
            item = items[j]

            history = item.history if hasattr(item, 'history') else item

            # If 1st item, plot con_band from initial period
            if j == 0:
                line = history.plot_avg_trajectory(var, color=color,
                    linestyle=linestyle, con_bands=con_bands, 
                    var_legends=True, ax=ax, start_period=
                    start_period1)
            
            # If 2nd item, plot con_band from specified start period
            else:
                line = history.plot_avg_trajectory(var, color=color, 
                    linestyle=linestyle, con_bands=con_bands,
                    var_legends=True, ax=ax, start_period=
                    start_period2)

            lines.append(line)
            labels.append(f'${var}_t$, {history.name}')

        if start_period2 != 0:
            ax.axvline(x=start_period2, color='black', linewidth=1)

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
        ncol=len(var_list), frameon=False)
        
    plt.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_value(*history_managers, mean1=False, mean2=False,
    window_size=None, save=None):
    """
    Plot the life-time utility averaged across training runs.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Colors to differentiate between histories
    colors = ['blue', 'red', 'green']

    for i, history_man in enumerate(history_managers):
        color = colors[i]
        if i == 1 and mean1:
            line = history_man.plot_average_value(color=color, 
                linestyle='--', mean=mean1, window_size=window_size,
                ax=ax)
        elif i == 2 and mean2:
            line = history_man.plot_average_value(color=color, 
                linestyle='--', mean=mean2, window_size=window_size,
                ax=ax)
        else:
            line = history_man.plot_average_value(color=color, 
                linestyle='-', window_size=window_size,
                ax=ax)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, format="pgf", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_policy(*agents):
    """
    Plot the policy of the agent.
    """
    
    # Only for buffer stock environment 
    ps = np.linspace(0.1, 2, 5)
    ms = np.linspace(0.1, 2, 100)

    for agent in agents:
        cs = np.zeros((len(ps), len(ms)))

        for i,p in enumerate(ps):
            for j,m in enumerate(ms):
                # Get state
                state = np.array([p,m,0.0])
                
                # Calculate consumption
                c_share = agent.select_action(state, noise=False, shift=0)
                c = c_share * m
                cs[i,j] = c                    
    
        for i in range(len(ps)):
            plt.plot(ms, cs[i,:], label=f'{agent}')
            plt.ylabel('Consumption')
            plt.xlabel('Wealth')
            plt.savefig(f'plots/policy_{agent}.png', format='png')

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

import numpy as np
import pandas as pd

def generate_avg_values_table(*agents, var_list, label="tab:avg_values",
    caption="Average Values and Standard Deviations"):
    """
    """
    data = {}
    for agent in agents:
        agent_name = agent.history.name
        data[(agent_name, 'Average')] = []
        data[(agent_name, 'Standard deviation')] = []

    for var in var_list:
        for agent in agents:
            agent_name = agent.history.name
            values = getattr(agent.history, var)
            avg = np.mean(values)
            std = np.std(values)
            data[(agent_name, 'Average')].append(avg)
            data[(agent_name, 'Standard deviation')].append(std)

    columns = pd.MultiIndex.from_tuples(data.keys(), names=['Agent', 
        'Statistic'])
    df = pd.DataFrame(data, index=var_list, columns=columns)
    df.index.name = 'Variable'
    df = df.round(2)

    # Generate LaTeX table
    latex_table = df.to_latex(column_format='l' + 'r' * len(df.columns),
        multirow=True, caption=caption, label=label, escape=False,
        float_format="%.2f")

    print(latex_table)

# ------------------------------
# - Consumption function stuff -
# ------------------------------

def plot_consumption_function(agent_dict, t, ax=None,
                              marker='o', alpha=0.3,
                              do_sort=True, do_bin=False, bins=20):
    """
    Plots the consumption function for a single period t, i.e. c_t vs (m_t/p_t),
    for the given agent_dict on a specified Matplotlib Axes (ax).
    
    Parameters
    ----------
    agent_dict : dict
        A dictionary containing at least:
          - 'history': a History object with .plot_consumption_vs_mOverp_by_t()
          - 'label': a string label (optional)
          - 'color': a color (optional)
          - etc.
    t : int
        The period (in a finite-horizon model) at which to plot the policy.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, uses current axis.
    marker, alpha, do_sort, do_bin, bins
        Passed along to the History.plot_consumption_vs_mOverp_by_t method.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis used for plotting.
    """
    if ax is None:
        ax = plt.gca()
    
    history = agent_dict['history']
    label   = agent_dict['label']
    color   = agent_dict['color']

    # Actually call the method from the History object:
    ax = history.plot_consumption_vs_wealth_by_t(
        t=t,
        ax=ax,
        marker=marker,
        alpha=alpha,
        do_sort=do_sort,
        do_bin=do_bin,
        bins=bins
    )

    # You can optionally add a small text label in the subplot:
    ax.text(0.05, 0.95, label, transform=ax.transAxes, 
            verticalalignment='top', color=color, fontsize=10)

    return ax

def plot_consumption_functions_subplots(
    t,
    agent_dicts,
    ncols=2,
    do_bin=False,
    bins=10,
    marker='o',
    alpha=0.3,
    do_sort=False,
    save=None
):
    """
    Create a figure of subplots, each showing the consumption function c_t vs.
    (m_t / p_t) at time t for a different agent.

    Parameters
    ----------
    t : int
        The time/period (in finite horizon) for which to plot c_t.
    agent_dicts : list of dict
        Each dict should have:
          - 'history': a History object with method plot_consumption_vs_mOverp_by_t
          - 'label': string label
          - 'color': optional color, etc.
    ncols : int
        Number of columns in the subplot grid. Rows will be computed automatically.
    do_bin, bins, marker, alpha, do_sort
        Passed to plot_consumption_function (controls binning, alpha, etc.).
    shock_line : float or int, optional
        If provided, a vertical line is drawn at x=shock_line in each subplot.
        (Sometimes used if x has a special meaning.)
    save : str or None
        If a filename (e.g. "my_plot.png" or "my_plot.pgf") is provided,
        saves the figure to that file. Otherwise displays on screen.
    """
    n_agents = len(agent_dicts)
    nrows = math.ceil(n_agents / ncols)

    # Set up a figure with enough subplots
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),  # adjust as you like
        squeeze=False
    )
    axes = axes.flatten()  # Flatten for easy iteration

    for i, agent in enumerate(agent_dicts):
        ax = axes[i]

        # Plot consumption vs. m/p for this agent on this subplot
        plot_consumption_function(
            agent_dict=agent,
            t=t,
            ax=ax,
            marker=marker,
            alpha=alpha,
            do_sort=do_sort,
            do_bin=do_bin,
            bins=bins
        )

        # Title: let's put the agent's label
        label = agent.get('label', f"Agent {i}")
        ax.set_title(f"{label}: consumption at t={t}")

    # Hide unused axes if the # of agents doesn't fill the grid
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, format='png', bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()