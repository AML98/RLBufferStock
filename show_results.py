import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------
# - Plots -
# ---------

def plot_avg_trajectories_separate(*items, var_list, con_bands=True, 
    start_period1=0, start_period2=0, train_agents=None, save=None):
    """
    Plot average trajectories of multiple variables in separate 
    subplots, arranged in a 2-column layout.

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

    # Layout
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

        # Colors and line styles to differentiate between items
        colors = ['blue', 'red', 'green']
        linestyles = ['-', '--', '-.']

        for j in range(len(items)):
            item = items[j]
            color = colors[j]
            linestyle = linestyles[j]
            history = item.history if hasattr(item, 'history') else item

            # If 1st item, plot con_band from initial period
            if j == 0:
                line = history.plot_avg_trajectory(var, color=color,
                    linestyle=linestyle, con_bands=con_bands, 
                    var_legends=False, ax=ax, start_period=
                    start_period1)
            
            # If 2nd item, plot con_band from specified start period
            else:
                line = history.plot_avg_trajectory(var, color=color, 
                    linestyle=linestyle, con_bands=con_bands,
                    var_legends=False, ax=ax, start_period=
                    start_period2)
            
            ax.set_title(f'${var}_t$')

            # Add only for 1st var to avoid duplicate legends
            if i == 0:
                lines.append(line)
                labels.append(f'{history.name}')

        # Plot training process  
        if train_agents is not None:
            colors_train = plt.cm.Blues(np.linspace(0.1, 0.9, 
                len(train_agents)))

            for k, agent in enumerate(train_agents):
                line = agent.history.plot_avg_trajectory(var, 
                    color=colors_train[k], linestyle='-', 
                    con_bands=False, var_legends=False,
                    start_period=0, linewidth=1.0, 
                    alpha=1.0, ax=ax)

        # Shock line
        if start_period2 != 0:
            ax.axvline(x=start_period2, color='black', linewidth=1)

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
