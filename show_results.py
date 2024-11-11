import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# ---------
# - Plots -
# ---------

def plot_avg_trajectories_separate(*items, var_list, con_bands=True, 
    start_period1=0, start_period2=0, save=False):
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

    # Colors and line styles to differentiate between items
    linestyles = ['-', '--']
    colors = ['blue', 'red']

    # Containers to use for legend
    lines = []
    labels = []

    for i in range(len(var_list)):
        var = var_list[i]
        ax = axes[i]

        for j in range(len(items)):
            linestyle = linestyles[j]
            color = colors[j]
            item = items[j]

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

        if start_period2 != 0:
            ax.axvline(x=start_period2, color='black', linewidth=1)

    fig.legend(lines, labels, loc="upper center", 
        bbox_to_anchor=(0.5, 0.0), ncol=len(var_list), frameon=False)

    # Hide any unused subplots
    for i in range(len(var_list), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig("plot_trajectories.pgf", format="pgf", 
            bbox_inches="tight")

def plot_avg_trajectories_together(*items, var_list, con_bands=True,
    start_period1=0, start_period2=0, save=False):
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
    plt.show()

    if save:
        fig.savefig("plot_trajectories.pgf", format="pgf", bbox_inches="tight")

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
