import pickle
import numpy as np
import matplotlib.pyplot as plt

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


class History:
    def __init__(self, name, env, n_episodes, env_vars=None, 
        custom_vars=None):
        """
        env_vars: 
            ['var_name1', 'var_name2', 'var_name3', ...]
        custom_vars: 
            [{'name': str, 'shape': tuple, 'function': function}, ...]
        """
        self.name = name
        
        # Use all env vars if none are specified
        if env_vars:
            self.env_vars = env_vars
        else:
            self.env_vars = [var['name'] for var in env.vars]
        
        self.custom_vars = custom_vars if custom_vars else []

        # Allocate memory
        for var in self.env_vars:
            setattr(self, var, np.zeros((n_episodes, env.T)))

        for var in self.custom_vars:
            setattr(self, var['name'], np.zeros(var['shape']))
        
    # ------------------
    # - Public methods -
    # ------------------

    def record_step(self, env, episode):
        """
        Record a single interaction at a specific step.
        """
        period = env.period

        # Record vars
        for var in self.env_vars:
            value = getattr(env, var)
            getattr(self, var)[episode, period] = value

        for var in self.custom_vars:
            func = var['function']
            value = func(self, env, episode)

            if len(var['shape']) != 1:
                getattr(self, var['name'])[episode, period] = value

            elif env.period == env.T-1:
                getattr(self, var['name'])[episode] = value

    def plot_avg_trajectory(self, var, color, linestyle, 
        con_bands, var_legends, start_period, ax=None):
        """
        Plot average trajectories of variables.

        Parameters
        ----------
        var : str
            The name of the environment variable to plot.
        
        color : str or tuple
            The color to use for the plot line and confidence band.
        
        linestyle : str
            Line style for the plot, such as '-', '--', or '-.'.
        
        con_bands : bool
            If True, a confidence band of one std will be plotted.
        
        var_legends : bool
            If True, the plot legend will include the variable name.
        
        start_period : int
            Specifies the starting period for plotting the confidence 
            bands. If set to 0, the band spans the entire time range. 
            For values greater than 0, the confidence band to start
            at that period.
        
        ax : matplotlib.axes._axes.Axes, optional
            An existing matplotlib Axes object on which to plot. If 
            None, a new Axes object will be created using `plt.gca()`.

        Returns
        -------
        line : matplotlib.lines.Line2D
            The line object representing the plotted average trajectory,
            which can be used for further customization or legend
            creation.
        """

        # Use the provided axis or get the current axis
        if ax is None:
            ax = plt.gca()

        # Layout
        num_periods = getattr(self, var).shape[1]  # Shape[1] = T
        ax.set_xticks(np.arange(num_periods, step=2))
        ax.set_xlim([0, num_periods-1])
        ax.set_xlabel('$t$')

        # Get data
        data = getattr(self, var)
        var_std = np.std(data, axis=0)
        var_avg = np.mean(data, axis=0)
        x_axis = np.arange(num_periods)

        # Plot average trajectory
        if var_legends:
            line, = ax.plot(x_axis, var_avg, label=f'${var}_t$, {self.name}',
                color=color, linestyle=linestyle)
        else:
            line, = ax.plot(x_axis, var_avg, label=f'{self.name}', 
                color=color, linestyle=linestyle)
            
        # Plot confidence band
        if con_bands:
            upper = var_avg[start_period:] + var_std[start_period:]
            lower = var_avg[start_period:] - var_std[start_period:]
            x_axis = x_axis[start_period:]

            ax.fill_between(x_axis, lower, upper, color=color,
                alpha=0.2)
            
        return line

    def save(self, filename=None):
        """
        Save the entire History object to a file using pickle.
        """
        if filename is None:
            filename = f'history_{self.name}.pkl'
        
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a History object from a file using pickle.
        """
        with open(filename, 'rb') as file:
            history = pickle.load(file)

        return history
    
    # -------------------
    # - Private methods -
    # -------------------

    def _compute_value(self, env, episode):
        """
        Compute the value of an episode.
        """
        period = env.period

        return np.sum(self.utility[episode] * env.beta ** period)
    
    def _compute_savings_rate(self, env, episode):
        """
        Compute the savings rate of an episode.
        """
        period = env.period

        return self.a[episode, period] / self.m[episode, period]