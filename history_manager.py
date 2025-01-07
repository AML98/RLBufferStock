import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


class HistoryManager:
    def __init__(self, name):
        self.name = name
        self.n_runs = 0      # Number of runs
        self.histories = []  # List to store History objects

    def add_history(self, history):
        """Add a History object to the manager."""
        self.histories.append(history)
        self.n_runs += 1

    def aggregate_var(self, var_name):
        """
        Aggregate a variable across all runs.

        Parameters
        ----------
        var_name : str
            The name of the variable to aggregate.

        Returns
        -------
        aggregated_data : np.ndarray
            A 3D array of shape (n_runs, n_episodes, T) for time-series data,
            or (n_runs, n_episodes) for per-episode data.
        """
        data_list = []
        for history in self.histories:
            data = getattr(history, var_name)
            data_list.append(data)

        aggregated_data = np.array(data_list)  # Shape: (n_runs, ...)
        return aggregated_data

    def compute_average(self, var_name):
        """
        Compute the average of a variable across runs.

        Parameters
        ----------
        var_name : str
            The name of the variable to average.

        Returns
        -------
        avg_data : np.ndarray
            The averaged data across runs.
        """
        aggregated_data = self.aggregate_var(var_name)
        avg_data = np.mean(aggregated_data, axis=0)
        return avg_data

    def compute_std(self, var_name):
        """
        Compute the standard deviation of a variable across runs.

        Parameters
        ----------
        var_name : str
            The name of the variable.

        Returns
        -------
        std_data : np.ndarray
            The standard deviation across runs.
        """
        aggregated_data = self.aggregate_var(var_name)
        std_data = np.std(aggregated_data, axis=0)
        return std_data

    def save(self, filename=None):
        """
        Save the HistoryManager object to a file using pickle.
        """
        folder = "histories"
        if filename is None:
            filename = f'history_manager_{self.name}.pkl'
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a HistoryManager object from a file using pickle.
        """
        folder = "histories"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'rb') as file:
            history_m = pickle.load(file)
        return history_m

    def plot_average_value(self, color, linestyle, mean=False, window_size=None, 
        ax=None):
        """
        Plot the average value across episodes for all histories.
        """
        # Get data
        avg_value = self.compute_average('value')  # Shape: (num_episodes,)
        std_value = self.compute_std('value')      # Shape: (num_episodes,)

        # Optional moving average
        if window_size is not None:
            avg_value = np.convolve(avg_value, 
                np.ones(window_size) / window_size, mode='valid')
            std_value = np.convolve(std_value, 
                np.ones(window_size) / window_size, mode='valid')
        x_axis = np.arange(len(avg_value))

        # Plot the average value
        label = self.histories[0].name
        if mean:
            avg_value = np.mean(avg_value) * np.ones_like(avg_value)
        line, = ax.plot(x_axis, avg_value, label=label, color=color,
            linestyle=linestyle, linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Episode')
        ax.legend()

        return line
    
    def plot_avg_trajectory(self, var, episode, color, linestyle,
        ax=None):
        """
        """
        # Use the provided axis or get the current axis
        if ax is None:
            ax = plt.gca()

        # Get data
        var_avg = self.compute_average(var)  # Shape: (n_episodes, T)
        var_avg = var_avg[episode]           # Shape: (T,)
        num_periods = var_avg.shape[0]
        x_axis = np.arange(num_periods)

        # Layout
        ax.set_xticks(np.arange(num_periods, step=2))
        ax.set_xlim([0, num_periods-1])
        ax.set_xlabel('$t$')

        # Plot average trajectory
        line, = ax.plot(x_axis, var_avg, label=f'{self.name}', 
            color=color, linestyle=linestyle)
            
        return line