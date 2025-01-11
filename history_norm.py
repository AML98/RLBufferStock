import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
    
    def plot_avg_trajectory2(self, var, color, linestyle, linewidth,
        linestart, cf_start, label, alpha=1.0, ax=None):
        """
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
        line, = ax.plot(x_axis[linestart:], var_avg[linestart:], 
            label=label, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha)
            
        # Plot confidence band
        if cf_start is not None:
            upper = var_avg + var_std
            lower = var_avg - var_std
            x_axis = x_axis

            ax.fill_between(x_axis[cf_start:], lower[cf_start:], 
                upper[cf_start:], color=color, alpha=0.2)
            
        return line    

    def plot_sd2(self, var, color, linestyle, linewidth,
        linestart, label, alpha=1.0, ax=None):
        """
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
        x_axis = np.arange(num_periods)

        # Plot average trajectory
        line, = ax.plot(x_axis[linestart:], var_std[linestart:], 
            label=label, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha)
            
        return line      
    
    def plot_value(self, color, linestyle, ax=None):
        """
        Plot the value across episodes.
        """
        if ax is None:
            ax = plt.gca()
        
        # Layout
        num_episodes = getattr(self, 'value').shape[0]  # Shape[0] = episodes
        ax.set_xticks(np.arange(num_episodes, step=100))
        ax.set_xlim([0, num_episodes-1])
        ax.set_xlabel('Episode')

        # Get data
        data = getattr(self, 'value')
        x_axis = np.arange(num_episodes)

        # Plot value
        line, = ax.plot(x_axis, data, label=f'{self.name}',
                color=color, linestyle=linestyle)

        return line

    def save(self, filename=None):
        """
        Save the entire History object to a file using pickle.
        """
        folder = "histories"
        if filename is None:
            filename = f'history_{self.name}.pkl'
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a History object from a file using pickle.
        """
        folder = "histories"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'rb') as file:
            history = pickle.load(file)

        return history
    
    # -------------------
    # - Private methods -
    # -------------------

    def _compute_value(self, env, episode):
        """
        Compute the value of an episode.
        """
        discount = env.beta ** np.arange(env.T)
        return np.sum(self.utility[episode] * discount)
    
    # ----------------------------------------------------------------
    #  1) Plot c vs. (m/p) using simulated data
    # ----------------------------------------------------------------
    def plot_consumption_vs_wealth_by_t(
        self, 
        t, 
        ax=None, 
        marker='o', 
        alpha=0.3, 
        do_sort=True,
        do_bin=False,
        bins=20
    ):
        """
        Plot the empirical policy function c_t vs. (m_t / p_t) 
        for a *specific* time t (finite-horizon setting).
        
        Parameters
        ----------
        t : int
            The time/period for which we want the policy function.
        ax : matplotlib.axes.Axes, optional
            Axis object on which to plot. If None, will use current axis.
        marker : str, optional
            Marker style for scatter. Default 'o'.
        alpha : float, optional
            Transparency for points. Default 0.3.
        do_sort : bool, optional
            If True, sort the data by x so that the line-plot is clearer.
            (Often helpful if do_bin=False.)
        do_bin : bool, optional
            If True, bin x-values and plot mean c in each bin 
            rather than plotting individual points.
        bins : int, optional
            Number of bins if do_bin=True.
        """
        if ax is None:
            ax = plt.gca()

        # shape of self.m is (n_episodes, T).
        # We extract the column t across all episodes.
        m_col = self.m[:, t]  # shape (n_episodes,)
        p_col = self.p[:, t]
        c_col = self.c[:, t]

        # Filter out cases where p == 0
        valid_mask = (p_col > 0)
        m_col = m_col[valid_mask]
        p_col = p_col[valid_mask]
        c_col = c_col[valid_mask]

        x_col = m_col / p_col  # (m_t / p_t)

        range_mask = (x_col > 0) & (x_col < 5)
        x_col = x_col[range_mask]
        c_col = c_col[range_mask]

        if do_bin:
            # Bin the data into intervals of x
            # then plot the average c in each bin
            bin_edges = np.linspace(x_col.min(), x_col.max(), bins+1)
            idx = np.digitize(x_col, bin_edges) - 1  # which bin each point goes to
            c_means = []
            x_centers = []
            for b in range(bins):
                in_bin = (idx == b)
                if np.any(in_bin):
                    c_means.append(np.mean(c_col[in_bin]))
                    # midpoint of the bin
                    x_centers.append(0.5*(bin_edges[b] + bin_edges[b+1]))
            
            ax.plot(x_centers, c_means, marker='o', linestyle='-', alpha=0.8)
        else:
            # Either scatter everything,
            # or sort + then do a line or scatter plot
            if do_sort:
                sort_idx = np.argsort(x_col)
                x_sorted = x_col[sort_idx]
                c_sorted = c_col[sort_idx]
                ax.plot(x_sorted, c_sorted, marker=marker, linestyle='none', alpha=alpha)
            else:
                # Just scatter raw (x, c)
                ax.scatter(x_col, c_col, marker=marker, alpha=alpha)

        ax.set_xlabel('$m_t / p_t$')
        ax.set_ylabel(f'$c_t$')
        ax.set_title(f'Empirical policy at time t={t} ({self.name})')
        return ax

    # ----------------------------------------------------------------
    #  2) Compute Euler errors to assess accuracy
    # ----------------------------------------------------------------
    def compute_euler_errors(self, rho, beta, R, ignore_borrowing_binds=True):
        """
        Compute the relative Euler error for each (episode, period) where the household
        is *not* liquidity constrained. The standard CRRA Euler condition is:
        
            c_t^(-rho)  =  beta * R * E[ c_{t+1}^(-rho) ]

        We'll compute a "relative log-error" measure using Equation (3.1) from the paper:
        
            E_t = log10( | Delta_t / c_t | )
        
        where Delta_t is the deviation from the Euler equation.

        If ignore_borrowing_binds=True, skip periods where assets=0 (corner solutions).

        Returns: euler_errors, shape = (n_episodes, T)
        """
        n_episodes, T = self.c.shape
        euler_errors = np.full((n_episodes, T), np.nan)

        for ep in range(n_episodes):
            for t in range(T - 1):  # no c_{t+1} for the last period
                # Skip if ignoring borrowing binds and assets are effectively zero
                if ignore_borrowing_binds and self.a[ep, t] <= 1e-12:
                    continue

                c_t = self.c[ep, t]
                c_next = self.c[ep, t + 1]

                # If consumption is non-positive, skip the calculation
                if c_t <= 0 or c_next <= 0:
                    continue

                # Compute deviation Delta_t from the Euler equation
                lhs = c_t ** (-rho)
                rhs = beta * R * (c_next ** (-rho))
                delta = lhs - rhs

                # Relative Euler error: log10(|Delta_t / c_t|)
                relative_error = np.log10(np.abs(delta / c_t))
                euler_errors[ep, t] = relative_error

        return euler_errors

    def plot_euler_errors(self, rho, beta, R, ignore_borrowing_binds=True, ax=None):
        """
        Compute Euler errors and plot their average over time.
        """
        if ax is None:
            ax = plt.gca()

        euler_errors = self.compute_euler_errors(rho, beta, R,
                                    ignore_borrowing_binds=ignore_borrowing_binds)

        # We'll take the absolute value of the errors and average across episodes
        abs_errors = np.nanmean(np.abs(euler_errors), axis=0)  # shape (T,)
        x_axis = np.arange(len(abs_errors))

        ax.plot(x_axis, abs_errors, label=f'{self.name} Euler Error')
        ax.set_xlabel('t')
        ax.set_ylabel('Mean |Euler Error|')
        ax.set_title('Euler Errors over time')
        ax.legend()

        return ax