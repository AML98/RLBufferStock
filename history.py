import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings


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

    def record_step(self, env, episode, agent, shift=0):
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
            value = func(self, env, episode, agent, shift=shift)

            if len(var['shape']) != 1:
                getattr(self, var['name'])[episode, period] = value

            elif env.period == env.T-1:
                getattr(self, var['name'])[episode] = value
    
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

    def compute_value(self, env, episode, agent, shift=0):
        """
        Compute the value of an episode
        """
        discount = env.beta ** np.arange(env.T)
        return np.sum(self.utility[episode] * discount)
    
    def compute_euler_error(self, env, episode, agent, shift=0):
        """
        Compute the relative Euler error
        """
        t = env.period
        e = episode

        c = self.c[e, t]
        a = self.a[e, t]

        # No Euler error in the last period
        if t == env.T - 1:
            euler_error = np.nan
        
        else:
            # Right-hand side
            rhs = 0.0            
            for idx in range(env.Nshocks):
                
                # i. shocks
                psi_w = env.psi_w[idx]
                xi_w = env.xi_w[idx]
                psi = env.psi[idx]
                xi = env.xi[idx]

                # ii. weight
                weight = psi_w * xi_w

                # iii. next-period states
                m_plus = (env.R / psi) * a + xi

                # iv. next-period consumption
                c_share = agent.select_action(
                    state=np.array([m_plus,(t+1)/(env.T-1)]),
                    noise=False,
                    shift=shift
                )
                c_plus = c_share * m_plus

                # v. next-period marginal utility
                rhs += weight * env.beta * env.R * ((c_plus+1e-8) ** (-env.rho))

            # Disregard constrained
            constrained = ((c + a)**(-env.rho) >= rhs)

            if constrained:
                euler_error = np.nan
            else:
                euler_error = c - rhs**(-1/env.rho)

        return euler_error
    
    def compute_trans_euler_error(self, env, n_episodes):
        """
        """
        self.trans_euler_error = np.zeros([n_episodes,env.T])
        self.constrained = np.zeros([n_episodes,env.T])

        for t in range(env.T):
            rel_euler_errors = self.euler_error[:, t]/self.c[:, t]
            self.constrained[:, t] = np.isnan(rel_euler_errors).astype(int)
            self.trans_euler_error[:,t] = np.log10(np.abs(rel_euler_errors))
    
    # -----------------------
    # - Functions for plots -
    # -----------------------

    def plot_avg_trajectory2(self, var, color, linestyle, linewidth,
        linestart, cf_start, label, alpha=1.0, ax=None):
        """
        """

        # Use the provided axis or get the current axis
        if ax is None:
            ax = plt.gca()

        data = getattr(self, var)
        
        # Use nanmean and nanstd due to Euler errors
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

            var_avg = np.nanmean(data, axis=0)
            var_std = np.nanstd(data, axis=0)
            var_p5 = np.nanpercentile(data, 5, axis=0)
            var_p95 = np.nanpercentile(data, 95, axis=0)
 
        num_periods = getattr(self, var).shape[1]  # Shape[1] = T
        ax.set_xticks(np.arange(num_periods, step=2))
        ax.set_xlim([0, num_periods-1])
        ax.set_xlabel('$t$')

        x_axis = np.arange(num_periods)

        # Plot average trajectory
        line, = ax.plot(
            x_axis[linestart:], 
            var_avg[linestart:], 
            label=label, 
            color=color, 
            linestyle=linestyle,
            linewidth=linewidth, 
            alpha=alpha
        )
            
        # ax.scatter(
        #     x_axis[linestart:],
        #     var_avg[linestart:],
        #     label=None,
        #     color=color,
        #     alpha=alpha,
        #     s=5
        # )    

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

        data = getattr(self, var)

        # Use nanstd and nanstd due to Euler errors
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
            var_std = np.nanstd(data, axis=0)
        
        num_periods = getattr(self, var).shape[1]  # Shape[1] = T
        ax.set_xticks(np.arange(num_periods, step=2))
        ax.set_xlim([0, num_periods-1])
        ax.set_xlabel('$t$')

        x_axis = np.arange(num_periods)

        # Plot average trajectory
        line, = ax.plot(x_axis[linestart:], var_std[linestart:], 
            label=label, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha)
            
        return line      
    
    def plot_value(self, color, linestyle, linewidth,
        label, alpha=1.0, ax=None):
        """
        Plot the value across episodes.
        """
        if ax is None:
            ax = plt.gca()
        
        data = getattr(self, 'value')

        num_episodes = getattr(self, 'value').shape[0]  # Shape[0] = episodes
        ax.set_xticks(np.arange(num_episodes, step=100))
        ax.set_xlim([0, num_episodes-1])
        ax.set_xlabel('Episode')

        x_axis = np.arange(num_episodes)

        line, = ax.plot(x_axis, data, color=color, linestyle=linestyle,
            linewidth=linewidth, label=label)

        return line
    
    def plot_value_error(self, color, linestyle, linewidth,
        label, alpha=1.0, ax=None):
        """
        Plot the value across episodes.
        """
        if ax is None:
            ax = plt.gca()
        
        data = getattr(self, 'value')

        num_episodes = getattr(self, 'value').shape[0]