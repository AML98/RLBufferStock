import numpy as np
from consav import linear_interp
from agent import Agent

class DPAgent(Agent):
    def __init__(self, model, env):
        super().__init__()
        self.name = 'DP household'
        self.model = model
        self.env = env

    def select_action(self, state, noise=False, infinite_horizon=None, shift=0):
        """
        Selects consumption given 'm' (and possibly 't') from the environment state.
        
        We do 1D interpolation in sol.c[t, :] over par.grid_m, 
        because the solution is in normalized form (no 'p' dimension).
        """

        par = self.model.par
        sol = self.model.sol

        m = state[0]
        t = round(state[1] * (self.env.T-1))

        if infinite_horizon is True:
            # Use policy from initial period
            c = linear_interp.interp_1d(par.grid_m, sol.c[0], m)
        else:
            # Use policy from period (t - shift)
            c = linear_interp.interp_1d(par.grid_m, sol.c[t - shift], m)

        # Return consumption as share of m
        if m < 0.01:
            return np.array([1.0])
        else:
            return np.array([c / m])
