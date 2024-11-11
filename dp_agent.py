import numpy as np

from consav import linear_interp
from agent import Agent


class DPAgent(Agent):
    def __init__(self, model, env):
        super().__init__()
        self.model = model
        self.env = env

    def select_action(self, state, noise=False, infinite_horizon=None, shift=0):
        par = self.model.par
        sol = self.model.sol

        p = state[0]
        m = state[1]
        t = self.env.period

        if infinite_horizon is True:
            # Use policy from initial period
            c = linear_interp.interp_2d(par.grid_p, par.grid_m, sol.c[0], p, m)
        else:
            # Use policy from period t+shift
            c = linear_interp.interp_2d(par.grid_p, par.grid_m, sol.c[t-shift], p, m)
        
        # Return consumption as share (action var in env)
        return np.array([c / m])
