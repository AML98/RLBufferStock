import numpy as np
from base_env import BaseEnv
from consav.quadrature import create_PT_shocks

class BufferStockEnv(BaseEnv):
    """
    An environment for the buffer stock model normalized by permanent income
    """

    def __init__(self, state_vars=None, action_vars=None, additional_vars=None, 
                 **kwargs):
        
        # -----------------------
        # - BaseEnv constructor -
        # -----------------------

        # State vars: market resources and time
        default_state_vars = [
            {'name': 'm', 'ini': [0,2], 'low': 0.0, 'high': 100},
            {'name': 't', 'ini': 0.0, 'low': 0.0, 'high': 1.0}
        ]
        
        # Action var: consumption as share of resources
        default_action_vars = [
            {'name': 'c_share', 'low': 0.0, 'high': 1.0}
        ]
        
        # Additional vars: utility, consumption, and assets
        default_additional_vars = [
            {'name': 'utility'},
            {'name': 'c'},
            {'name': 'a'}
        ]

        state_vars = state_vars or default_state_vars
        action_vars = action_vars or default_action_vars
        additional_vars = additional_vars or default_additional_vars

        super().__init__(
            state_vars=state_vars, 
            action_vars=action_vars,
            additional_vars=additional_vars
        )

        # --------------------
        # - Model parameters -
        # --------------------
    
        self.T = 10                
        self.beta = 0.96           
        self.rho = 1.0             
        self.R = 1.03              

        # Shock parameters
        self.Npsi = 6               
        self.Nxi = 6
        self.sigma_psi = 0.1
        self.sigma_xi = 0.1
        self.pi = 0.1
        self.mu = 0.5

        # Update attributes with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Discretized shock distribution (meshgrid)
        self.psi, self.psi_w, self.xi, self.xi_w, self.Nshocks = create_PT_shocks(
            self.sigma_psi, self.Npsi,
            self.sigma_xi, self.Nxi,
            self.pi, self.mu
        )
        self.shock_probs = self.psi_w * self.xi_w

    # ------------------
    # - Public methods -
    # ------------------

    def transition(self):
        """
        Transition function: updates (P, m, t) to next period's values
        """
        a = self.a
        t = self.t

        # Draw shock from the discrete distribution
        idx = np.random.choice(self.Nshocks, p=self.shock_probs)
        psi_plus = self.psi[idx]
        xi_plus = self.xi[idx]

        # Market resources
        m_plus = self.R * a / psi_plus + xi_plus

        self.m = m_plus
        self.t = t + 1 / self.T

    def compute_reward(self, c_share):
        """
        Compute utility and reward from consumption choice
        """
        m = self.m

        # Consumption and assets
        c = c_share[0] * m  
        a = m - c

        # Utility and reward
        utility = self._compute_utility(c)
        reward = self._compute_clipped_utility(c)

        # Log variables to the environment
        self.c_share = c_share
        self.utility = utility
        self.reward = reward
        self.c = c
        self.a = a

    def terminate(self):
        """
        Termination function
        """
        self.done = (self.period == self.T)
    
    # -------------------
    # - Private methods -
    # -------------------
    
    def _compute_utility(self, c):
        """
        CRRA utility function
        """
        if self.rho == 1.0:
            return np.log(c + 1e-8)
        else:
            return (c + 1e-8) ** (1 - self.rho) / (1 - self.rho)
        
    def _compute_clipped_utility(self, c):
        """
        Clip utility to be between -5 and 5
        """
        return np.clip(self._compute_utility(c), -5, 5)
    
    def _compute_rescaled_utility(self, c):
        """
        Rescale CRRA utility to be between -1 and 1
        """
        c_min = 0
        c_max = 2
        u_min = self._compute_utility(c_min)
        u_max = self._compute_utility(c_max)
        
        A = 2 / (u_max - u_min)
        B = -1 - A * u_min

        return np.clip(A * self._compute_utility(c) + B, -1, 1)
