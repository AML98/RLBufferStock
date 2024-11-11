import numpy as np

from base_env import BaseEnv


class BufferStockEnv(BaseEnv):
    """
    An environment for the buffer stock model
    """
    def __init__(self, state_vars=None, action_vars=None, additional_vars=None, 
        **kwargs):
        
        # -----------------------
        # - BaseEnv constructor -
        # -----------------------

        # State vars: permanent income, market resources, and time
        default_state_vars = [
            {'name': 'p', 'ini': 1.0, 'low': 0.0, 'high': 100},
            {'name': 'm', 'ini': 1.0, 'low': 0.0, 'high': 100},
            {'name': 't', 'ini': 0.0, 'low': 0.0, 'high': 1.0}
        ]
        
        # Action var: consumption as share of market resources
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

        # Terminal period
        self.T = 10

        # Preferences
        self.beta = 0.96
        self.rho = 1.0

        # Income process
        self.sigma_psi = 0.1
        self.sigma_xi = 0.1
        self.pi = 0.1
        self.mu = 0.5
        self.R = 1.03

        # Update attributes with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    # ------------------
    # - Public methods -
    # ------------------

    def transition(self):
        """
        Transition function
        """
        p = self.p
        t = self.t
        a = self.a

        # Income and market resources 
        p_plus = self._compute_p_plus(p)
        m_plus = self._compute_m_plus(p_plus, a)

        self.p = p_plus
        self.m = m_plus
        self.t = t + 1 / self.T

    def compute_reward(self, c_share):
        """
        Compute utility and reward
        """
        m = self.m

        # Consumption and assets
        c = c_share[0] * m
        a = m - c

        # Utility and reward
        utility = self._compute_utility(c)
        reward = np.clip(utility + 3, -5, 5)

        self.utility = utility
        self.reward = reward

        self.c_share = c_share
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
    
    def _compute_p_plus(self, p):
        """
        Compute next period p given p and a
        """
        psi_plus = np.random.lognormal(mean=-0.5 * self.sigma_psi ** 2, 
            sigma=self.sigma_psi)

        p_plus = psi_plus * p
        
        return p_plus
    
    def _compute_m_plus(self, p_plus, a):
        """
        Compute next period m given p_plus and a
        """
        xi_t1 = np.random.lognormal(mean=-0.5 * self.sigma_xi ** 2, 
            sigma=self.sigma_xi)
                
        if np.random.rand() < self.pi:
            tilde_xi_t1 = self.mu
        else:
            tilde_xi_t1 = (xi_t1 - self.pi * self.mu) / (1 - self.pi)

        m_plus = self.R * a + tilde_xi_t1 * p_plus

        return m_plus

    def _compute_utility(self, c):
        """
        CRRA utility function
        """
        # Add small number to avoid log(0)
        if self.rho == 1.0:
            utility = np.log(c + 1e-8)
        else:
            utility = (c + 1e-8) ** (1 - self.rho) / (1 - self.rho)
        
        return utility