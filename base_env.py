from abc import ABC, abstractmethod
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class BaseEnv(ABC, gym.Env):
    def __init__(self, state_vars, action_vars, additional_vars=None):
        '''
        '''
        self.state_vars = state_vars
        self.action_vars = action_vars
        self.additional_vars = additional_vars

        # Initialize variables
        self.vars = state_vars + action_vars + (additional_vars or [])
        self._initialize_variables(self.vars)

        # State and action spaces
        s_lows, s_highs = self._get_bounds(self.state_vars)
        a_lows, a_highs = self._get_bounds(self.action_vars)

        self.action_space = spaces.Box(low=a_lows, high=a_highs,
            dtype=np.float64)
        self.observation_space = spaces.Box(low=s_lows, high=s_highs,
            dtype=np.float64)

        # Termination flag and reward
        self.done = None
        self.reward = None
        self.reset_period = 0
        self.period = 0

    # ------------------
    # - Public methods -
    # ------------------
    
    @abstractmethod
    def compute_reward(self, *action):
        '''
        '''
        pass

    @abstractmethod    
    def transition(self):
        '''
        '''
        pass

    @abstractmethod
    def terminate(self):
        '''
        '''
        pass

    def step(self, *action):
        '''
        ''' 
        self.period = self.period + 1
        self.transition()
        self.terminate()
        
        # Return state as numpy array
        state = np.array([getattr(self, state_var['name']) 
            for state_var in self.state_vars])

        return state, self.reward, self.done, {}
    
    def reset(self, ini=None):
        '''
        '''
        self.done = False
        self.reward = 0.0
        self.period = self.reset_period
        
        # --- 1. State variables ---

        # Custom initial state
        if ini is not None:
            for state_var in self.state_vars:
                setattr(self, state_var['name'], ini[state_var['name']])
        
        # Default initial state
        else:
            for state_var in self.state_vars:
                ini = getattr(self, f"ini_{state_var['name']}")
                
                # If random initial state
                if isinstance(ini, list) == True:
                    ini = self.np_random.uniform(low=ini[0], 
                        high=ini[1])
                
                setattr(self, state_var['name'], ini)

        # --- 2. Additional variables ---

        for add_var in self.additional_vars:

            # If initial value is specified
            if hasattr(self, f"ini_{add_var['name']}") == True:
                ini = getattr(self, f"ini_{add_var['name']}")
                if isinstance(ini, list) == True:
                    ini = self.np_random.uniform(
                        low=ini[0], 
                        high=ini[1]
                    )
                
                setattr(self, add_var['name'], ini)

            # If no initial value is specified
            else:
                setattr(self, add_var['name'], None)

        # --- 3. Action variables ---
        
        for action_var in self.action_vars:
            setattr(self, action_var['name'], None)

        # Return state as numpy array
        state = np.array([getattr(self, state_var['name']) 
            for state_var in self.state_vars])
        
        return state
    
    def seed(self, seed=None):
        '''
        '''
        self.np_random, seed = seeding.np_random(seed)
        if hasattr(self, 'action_space'):
            self.action_space.seed(seed)

        return [seed]

    # -------------------
    # - Private methods -
    # -------------------
    
    def _initialize_variables(self, var_list):
        '''
        '''
        for var in var_list:
            if 'ini' in var:
                # Initial state value
                setattr(self, f"ini_{var['name']}", var['ini'])

            # Current value    
            setattr(self, var['name'], None)
    
    def _get_bounds(self, var_list):
        '''
        '''
        # Space bounds
        lows = np.array([var['low'] for var in var_list])
        highs = np.array([var['high'] for var in var_list])
        
        return lows, highs