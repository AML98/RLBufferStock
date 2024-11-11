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
    
    def reset(self):
        '''
        '''
        self.done = False
        self.reward = 0.0
        self.period = self.reset_period
        
        for state_var in self.state_vars:
            ini = getattr(self, f"ini_{state_var['name']}")
            setattr(self, state_var['name'], ini)
        
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