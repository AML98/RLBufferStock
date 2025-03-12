import numpy as np
from agent import Agent

class HTMAgent(Agent):
    def __init__(self, env):
        super().__init__()
        self.name = 'HTM household'
        self.env = env

    def select_action(self, state, noise=False, shift=None):
        """
        """

        # Consume everything
        return np.array([1.0])
