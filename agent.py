from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
import wandb
import copy

import warnings
# warnings.simplefilter("error", RuntimeWarning)

from history import History

class Agent(ABC):
    '''
    Abstract class for agents.
    '''
    def __init__(self):
        self.print_freq = 100
        self.updates_per_step = 1
        self.warmup_episodes = 5
        
        self.history = None  # Initialize later

    @abstractmethod
    def select_action(self, state, noise=False):
        pass

    def learn(self):
        pass

    def interact(self, env, n_episodes, train, keep_history=False,
        shift=0, track=False, do_print=True, inis=None):
        '''
        Simulate interaction with an environment using the agent
        '''

        if keep_history is False:
            # Custom vars must be in order they are calculated
            custom_vars = [
            
            # Life-time utility
            {'name': 'value', 
            'shape': (n_episodes,), 
            'function': History.compute_value},

            # Euler error
            {'name': 'euler_error',
             'shape': (n_episodes, env.T),
             'function': History.compute_euler_error},

            ]

            name = f'{self.__class__.__name__}'
            self.history = History(name, env, n_episodes,
                custom_vars=custom_vars)

        if train:
            self.reset_explore_noise()

            self.history.a_loss = np.zeros(n_episodes)
            self.history.c_loss = np.zeros(n_episodes)

            for episode in range(n_episodes):
                if inis is not None:
                    ini = inis[episode]
                else:
                    ini = None

                self._run_episode(env, episode, train, ini=ini, shift=shift)

                if hasattr(self, 'explore_noise'):
                    self.decay_explore_noise(episode)

                if do_print:
                    self._print_progress(episode)

                if episode == n_episodes - 1 and hasattr(self.history, 'euler_error'):
                    self.history.compute_trans_euler_error(env, n_episodes) 

        else:
            for episode in range(n_episodes):
                if inis is not None:
                    ini = inis[episode]
                else:
                    ini = None

                self._run_episode(env, episode, train, ini=ini, shift=shift)

                if episode == n_episodes - 1:
                    self.history.compute_trans_euler_error(env, n_episodes) 

    # -------------------
    # - Private methods -
    # -------------------

    def _run_episode(self, env, episode, train, ini=None, shift=0):
        '''
        Run episode
        '''
        state = env.reset(ini)
        done = False

        c_losses = []
        a_losses = []
        
        while not done:
            if train and episode < self.warmup_episodes:
                action = env.action_space.sample()
            else:
                action = self.select_action(state, noise=train, shift=shift)
            
            # Save history before taking step
            env.compute_reward(action)
            if self.history is not None:
                self.history.record_step(env, episode, self, shift=shift)
            next_state, reward, done, _ = env.step(action)
            
            if train:
                self.replay_buffer.add(state, action, reward,
                    next_state, done)
                
                for _ in range(self.updates_per_step):
                    c_loss, a_loss = self.learn()

                    if a_loss is not None and c_loss is not None:
                        c_losses.append(c_loss)
                        a_losses.append(a_loss)

                # if a_loss is not None:
                #    wandb.log({
                #        "critic_loss": c_loss,
                #        "actor_loss": a_loss
                #    })

            state = next_state
            #if self.name == 'DP household':
            #    print(state)

        if train == True:

            # Mean will be empty if replay buffer < batch size
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

                self.history.a_loss[episode] = np.mean(a_losses)
                self.history.c_loss[episode] = np.mean(c_losses)

    def _print_progress(self, episode):
        if (episode + 1) % self.print_freq == 0:
            print(f"Episode {episode + 1}")