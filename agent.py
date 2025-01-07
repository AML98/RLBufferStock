from abc import ABC, abstractmethod
import numpy as np
import wandb
import copy

from history import History


class Agent(ABC):
    '''
    Abstract class for agents.
    '''
    def __init__(self):
        self.history = None
        self.print_freq = 100
        self.updates_per_step = 1
        self.warmup_episodes = 5

    @abstractmethod
    def select_action(self, state, noise=False):
        pass

    def learn(self):
        pass

    def interact(self, env, n_episodes, train, keep_history=False,
        shift=0, track=None, do_print=True):
        '''
        Simulate interaction with an environment using the agent
        '''
        if keep_history is False:
            # Initialize history, otherwise overwrite
            custom_vars = [
            
            # Life-time utility
            {'name': 'value', 
            'shape': (n_episodes,), 
            'function': History._compute_value},
            
            # Savings rate
            {'name': 's',
            'shape': (n_episodes, env.T),
            'function': History._compute_savings_rate}

            ]

            name = f'{self.__class__.__name__}'
            self.history = History(name, env, n_episodes,
                custom_vars=custom_vars)

        # Run episodes
        if train:
            #wandb.init(project='Speciale')

            self.set_explore_noise(self.ini_explore_noise)

            if track is not None:
                agents = []

            for episode in range(n_episodes):
                self._run_episode(env, episode, train, shift=shift)

                if track and episode in track:
                    agents.append(copy.deepcopy(self))

                if do_print:
                    self._print_progress(episode)

            #wandb.finish()

        else:
            for episode in range(n_episodes):
                self._run_episode(env, episode, train, shift=shift)

        if track:
            return agents

    # -------------------
    # - Private methods -
    # -------------------

    def _run_episode(self, env, episode, train, shift=0):
        '''
        Run episode
        '''
        state = env.reset()
        done = False
        
        while not done:
            if train and episode < self.warmup_episodes:
                    action = env.action_space.sample()
            else:
                action = self.select_action(state, noise=train, 
                    shift=shift)
            
            # Save history before taking step
            env.compute_reward(action)
            self.history.record_step(env, episode)
            next_state, reward, done, _ = env.step(action)
            
            if train:
                self.replay_buffer.add(state, action, reward,
                    next_state, done)
                
                for _ in range(self.updates_per_step):
                    c_loss, a_loss = self.learn()

                if hasattr(self, 'explore_noise'):
                    self.explore_noise *= self.noise_decay

                #if a_loss is not None:
                #    wandb.log({
                #        "critic_loss": c_loss,
                #        "actor_loss": a_loss
                #    })

            state = next_state  

    def _print_progress(self, episode):
        if (episode + 1) % self.print_freq == 0:
            print(f"Episode {episode + 1}")