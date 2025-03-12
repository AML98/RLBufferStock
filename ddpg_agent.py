import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent
from networks import Actor, Critic
from replay_buffer import ReplayBuffer


class DDPGAgent(Agent):
    """
    DDPG Agent (Deep Deterministic Policy Gradient).
    """
    def __init__(self, state_dim, action_dim, **kwargs):
        """
        Initialize the DDPGAgent.
        """
        super().__init__()
        self.name = 'DDPG household'

        # Set to environment values
        self.beta = 0.96
        self.action_low = 0.0
        self.action_high = 1.0

        # Network settings
        self.actor_lr = 0.5*1e-4
        self.critic_lr = 1.0*1e-4
        self.actor_hidden_dim = 64
        self.critic_hidden_dim = 64 

        # Other hyperparameters
        self.tau = 0.005
        self.batch_size = 256
        self.buffer_size = 1_000_000
        self.ini_explore_noise = 0.2
        self.explore_noise = None  # Will be depreciated - set later
        self.noise_decay = 0.01

        # Update attributes with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Networks and optimizers
        self._initialize_networks(state_dim, action_dim)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)

        # Exploration noise
        self.reset_explore_noise()

    # ------------------
    # - Public methods -
    # ------------------

    def learn(self):
        """
        Perform a learning step by updating the networks.
        """
        if self.replay_buffer.size() < self.batch_size:
            return None, None

        # Sample batch
        batch_np = self.replay_buffer.sample(self.batch_size)

        # Convert NumPy arrays to tensors
        batch = tuple(
            torch.FloatTensor(item).unsqueeze(1) if item.ndim == 1
            else torch.FloatTensor(item)
            for item in batch_np
        )

        # Compute critic loss
        critic_loss = self._compute_critic_loss(batch)

        # Update critic
        self._update_network(self.critic, self.critic_optimizer, critic_loss)

        # Update actor
        actor_loss = self._compute_actor_loss(batch)
        self._update_network(self.actor, self.actor_optimizer, actor_loss)

        # Soft update targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return (
            critic_loss.item(),
            actor_loss.item()
        )

    def select_action(self, state, noise=True, shift=0):
        """
        Select an action given the current state.
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]

        if noise:
            action += np.random.normal(0, self.explore_noise, size=action.shape)

        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)

        return action

    def reset_explore_noise(self):
        '''
        '''
        self.explore_noise = self.ini_explore_noise

    def decay_explore_noise(self, episode):
        '''
        '''
        self.explore_noise = (self.ini_explore_noise 
                              * np.exp(-self.noise_decay * episode))

    # -------------------
    # - Private methods -
    # -------------------

    def _initialize_networks(self, state_dim, action_dim):
        """
        Initialize actor, critic, and their target networks (and optimizers).
        """
        # Actor
        self.actor = Actor(state_dim, action_dim, self.actor_hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic
        self.critic = Critic(state_dim, action_dim, self.critic_hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

    def _compute_critic_loss(self, batch):
        """
        Compute and return the loss for the critic network.
        """
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            # Target action
            next_actions = self.actor_target(next_states)
            next_actions = next_actions.clamp(self.action_low, self.action_high)

            # Compute target Q value
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones.float()) * self.beta * target_q

        # Compute current Q estimate
        current_q = self.critic(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        return critic_loss

    def _compute_actor_loss(self, batch):
        """
        Compute and return the actor loss.
        """
        states, _, _, _, _ = batch
        actor_loss = -self.critic(states, self.actor(states)).mean()
        return actor_loss

    def _update_network(self, network, optimizer, loss):
        """
        Update network parameters with the given optimizer and loss.
        """
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()

    def _soft_update(self, local_model, target_model):
        """
        Soft-update model parameters.
        """
        for param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )