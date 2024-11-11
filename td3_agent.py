import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent
from networks import Actor, Critic
from replay_buffer import ReplayBuffer


class TD3Agent(Agent):
    """
    TD3 Agent implementing the Twin Delayed DDPG algorithm.
    """
    def __init__(self, state_dim, action_dim, **kwargs):
        """
        Initialize the TD3Agent.
        """
        super().__init__()

        # Set to environment values
        self.beta = 0.96
        self.action_low = 0.0
        self.action_high = 1.0

        # Network settings
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128

        # Other hyperparameters
        self.tau = 0.005
        self.batch_size = 512
        self.buffer_size = 1_000_000
        self.ini_explore_noise = 0.2
        self.explore_noise = 0.2  # Will be depreciated

        # Policy updates
        self.total_it = 0
        self.policy_noise = 0.2
        self.policy_delay = 2 # Can interfere with time state?
        self.noise_clip = 0.5

        # Update attributes with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Networks and optimizers
        self._initialize_networks(state_dim, action_dim)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)

    # ------------------
    # - Public methods -
    # ------------------

    def learn(self):
        """
        Perform a learning step by updating the networks.
        """
        if self.replay_buffer.size() < self.batch_size:
            return None, None

        self.total_it += 1

        # Sample batch
        batch_np = self.replay_buffer.sample(self.batch_size)

        # Convert NumPy arrays to tensors
        batch = tuple(
            torch.FloatTensor(item).unsqueeze(1) if item.ndim == 1
            else torch.FloatTensor(item)
            for item in batch_np
        )

        # Compute critic loss
        critic_loss1, critic_loss2 = self._compute_critic_loss(batch)

        # Update critics
        self._update_network(self.critic1_optimizer, critic_loss1)
        self._update_network(self.critic2_optimizer, critic_loss2)

        if self.total_it % self.policy_delay == 0:
            # Update actor
            actor_loss = self._compute_actor_loss(batch)
            self._update_network(self.actor_optimizer, actor_loss)

            # Soft update targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
        else:
            actor_loss = None

        return (
            critic_loss1.item(),
            actor_loss.item() if actor_loss is not None else None
        )

    def select_action(self, state, noise=True, shift=0):
        """
        Select an action given the current state.
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]

        if noise:
            action += self.explore_noise * np.random.randn(*action.shape)

        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)

        return action
    
    # -------------------
    # - Private methods -
    # -------------------

    def _initialize_networks(self, state_dim, action_dim):
        """
        Initialize networks and their optimizers.
        """
        networks = {
            'actor': [Actor, self.actor_lr, self.actor_hidden_dim],
            'critic1': [Critic, self.critic_lr, self.critic_hidden_dim],
            'critic2': [Critic, self.critic_lr, self.critic_hidden_dim]
        }

        for net_name, (net_class, lr, hidden_dim) in networks.items():
            # Initialize network
            net = net_class(state_dim, action_dim, hidden_dim)
            optimizer = optim.Adam(net.parameters(), lr=lr)
            net_target = copy.deepcopy(net)

            # Set attributes
            setattr(self, net_name, net)
            setattr(self, f"{net_name}_optimizer", optimizer)
            setattr(self, f"{net_name}_target", net_target)

    def _compute_critic_loss(self, batch):
        """
        Compute and return the losses for both critic networks.
        """
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            # Target actions with noise
            noise = (torch.randn_like(actions) * self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(self.action_low, 
                self.action_high)

            # Compute target Q values using target critics
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = (rewards + (1 - dones.float()) 
                * self.beta * target_q)

        # Compute current Q estimates using both critics
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Compute critic losses
        critic_loss1 = nn.MSELoss()(current_q1, target_q)
        critic_loss2 = nn.MSELoss()(current_q2, target_q)

        return critic_loss1, critic_loss2

    def _compute_actor_loss(self, batch):
        """
        Compute and return the actor loss.
        """
        states, _, _, _, _ = batch
        actor_loss = -self.critic1(states, self.actor(states)).mean()

        return actor_loss

    def _update_network(self, optimizer, loss):
        """
        Update network parameters with the given optimizer and loss.
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _soft_update(self, local_model, target_model):
        """
        Soft-update model parameters.
        """
        local_params = local_model.parameters()
        target_params = target_model.parameters()

        for param, target_param in zip(local_params, target_params):
            target_param.data.copy_(self.tau * param.data 
                + (1 - self.tau) * target_param.data)
