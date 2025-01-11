import numpy as np
import torch
import torch.nn as nn

class LinearActor(nn.Module):
    """
    Linear Actor: Represents the policy as a linear function.
    Action = W * state + b, where W and b are learnable parameters.
    """
    def __init__(self, state_dim, action_dim):
        super(LinearActor, self).__init__()
        # Linear transformation: state_dim -> action_dim
        self.linear = nn.Linear(state_dim, action_dim, bias=True)

    def forward(self, state):
        """
        Forward pass to compute the action.
        Args:
            state: shape (batch_size, state_dim)
        Returns:
            action: shape (batch_size, action_dim)
        """
        return torch.sigmoid(self.linear(state))  # Clamp action to [0, 1]


class LinearCritic(nn.Module):
    """
    Linear Critic: Approximates the Q-value function.
    Q(s, a) = W * [state, action] + b
    """
    def __init__(self, state_dim, action_dim):
        super(LinearCritic, self).__init__()
        # Linear transformation: state_dim + action_dim -> 1
        self.linear = nn.Linear(state_dim + action_dim, 1, bias=True)

    def forward(self, state, action):
        """
        Forward pass to compute Q-value.
        Args:
            state: shape (batch_size, state_dim)
            action: shape (batch_size, action_dim)
        Returns:
            Q-value: shape (batch_size, 1)
        """
        # Concatenate state and action along the feature dimension
        x = torch.cat([state, action], dim=1)
        return self.linear(x)
