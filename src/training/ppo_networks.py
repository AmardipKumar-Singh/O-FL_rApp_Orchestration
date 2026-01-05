#!/usr/bin/env python3
"""
training/ppo_networks.py - PPO Actor-Critic Networks
O-FL rApp: Neural networks for PPO/MAPPO
"""

import numpy as np
from typing import Tuple, Optional


class ActorNetwork:
    """
    Actor network for PPO (policy network)

    Outputs mean and log_std for Gaussian policy (continuous actions)
    or logits for categorical policy (discrete actions)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 action_type: str = 'continuous'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_type = action_type

        # Initialize weights
        self.W1 = np.random.randn(obs_dim, hidden_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))

        if action_type == 'continuous':
            # Mean and log_std for Gaussian policy
            self.W_mean = np.random.randn(hidden_dim, action_dim) * 0.01
            self.b_mean = np.zeros((1, action_dim))
            self.log_std = np.zeros((1, action_dim))  # Learnable log std
        else:
            # Logits for categorical policy
            self.W_logits = np.random.randn(hidden_dim, action_dim) * 0.01
            self.b_logits = np.zeros((1, action_dim))

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass

        Returns:
            For continuous: (mean, log_std)
            For discrete: (logits, None)
        """
        # Hidden layers with ReLU
        h1 = np.maximum(0, obs @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)

        if self.action_type == 'continuous':
            mean = h2 @ self.W_mean + self.b_mean
            return mean, self.log_std
        else:
            logits = h2 @ self.W_logits + self.b_logits
            return logits, None

    def sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Sample action from policy

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        if self.action_type == 'continuous':
            mean, log_std = self.forward(obs)
            std = np.exp(log_std)
            action = mean + std * np.random.randn(*mean.shape)

            # Compute log probability
            log_prob = -0.5 * np.sum(
                ((action - mean) / (std + 1e-8)) ** 2 + 
                2 * log_std + 
                np.log(2 * np.pi),
                axis=-1
            )

            return action, log_prob
        else:
            logits, _ = self.forward(obs)
            probs = self._softmax(logits)
            action = np.array([np.random.choice(self.action_dim, p=p) 
                              for p in probs])

            # Compute log probability
            log_prob = np.log(probs[np.arange(len(action)), action] + 1e-8)

            return action, log_prob

    def get_log_prob(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute log probability of given action"""
        if self.action_type == 'continuous':
            mean, log_std = self.forward(obs)
            std = np.exp(log_std)

            log_prob = -0.5 * np.sum(
                ((action - mean) / (std + 1e-8)) ** 2 + 
                2 * log_std + 
                np.log(2 * np.pi),
                axis=-1
            )
            return log_prob
        else:
            logits, _ = self.forward(obs)
            probs = self._softmax(logits)
            action_int = action.astype(int).flatten()
            log_prob = np.log(probs[np.arange(len(action_int)), action_int] + 1e-8)
            return log_prob

    def get_parameters(self) -> dict:
        """Get all parameters"""
        params = {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }

        if self.action_type == 'continuous':
            params['W_mean'] = self.W_mean.copy()
            params['b_mean'] = self.b_mean.copy()
            params['log_std'] = self.log_std.copy()
        else:
            params['W_logits'] = self.W_logits.copy()
            params['b_logits'] = self.b_logits.copy()

        return params

    def set_parameters(self, params: dict) -> None:
        """Set all parameters"""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()

        if self.action_type == 'continuous':
            self.W_mean = params['W_mean'].copy()
            self.b_mean = params['b_mean'].copy()
            self.log_std = params['log_std'].copy()
        else:
            self.W_logits = params['W_logits'].copy()
            self.b_logits = params['b_logits'].copy()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class CriticNetwork:
    """
    Critic network for PPO (value function)

    Estimates state value V(s)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.W1 = np.random.randn(obs_dim, hidden_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))

        self.W_value = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_value = np.zeros((1, 1))

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Returns:
            value: State value V(s)
        """
        # Hidden layers with ReLU
        h1 = np.maximum(0, obs @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)

        # Value output
        value = h2 @ self.W_value + self.b_value

        return value

    def get_parameters(self) -> dict:
        """Get all parameters"""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy(),
            'W_value': self.W_value.copy(),
            'b_value': self.b_value.copy()
        }

    def set_parameters(self, params: dict) -> None:
        """Set all parameters"""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
        self.W_value = params['W_value'].copy()
        self.b_value = params['b_value'].copy()


if __name__ == "__main__":
    print("PPO Actor-Critic Networks")

    # Test continuous action
    actor = ActorNetwork(obs_dim=10, action_dim=4, action_type='continuous')
    obs = np.random.randn(5, 10)  # Batch of 5 observations
    action, log_prob = actor.sample_action(obs)
    print(f"Continuous action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")

    # Test discrete action
    actor_discrete = ActorNetwork(obs_dim=10, action_dim=4, action_type='discrete')
    action_discrete, log_prob_discrete = actor_discrete.sample_action(obs)
    print(f"Discrete action shape: {action_discrete.shape}")

    # Test critic
    critic = CriticNetwork(obs_dim=10)
    value = critic.forward(obs)
    print(f"Value shape: {value.shape}")
