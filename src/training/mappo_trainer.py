#!/usr/bin/env python3
"""
training/mappo_trainer.py - Multi-Agent PPO (MAPPO) Trainer
O-FL rApp: MARL implementation with PPO

"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from training.marl_base import (MARLTrainer, MARLState, MARLAction, MARLReward,
                                MARLTransition, ReplayBuffer)
from training.ppo_networks import ActorNetwork, CriticNetwork


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO"""
    gamma: float = 0.99  # Discount factor
    lambda_gae: float = 0.95  # GAE lambda
    epsilon_clip: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    ppo_epochs: int = 4  # PPO update epochs per batch
    mini_batch_size: int = 64  # Mini-batch size for PPO updates
    lr_actor: float = 3e-4  # Actor learning rate
    lr_critic: float = 1e-3  # Critic learning rate
    hidden_dim: int = 64  # Hidden layer dimension
    action_type: str = 'continuous'  # 'continuous' or 'discrete'
    use_shared_critic: bool = True  # Share critic across agents


class MAPPOAgent:
    """Single agent in MAPPO system"""

    def __init__(self, agent_id: str, obs_dim: int, action_dim: int,
                 config: MAPPOConfig):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # Actor network (policy)
        self.actor = ActorNetwork(
            obs_dim, action_dim, config.hidden_dim, config.action_type
        )

        # Critic network (value function) - may be shared
        self.critic: Optional[CriticNetwork] = None
        if not config.use_shared_critic:
            self.critic = CriticNetwork(obs_dim, config.hidden_dim)


class MAPPOTrainer(MARLTrainer):
   

    def __init__(self, agent_ids: List[str], obs_dims: Dict[str, int],
                 action_dims: Dict[str, int], config: Optional[MAPPOConfig] = None):
        self.agent_ids = agent_ids
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.config = config or MAPPOConfig()

        # Create agents
        self.agents: Dict[str, MAPPOAgent] = {}
        for agent_id in agent_ids:
            self.agents[agent_id] = MAPPOAgent(
                agent_id, obs_dims[agent_id], action_dims[agent_id], self.config
            )

        # Shared critic (if enabled)
        self.shared_critic: Optional[CriticNetwork] = None
        if self.config.use_shared_critic:
            # Use concatenated observations for centralized critic
            total_obs_dim = sum(obs_dims.values())
            self.shared_critic = CriticNetwork(total_obs_dim, self.config.hidden_dim)

        # Replay buffer (for on-policy data)
        self.buffer = ReplayBuffer(capacity=10000)

        # Training statistics
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_reward': [],
            'episode_length': []
        }

        self.episode_count = 0
        self.step_count = 0

    def select_actions(self, state: MARLState, 
                      explore: bool = True) -> MARLAction:
        """
        Select actions for all agents

        Args:
            state: Current environment state
            explore: If True, sample from policy; if False, use mean

        Returns:
            Actions for all agents
        """
        actions = {}

        for agent_id in self.agent_ids:
            obs = state.get_agent_obs(agent_id).reshape(1, -1)
            agent = self.agents[agent_id]

            if explore:
                action, _ = agent.actor.sample_action(obs)
            else:
                # Use mean action (for evaluation)
                if self.config.action_type == 'continuous':
                    action, _ = agent.actor.forward(obs)
                else:
                    logits, _ = agent.actor.forward(obs)
                    action = np.argmax(logits, axis=-1)

            actions[agent_id] = action.flatten()

        return MARLAction(actions)

    def train_step(self, batch: List[MARLTransition]) -> Dict[str, float]:
        """
        Perform MAPPO training step

        Args:
            batch: Batch of on-policy transitions

        Returns:
            Training metrics
        """
        if len(batch) < self.config.mini_batch_size:
            return {}

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(batch)

        # PPO update for multiple epochs
        metrics = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}

        for _ in range(self.config.ppo_epochs):
            # Shuffle batch
            indices = np.random.permutation(len(batch))

            # Mini-batch updates
            for start in range(0, len(batch), self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, len(batch))
                mini_batch_idx = indices[start:end]

                mini_batch = [batch[i] for i in mini_batch_idx]
                mini_adv = {agent_id: adv[mini_batch_idx] 
                           for agent_id, adv in advantages.items()}
                mini_ret = {agent_id: ret[mini_batch_idx]
                           for agent_id, ret in returns.items()}

                # Update each agent
                for agent_id in self.agent_ids:
                    agent_metrics = self._update_agent(
                        agent_id, mini_batch, mini_adv[agent_id], mini_ret[agent_id]
                    )

                    for key, value in agent_metrics.items():
                        metrics[key] += value

        # Average metrics
        n_updates = self.config.ppo_epochs * (len(batch) // self.config.mini_batch_size)
        for key in metrics:
            metrics[key] /= max(1, n_updates * len(self.agent_ids))

        # Store statistics
        for key, value in metrics.items():
            self.training_stats[key].append(value)

        return metrics

    def _update_agent(self, agent_id: str, batch: List[MARLTransition],
                     advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Update single agent's actor and critic"""
        agent = self.agents[agent_id]

        # Extract observations and actions
        obs = np.array([t.state.get_agent_obs(agent_id) for t in batch])
        actions = np.array([t.action.get_agent_action(agent_id) for t in batch])
        old_log_probs = np.array([
            agent.actor.get_log_prob(
                t.state.get_agent_obs(agent_id).reshape(1, -1),
                t.action.get_agent_action(agent_id).reshape(1, -1)
            )[0]
            for t in batch
        ])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute new log probs
        new_log_probs = agent.actor.get_log_prob(obs, actions)

        # PPO clipped objective
        ratio = np.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.config.epsilon_clip,
                       1 + self.config.epsilon_clip) * advantages
        actor_loss = -np.mean(np.minimum(surr1, surr2))

        # Entropy bonus
        if self.config.action_type == 'continuous':
            mean, log_std = agent.actor.forward(obs)
            entropy = np.mean(0.5 * (np.log(2 * np.pi * np.exp(2 * log_std)) + 1))
        else:
            logits, _ = agent.actor.forward(obs)
            probs = agent.actor._softmax(logits)
            entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=-1))

        # Value loss
        critic = self.shared_critic if self.config.use_shared_critic else agent.critic
        if self.config.use_shared_critic:
            # Use all observations for centralized critic
            critic_obs = np.array([t.state.get_all_observations() for t in batch])
        else:
            critic_obs = obs

        values = critic.forward(critic_obs).flatten()
        value_loss = np.mean((returns - values) ** 2)

        # Total loss
        total_loss = (actor_loss + 
                     self.config.value_coef * value_loss - 
                     self.config.entropy_coef * entropy)

        # Gradient descent (simplified - using finite differences)
        self._update_parameters(agent, critic, obs, actions, advantages, returns,
                               critic_obs)

        return {
            'actor_loss': actor_loss,
            'critic_loss': value_loss,
            'entropy': entropy
        }

    def _update_parameters(self, agent: MAPPOAgent, critic: CriticNetwork,
                          obs: np.ndarray, actions: np.ndarray,
                          advantages: np.ndarray, returns: np.ndarray,
                          critic_obs: np.ndarray) -> None:
        """Update network parameters using gradient descent"""

        # Actor update (simplified)
        actor_params = agent.actor.get_parameters()
        for key in actor_params:
            grad = self._compute_gradient(
                lambda p: self._actor_loss(agent, obs, actions, advantages, key, p),
                actor_params[key]
            )
            actor_params[key] -= self.config.lr_actor * grad
        agent.actor.set_parameters(actor_params)

        # Critic update (simplified)
        critic_params = critic.get_parameters()
        for key in critic_params:
            grad = self._compute_gradient(
                lambda p: self._critic_loss(critic, critic_obs, returns, key, p),
                critic_params[key]
            )
            critic_params[key] -= self.config.lr_critic * grad
        critic.set_parameters(critic_params)

    def _compute_gradient(self, loss_fn, param: np.ndarray, 
                         eps: float = 1e-5) -> np.ndarray:
        """Compute gradient using finite differences"""
        grad = np.zeros_like(param)
        flat_param = param.flatten()
        flat_grad = grad.flatten()

        # Sample subset of parameters for efficiency
        sample_size = min(100, len(flat_param))
        indices = np.random.choice(len(flat_param), sample_size, replace=False)

        for i in indices:
            flat_param[i] += eps
            loss_plus = loss_fn(param)
            flat_param[i] -= 2 * eps
            loss_minus = loss_fn(param)
            flat_param[i] += eps

            flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        return grad

    def _actor_loss(self, agent: MAPPOAgent, obs: np.ndarray, actions: np.ndarray,
                   advantages: np.ndarray, param_key: str, param_value: np.ndarray) -> float:
        """Compute actor loss for gradient calculation"""
        # Temporarily set parameter
        params = agent.actor.get_parameters()
        old_value = params[param_key].copy()
        params[param_key] = param_value
        agent.actor.set_parameters(params)

        # Compute loss
        log_probs = agent.actor.get_log_prob(obs, actions)
        loss = -np.mean(log_probs * advantages)

        # Restore parameter
        params[param_key] = old_value
        agent.actor.set_parameters(params)

        return loss

    def _critic_loss(self, critic: CriticNetwork, obs: np.ndarray,
                    returns: np.ndarray, param_key: str, param_value: np.ndarray) -> float:
        """Compute critic loss for gradient calculation"""
        # Temporarily set parameter
        params = critic.get_parameters()
        old_value = params[param_key].copy()
        params[param_key] = param_value
        critic.set_parameters(params)

        # Compute loss
        values = critic.forward(obs).flatten()
        loss = np.mean((returns - values) ** 2)

        # Restore parameter
        params[param_key] = old_value
        critic.set_parameters(params)

        return loss

    def _compute_gae(self, batch: List[MARLTransition]) -> Tuple[Dict[str, np.ndarray], 
                                                                  Dict[str, np.ndarray]]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Returns:
            advantages: Dict mapping agent_id to advantages array
            returns: Dict mapping agent_id to returns array
        """
        advantages = {agent_id: np.zeros(len(batch)) for agent_id in self.agent_ids}
        returns = {agent_id: np.zeros(len(batch)) for agent_id in self.agent_ids}

        for agent_id in self.agent_ids:
            # Get critic
            critic = self.shared_critic if self.config.use_shared_critic else self.agents[agent_id].critic

            # Compute values
            values = []
            next_values = []
            rewards = []

            for t in batch:
                if self.config.use_shared_critic:
                    obs = t.state.get_all_observations().reshape(1, -1)
                    next_obs = t.next_state.get_all_observations().reshape(1, -1)
                else:
                    obs = t.state.get_agent_obs(agent_id).reshape(1, -1)
                    next_obs = t.next_state.get_agent_obs(agent_id).reshape(1, -1)

                values.append(critic.forward(obs)[0, 0])
                next_values.append(critic.forward(next_obs)[0, 0] * (1 - t.done))
                rewards.append(t.reward.get_agent_reward(agent_id))

            values = np.array(values)
            next_values = np.array(next_values)
            rewards = np.array(rewards)

            # Compute TD errors
            deltas = rewards + self.config.gamma * next_values - values

            # Compute GAE
            gae = 0
            for t in reversed(range(len(batch))):
                gae = deltas[t] + self.config.gamma * self.config.lambda_gae * gae * (1 - batch[t].done)
                advantages[agent_id][t] = gae

            # Compute returns
            returns[agent_id] = advantages[agent_id] + values

        return advantages, returns

    def save_model(self, path: str) -> None:
        """Save all agent models"""
        import pickle
        models = {
            'agents': {agent_id: {
                'actor': agent.actor.get_parameters(),
                'critic': agent.critic.get_parameters() if agent.critic else None
            } for agent_id, agent in self.agents.items()},
            'shared_critic': self.shared_critic.get_parameters() if self.shared_critic else None,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(models, f)

    def load_model(self, path: str) -> None:
        """Load all agent models"""
        import pickle
        with open(path, 'rb') as f:
            models = pickle.load(f)

        for agent_id, agent in self.agents.items():
            agent.actor.set_parameters(models['agents'][agent_id]['actor'])
            if agent.critic and models['agents'][agent_id]['critic']:
                agent.critic.set_parameters(models['agents'][agent_id]['critic'])

        if self.shared_critic and models['shared_critic']:
            self.shared_critic.set_parameters(models['shared_critic'])


if __name__ == "__main__":
    print("MAPPO Trainer")
    print("Multi-Agent Proximal Policy Optimization")
