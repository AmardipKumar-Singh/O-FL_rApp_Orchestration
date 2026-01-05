#!/usr/bin/env python3
"""
training/marl_base.py - Abstract MARL Trainer Interface
O-FL rApp: Multi-Agent Reinforcement Learning Base Classes
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MARLState:
    """MARL environment state"""
    observations: Dict[str, np.ndarray]  # agent_id -> observation
    global_state: Optional[np.ndarray] = None  # For centralized training
    available_actions: Optional[Dict[str, np.ndarray]] = None

    def get_agent_obs(self, agent_id: str) -> np.ndarray:
        return self.observations.get(agent_id, np.array([]))

    def get_all_observations(self) -> np.ndarray:
        """Stack all agent observations"""
        return np.concatenate(list(self.observations.values()))


@dataclass
class MARLAction:
    """MARL actions from agents"""
    actions: Dict[str, np.ndarray]  # agent_id -> action

    def get_agent_action(self, agent_id: str) -> np.ndarray:
        return self.actions.get(agent_id, np.array([]))

    def get_all_actions(self) -> np.ndarray:
        """Stack all agent actions"""
        return np.concatenate(list(self.actions.values()))


@dataclass
class MARLReward:
    """MARL rewards"""
    individual_rewards: Dict[str, float]  # agent_id -> reward
    team_reward: float = 0.0

    def get_agent_reward(self, agent_id: str) -> float:
        return self.individual_rewards.get(agent_id, 0.0)

    def get_total_reward(self) -> float:
        return sum(self.individual_rewards.values())


@dataclass
class MARLTransition:
    """Single MARL transition for experience replay"""
    state: MARLState
    action: MARLAction
    reward: MARLReward
    next_state: MARLState
    done: bool
    info: Dict = field(default_factory=dict)


class ReplayBuffer:
    """Experience replay buffer for MARL"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[MARLTransition] = []
        self.position = 0

    def push(self, transition: MARLTransition) -> None:
        """Add transition to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[MARLTransition]:
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def sample_last(self, n: int) -> List[MARLTransition]:
        """Sample last n transitions (for on-policy algorithms)"""
        return self.buffer[-n:]

    def clear(self) -> None:
        """Clear buffer"""
        self.buffer.clear()
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)


class MARLEnvironment(ABC):
    """Abstract MARL environment interface"""

    @abstractmethod
    def reset(self) -> MARLState:
        """Reset environment and return initial state"""
        pass

    @abstractmethod
    def step(self, action: MARLAction) -> Tuple[MARLState, MARLReward, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        pass

    @abstractmethod
    def get_observation_space(self, agent_id: str) -> Tuple[int, ...]:
        """Get observation space shape for agent"""
        pass

    @abstractmethod
    def get_action_space(self, agent_id: str) -> Tuple[int, ...]:
        """Get action space shape for agent"""
        pass

    @abstractmethod
    def get_agent_ids(self) -> List[str]:
        """Get list of all agent IDs"""
        pass


class MARLTrainer(ABC):
    """Abstract MARL trainer interface"""

    @abstractmethod
    def select_actions(self, state: MARLState, explore: bool = True) -> MARLAction:
        """Select actions for all agents given state"""
        pass

    @abstractmethod
    def train_step(self, batch: List[MARLTransition]) -> Dict[str, float]:
        """Perform one training step"""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model parameters"""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model parameters"""
        pass


if __name__ == "__main__":
    print("MARL Base Classes")
