#!/usr/bin/env python3
"""
models/task.py - Task Implementations
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks
"""

import numpy as np
from typing import Dict, List, Optional
from core.base import (ITask, TaskType, TaskState, ResourceRequirements, 
                       ResourceAllocation, ComponentFactory)


class FederatedMARLTask(ITask):
    """Base implementation of Federated MARL Task"""

    def __init__(self, task_id: str, task_type: TaskType, agents: List[str],
                 requirements: ResourceRequirements, priority: float = 1.0,
                 reward_params: Optional[Dict] = None):
        """
        Initialize Federated MARL Task

        Args:
            task_id: Unique task identifier
            task_type: Type of task
            agents: List of agent IDs
            requirements: Resource requirements
            priority: Task priority (default 1.0)
            reward_params: Parameters for reward computation
        """
        self._task_id = task_id
        self._task_type = task_type
        self._agents = agents
        self._requirements = requirements
        self._priority = priority
        self._state = TaskState.PENDING
        self._reward_params = reward_params or {}

    def get_id(self) -> str:
        return self._task_id

    def get_type(self) -> TaskType:
        return self._task_type

    def get_agents(self) -> List[str]:
        return self._agents.copy()

    def get_requirements(self) -> ResourceRequirements:
        return self._requirements

    def get_priority(self) -> float:
        return self._priority

    def get_state(self) -> TaskState:
        return self._state

    def set_state(self, state: TaskState) -> None:
        self._state = state

    def compute_reward(self, allocation: ResourceAllocation, 
                      actual_performance: Dict) -> float:
        """Base reward computation - to be overridden"""
        return 0.0

    def __repr__(self) -> str:
        return (f"FederatedMARLTask(id={self._task_id}, type={self._task_type.value}, "
                f"agents={len(self._agents)}, state={self._state.value})")


class ThroughputOrientedTask(FederatedMARLTask):
    """
    Throughput-Oriented Task (eMBB)
    Reward: R_global = base * log(1 + β_total)
    """

    def __init__(self, task_id: str, agents: List[str],
                 requirements: ResourceRequirements, priority: float = 1.0,
                 base_reward: float = 200.0):
        reward_params = {'base': base_reward, 'type': 'log'}
        super().__init__(task_id, TaskType.THROUGHPUT_ORIENTED, agents,
                        requirements, priority, reward_params)
        self._base_reward = base_reward

    def compute_reward(self, allocation: ResourceAllocation, 
                      actual_performance: Dict) -> float:
        """
        Compute logarithmic reward for throughput
        R_global = base * log(1 + β_total)
        """
        total_bandwidth = allocation.get_total_bandwidth()
        return self._base_reward * np.log(1.0 + total_bandwidth)


class LatencyCriticalTask(FederatedMARLTask):
    """
    Latency-Critical Task (uRLLC)
    Reward: Fixed reward if latency constraint met, 0 otherwise
    """

    def __init__(self, task_id: str, agents: List[str],
                 requirements: ResourceRequirements, priority: float = 2.0,
                 latency_budget: float = 6.0, base_reward: float = 50.0):
        reward_params = {'base': base_reward, 'type': 'fixed', 
                        'latency_budget': latency_budget}
        super().__init__(task_id, TaskType.LATENCY_CRITICAL, agents,
                        requirements, priority, reward_params)
        self._latency_budget = latency_budget
        self._base_reward = base_reward

        # Set latency constraint in requirements
        self._requirements.latency_max = latency_budget

    def compute_reward(self, allocation: ResourceAllocation, 
                      actual_performance: Dict) -> float:
        """
        Compute fixed reward if latency constraint satisfied
        """
        actual_latency = actual_performance.get('latency', float('inf'))

        if actual_latency <= self._latency_budget:
            return self._base_reward
        return 0.0

    def get_latency_budget(self) -> float:
        """Get latency budget for this task"""
        return self._latency_budget


class MixedTask(FederatedMARLTask):
    """
    Mixed Task - combines throughput and latency considerations
    """

    def __init__(self, task_id: str, agents: List[str],
                 requirements: ResourceRequirements, priority: float = 1.5,
                 base_reward: float = 150.0, latency_weight: float = 0.3):
        reward_params = {'base': base_reward, 'type': 'mixed',
                        'latency_weight': latency_weight}
        super().__init__(task_id, TaskType.MIXED, agents,
                        requirements, priority, reward_params)
        self._base_reward = base_reward
        self._latency_weight = latency_weight

    def compute_reward(self, allocation: ResourceAllocation, 
                      actual_performance: Dict) -> float:
        """
        Compute mixed reward combining throughput and latency
        """
        total_bandwidth = allocation.get_total_bandwidth()
        throughput_reward = self._base_reward * np.log(1.0 + total_bandwidth)

        # Latency penalty
        actual_latency = actual_performance.get('latency', 10.0)
        latency_penalty = self._latency_weight * actual_latency

        return max(0, throughput_reward - latency_penalty)


class TaskBuilder:
    """Builder pattern for creating tasks"""

    def __init__(self):
        self._task_id = None
        self._task_type = None
        self._agents = []
        self._compute_agent = 0.0
        self._compute_aggregator = 0.0
        self._data_transfer = 0.0
        self._priority = 1.0
        self._base_reward = 100.0
        self._latency_budget = None

    def with_id(self, task_id: str) -> 'TaskBuilder':
        self._task_id = task_id
        return self

    def with_type(self, task_type: TaskType) -> 'TaskBuilder':
        self._task_type = task_type
        return self

    def with_agents(self, agents: List[str]) -> 'TaskBuilder':
        self._agents = agents
        return self

    def with_compute_requirements(self, agent: float, aggregator: float) -> 'TaskBuilder':
        self._compute_agent = agent
        self._compute_aggregator = aggregator
        return self

    def with_data_transfer(self, data_transfer: float) -> 'TaskBuilder':
        self._data_transfer = data_transfer
        return self

    def with_priority(self, priority: float) -> 'TaskBuilder':
        self._priority = priority
        return self

    def with_reward(self, base_reward: float) -> 'TaskBuilder':
        self._base_reward = base_reward
        return self

    def with_latency_budget(self, latency_budget: float) -> 'TaskBuilder':
        self._latency_budget = latency_budget
        return self

    def build(self) -> ITask:
        """Build the task"""
        if not self._task_id or not self._task_type or not self._agents:
            raise ValueError("Task ID, type, and agents are required")

        requirements = ResourceRequirements(
            compute_agent=self._compute_agent,
            compute_aggregator=self._compute_aggregator,
            data_transfer=self._data_transfer,
            latency_max=self._latency_budget
        )

        if self._task_type == TaskType.THROUGHPUT_ORIENTED:
            return ThroughputOrientedTask(
                self._task_id, self._agents, requirements,
                self._priority, self._base_reward
            )
        elif self._task_type == TaskType.LATENCY_CRITICAL:
            if self._latency_budget is None:
                raise ValueError("Latency budget required for latency-critical task")
            return LatencyCriticalTask(
                self._task_id, self._agents, requirements,
                self._priority, self._latency_budget, self._base_reward
            )
        elif self._task_type == TaskType.MIXED:
            return MixedTask(
                self._task_id, self._agents, requirements,
                self._priority, self._base_reward
            )
        else:
            return FederatedMARLTask(
                self._task_id, self._task_type, self._agents,
                requirements, self._priority
            )


# Register task types with factory
ComponentFactory.register_task_type('throughput', ThroughputOrientedTask)
ComponentFactory.register_task_type('latency_critical', LatencyCriticalTask)
ComponentFactory.register_task_type('mixed', MixedTask)


if __name__ == "__main__":
    # Test task creation
    builder = TaskBuilder()

    task1 = (builder.with_id('T1')
             .with_type(TaskType.THROUGHPUT_ORIENTED)
             .with_agents(['a1', 'a2'])
             .with_compute_requirements(5.0, 10.0)
             .with_data_transfer(500.0)
             .with_priority(1.0)
             .with_reward(200.0)
             .build())

    print(f"Created: {task1}")
    print(f"  Requirements: {task1.get_requirements()}")
