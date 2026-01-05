#!/usr/bin/env python3
"""
baselines/baseline_algorithms.py - Baseline Algorithms for Comparison
O-FL rApp: Baseline orchestration approaches

Implements four baseline algorithms:
1. Independent FL - No coordination between tasks
2. Static Partitioning - Fixed equal resource division
3. Auction-Based - Tasks bid for resources
4. Priority-Based - Fixed priorities (uRLLC > eMBB)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from core.base import (ITask, INetworkTopology, IOrchestrator, 
                      PerformanceMetrics, TaskState)
from models.task import TaskType


@dataclass
class BaselineConfig:
    """Configuration for baseline algorithms"""
    max_iterations: int = 50
    noise_level: float = 0.0


class IndependentFLOrchestrator(IOrchestrator):
    """
    Baseline 1: Independent FL

    Each task allocates resources independently without coordination.
    No awareness of other tasks or network constraints.
    - Simple greedy allocation
    - No resource sharing optimization
    - Tasks may violate capacity constraints
    """

    def __init__(self, topology: INetworkTopology, config: BaselineConfig):
        self.topology = topology
        self.config = config
        self.tasks: List[ITask] = []
        self.iteration = 0
        self.metrics_history: List[PerformanceMetrics] = []

    def initialize(self, tasks: List[ITask], 
                  topology: Optional[INetworkTopology] = None) -> None:
        """Initialize with tasks"""
        self.tasks = tasks
        if topology:
            self.topology = topology

        # Set all tasks to active
        for task in tasks:
            task.set_state(TaskState.ACTIVE)

    def run_iteration(self) -> List[PerformanceMetrics]:
        """Run one iteration - each task allocates independently"""
        iteration_metrics = []

        # Get total available resources
        nodes = self.topology.get_nodes()
        links = self.topology.get_links()

        for task in self.tasks:
            if task.get_state() != TaskState.ACTIVE:
                continue

            task_id = task.get_id()
            agents = task.get_agents()

            # Greedy allocation: divide resources equally among agents
            num_agents = len(agents)

            # Compute allocation (simple equal division)
            compute_per_agent = 100.0 / max(1, len(self.tasks) * num_agents)

            # Bandwidth allocation (simple equal division)
            total_bandwidth = sum(props['bandwidth'] for props in links.values())
            bandwidth_per_task = total_bandwidth / max(1, len(self.tasks))

            # Simulate performance (no real optimization)
            # Performance degrades due to lack of coordination
            base_reward = task.get_base_reward()
            coordination_penalty = 0.3 * (len(self.tasks) - 1) / max(1, len(self.tasks))
            actual_reward = base_reward * (1 - coordination_penalty)

            # Add noise
            if self.config.noise_level > 0:
                noise = np.random.normal(0, self.config.noise_level * actual_reward)
                actual_reward += noise

            # Check for resource conflicts (likely to violate constraints)
            qos_violation = 0.0
            if len(self.tasks) > 2:
                # High contention causes QoS violations
                qos_violation = 5.0 * (len(self.tasks) - 2)

            # Create metrics
            metrics = PerformanceMetrics(
                task_id=task_id,
                global_reward=max(0, actual_reward),
                qos_violation=qos_violation,
                resource_cost=compute_per_agent * num_agents + bandwidth_per_task,
                latency=10.0,  # Fixed latency (no optimization)
                success=qos_violation < 1.0
            )

            iteration_metrics.append(metrics)
            self.metrics_history.append(metrics)

        self.iteration += 1
        return iteration_metrics

    def run_until_convergence(self) -> List[PerformanceMetrics]:
        """Run for fixed number of iterations"""
        print(f"\n{'='*70}")
        print("BASELINE 1: Independent FL")
        print(f"{'='*70}")
        print("No coordination - each task allocates independently\n")

        for i in range(self.config.max_iterations):
            metrics = self.run_iteration()

            if (i + 1) % 10 == 0:
                avg_reward = np.mean([m.global_reward for m in metrics])
                avg_qos_vio = np.mean([m.qos_violation for m in metrics])
                print(f"Iteration {i+1}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg QoS Violation = {avg_qos_vio:.2f}")

        print(f"\nCompleted {self.config.max_iterations} iterations\n")
        return self.metrics_history


class StaticPartitioningOrchestrator(IOrchestrator):
    """
    Baseline 2: Static Resource Partitioning

    Fixed equal division of resources among tasks.
    - No adaptation to task requirements
    - No consideration of task priorities
    - Fair but inefficient allocation
    """

    def __init__(self, topology: INetworkTopology, config: BaselineConfig):
        self.topology = topology
        self.config = config
        self.tasks: List[ITask] = []
        self.iteration = 0
        self.metrics_history: List[PerformanceMetrics] = []

    def initialize(self, tasks: List[ITask],
                  topology: Optional[INetworkTopology] = None) -> None:
        """Initialize with tasks"""
        self.tasks = tasks
        if topology:
            self.topology = topology

        for task in tasks:
            task.set_state(TaskState.ACTIVE)

    def run_iteration(self) -> List[PerformanceMetrics]:
        """Run iteration with static equal partitioning"""
        iteration_metrics = []

        num_tasks = len([t for t in self.tasks if t.get_state() == TaskState.ACTIVE])
        if num_tasks == 0:
            return iteration_metrics

        # Static equal partitioning
        compute_share = 100.0 / num_tasks
        links = self.topology.get_links()
        total_bandwidth = sum(props['bandwidth'] for props in links.values())
        bandwidth_share = total_bandwidth / num_tasks

        for task in self.tasks:
            if task.get_state() != TaskState.ACTIVE:
                continue

            task_id = task.get_id()
            requirements = task.get_requirements()

            # Check if static allocation meets requirements
            compute_needed = requirements.compute_aggregator
            bandwidth_needed = requirements.data_transfer

            # Performance based on requirement satisfaction
            compute_ratio = min(1.0, compute_share / compute_needed)
            bandwidth_ratio = min(1.0, bandwidth_share / bandwidth_needed)
            satisfaction = (compute_ratio + bandwidth_ratio) / 2

            base_reward = task.get_base_reward()
            actual_reward = base_reward * satisfaction

            # QoS violation if requirements not met
            qos_violation = 0.0
            if requirements.latency_max is not None:
                # Static allocation often violates latency
                qos_violation = max(0, 15.0 - requirements.latency_max)

            metrics = PerformanceMetrics(
                task_id=task_id,
                global_reward=max(0, actual_reward),
                qos_violation=qos_violation,
                resource_cost=compute_share + bandwidth_share,
                latency=15.0,  # Fixed latency
                success=satisfaction > 0.8
            )

            iteration_metrics.append(metrics)
            self.metrics_history.append(metrics)

        self.iteration += 1
        return iteration_metrics

    def run_until_convergence(self) -> List[PerformanceMetrics]:
        """Run for fixed iterations"""
        print(f"\n{'='*70}")
        print("BASELINE 2: Static Resource Partitioning")
        print(f"{'='*70}")
        print("Fixed equal division among all tasks\n")

        for i in range(self.config.max_iterations):
            metrics = self.run_iteration()

            if (i + 1) % 10 == 0:
                avg_reward = np.mean([m.global_reward for m in metrics])
                avg_cost = np.mean([m.resource_cost for m in metrics])
                print(f"Iteration {i+1}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Cost = {avg_cost:.2f}")

        print(f"\nCompleted {self.config.max_iterations} iterations\n")
        return self.metrics_history


class AuctionBasedOrchestrator(IOrchestrator):
    """
    Baseline 3: Auction-Based Resource Allocation

    Tasks bid for resources based on their utility.
    - Market-based mechanism
    - Tasks with higher bids get more resources
    - May not satisfy QoS constraints
    """

    def __init__(self, topology: INetworkTopology, config: BaselineConfig):
        self.topology = topology
        self.config = config
        self.tasks: List[ITask] = []
        self.iteration = 0
        self.metrics_history: List[PerformanceMetrics] = []
        self.task_budgets: Dict[str, float] = {}

    def initialize(self, tasks: List[ITask],
                  topology: Optional[INetworkTopology] = None) -> None:
        """Initialize with tasks and budgets"""
        self.tasks = tasks
        if topology:
            self.topology = topology

        # Initialize budgets based on task rewards
        for task in tasks:
            task.set_state(TaskState.ACTIVE)
            self.task_budgets[task.get_id()] = task.get_base_reward()

    def run_iteration(self) -> List[PerformanceMetrics]:
        """Run auction-based allocation"""
        iteration_metrics = []

        active_tasks = [t for t in self.tasks if t.get_state() == TaskState.ACTIVE]
        if not active_tasks:
            return iteration_metrics

        # Auction: tasks bid based on their budget
        total_budget = sum(self.task_budgets[t.get_id()] for t in active_tasks)

        links = self.topology.get_links()
        total_bandwidth = sum(props['bandwidth'] for props in links.values())
        total_compute = 100.0

        for task in active_tasks:
            task_id = task.get_id()
            budget = self.task_budgets[task_id]

            # Allocation proportional to budget
            budget_ratio = budget / total_budget if total_budget > 0 else 0

            compute_allocated = total_compute * budget_ratio
            bandwidth_allocated = total_bandwidth * budget_ratio

            # Performance based on allocation
            requirements = task.get_requirements()
            compute_satisfaction = min(1.0, compute_allocated / requirements.compute_aggregator)
            bandwidth_satisfaction = min(1.0, bandwidth_allocated / requirements.data_transfer)

            overall_satisfaction = (compute_satisfaction + bandwidth_satisfaction) / 2

            base_reward = task.get_base_reward()
            actual_reward = base_reward * overall_satisfaction

            # Cost = bid amount
            cost = budget * 0.1  # 10% of budget

            # QoS violations possible
            qos_violation = 0.0
            if overall_satisfaction < 0.7:
                qos_violation = (0.7 - overall_satisfaction) * 10

            metrics = PerformanceMetrics(
                task_id=task_id,
                global_reward=max(0, actual_reward),
                qos_violation=qos_violation,
                resource_cost=cost,
                latency=12.0,
                success=overall_satisfaction > 0.7
            )

            iteration_metrics.append(metrics)
            self.metrics_history.append(metrics)

            # Update budget based on performance
            self.task_budgets[task_id] *= (1 + 0.1 * overall_satisfaction)

        self.iteration += 1
        return iteration_metrics

    def run_until_convergence(self) -> List[PerformanceMetrics]:
        """Run auction-based allocation"""
        print(f"\n{'='*70}")
        print("BASELINE 3: Auction-Based Orchestration")
        print(f"{'='*70}")
        print("Market mechanism - tasks bid for resources\n")

        for i in range(self.config.max_iterations):
            metrics = self.run_iteration()

            if (i + 1) % 10 == 0:
                avg_reward = np.mean([m.global_reward for m in metrics])
                total_budget = sum(self.task_budgets.values())
                print(f"Iteration {i+1}: Avg Reward = {avg_reward:.2f}, "
                      f"Total Budget = {total_budget:.2f}")

        print(f"\nCompleted {self.config.max_iterations} iterations\n")
        return self.metrics_history


class PriorityBasedOrchestrator(IOrchestrator):
    """
    Baseline 4: Priority-Based Scheduling

    Fixed priorities: uRLLC > eMBB > Others
    - Simple priority rules
    - Sequential allocation
    - No optimization within priority levels
    """

    def __init__(self, topology: INetworkTopology, config: BaselineConfig):
        self.topology = topology
        self.config = config
        self.tasks: List[ITask] = []
        self.iteration = 0
        self.metrics_history: List[PerformanceMetrics] = []

    def initialize(self, tasks: List[ITask],
                  topology: Optional[INetworkTopology] = None) -> None:
        """Initialize with tasks"""
        self.tasks = tasks
        if topology:
            self.topology = topology

        # Sort by priority
        self.tasks.sort(key=lambda t: self._get_priority(t), reverse=True)

        for task in tasks:
            task.set_state(TaskState.ACTIVE)

    def _get_priority(self, task: ITask) -> int:
        """Get priority based on task type"""
        task_type = task.get_type()
        if task_type == TaskType.LATENCY_CRITICAL:
            return 3  # Highest (uRLLC)
        elif task_type == TaskType.THROUGHPUT_ORIENTED:
            return 2  # Medium (eMBB)
        else:
            return 1  # Lowest

    def run_iteration(self) -> List[PerformanceMetrics]:
        """Run priority-based allocation"""
        iteration_metrics = []

        links = self.topology.get_links()
        total_bandwidth = sum(props['bandwidth'] for props in links.values())
        total_compute = 100.0

        # Remaining resources
        remaining_bandwidth = total_bandwidth
        remaining_compute = total_compute

        # Allocate in priority order
        for task in self.tasks:
            if task.get_state() != TaskState.ACTIVE:
                continue

            task_id = task.get_id()
            requirements = task.get_requirements()

            # Allocate requested amount (if available)
            compute_requested = requirements.compute_aggregator
            bandwidth_requested = requirements.data_transfer

            compute_allocated = min(compute_requested, remaining_compute)
            bandwidth_allocated = min(bandwidth_requested, remaining_bandwidth)

            # Update remaining
            remaining_compute -= compute_allocated
            remaining_bandwidth -= bandwidth_allocated

            # Performance
            compute_satisfaction = compute_allocated / compute_requested if compute_requested > 0 else 1.0
            bandwidth_satisfaction = bandwidth_allocated / bandwidth_requested if bandwidth_requested > 0 else 1.0

            overall_satisfaction = min(compute_satisfaction, bandwidth_satisfaction)

            base_reward = task.get_base_reward()
            actual_reward = base_reward * overall_satisfaction

            # QoS violation if not fully satisfied
            qos_violation = 0.0
            if overall_satisfaction < 1.0:
                qos_violation = (1.0 - overall_satisfaction) * 5

            # Lower priority tasks suffer more
            priority = self._get_priority(task)
            if priority < 3:
                qos_violation *= (4 - priority)

            metrics = PerformanceMetrics(
                task_id=task_id,
                global_reward=max(0, actual_reward),
                qos_violation=qos_violation,
                resource_cost=compute_allocated + bandwidth_allocated,
                latency=8.0 if priority == 3 else 20.0,
                success=overall_satisfaction > 0.9
            )

            iteration_metrics.append(metrics)
            self.metrics_history.append(metrics)

        self.iteration += 1
        return iteration_metrics

    def run_until_convergence(self) -> List[PerformanceMetrics]:
        """Run priority-based scheduling"""
        print(f"\n{'='*70}")
        print("BASELINE 4: Priority-Based Scheduling")
        print(f"{'='*70}")
        print("Fixed priorities: uRLLC > eMBB > Others\n")

        for i in range(self.config.max_iterations):
            metrics = self.run_iteration()

            if (i + 1) % 10 == 0:
                avg_reward = np.mean([m.global_reward for m in metrics])
                avg_qos = np.mean([m.qos_violation for m in metrics])
                print(f"Iteration {i+1}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg QoS Vio = {avg_qos:.2f}")

        print(f"\nCompleted {self.config.max_iterations} iterations\n")
        return self.metrics_history


def compare_baselines(tasks: List[ITask], topology: INetworkTopology,
                     iterations: int = 50) -> Dict:
    """
    Compare all baseline algorithms

    Returns:
        Dictionary with performance metrics for each baseline
    """
    config = BaselineConfig(max_iterations=iterations)

    results = {}

    # Baseline 1: Independent FL
    orch1 = IndependentFLOrchestrator(topology, config)
    orch1.initialize(tasks.copy(), topology)
    metrics1 = orch1.run_until_convergence()
    results['Independent_FL'] = {
        'avg_reward': np.mean([m.global_reward for m in metrics1]),
        'avg_qos_violation': np.mean([m.qos_violation for m in metrics1]),
        'avg_cost': np.mean([m.resource_cost for m in metrics1]),
        'success_rate': np.mean([m.success for m in metrics1])
    }

    # Baseline 2: Static Partitioning
    orch2 = StaticPartitioningOrchestrator(topology, config)
    orch2.initialize(tasks.copy(), topology)
    metrics2 = orch2.run_until_convergence()
    results['Static_Partitioning'] = {
        'avg_reward': np.mean([m.global_reward for m in metrics2]),
        'avg_qos_violation': np.mean([m.qos_violation for m in metrics2]),
        'avg_cost': np.mean([m.resource_cost for m in metrics2]),
        'success_rate': np.mean([m.success for m in metrics2])
    }

    # Baseline 3: Auction-Based
    orch3 = AuctionBasedOrchestrator(topology, config)
    orch3.initialize(tasks.copy(), topology)
    metrics3 = orch3.run_until_convergence()
    results['Auction_Based'] = {
        'avg_reward': np.mean([m.global_reward for m in metrics3]),
        'avg_qos_violation': np.mean([m.qos_violation for m in metrics3]),
        'avg_cost': np.mean([m.resource_cost for m in metrics3]),
        'success_rate': np.mean([m.success for m in metrics3])
    }

    # Baseline 4: Priority-Based
    orch4 = PriorityBasedOrchestrator(topology, config)
    orch4.initialize(tasks.copy(), topology)
    metrics4 = orch4.run_until_convergence()
    results['Priority_Based'] = {
        'avg_reward': np.mean([m.global_reward for m in metrics4]),
        'avg_qos_violation': np.mean([m.qos_violation for m in metrics4]),
        'avg_cost': np.mean([m.resource_cost for m in metrics4]),
        'success_rate': np.mean([m.success for m in metrics4])
    }

    # Print comparison
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON")
    print(f"{'='*70}\n")

    for name, perf in results.items():
        print(f"{name}:")
        print(f"  Avg Reward: {perf['avg_reward']:.2f}")
        print(f"  Avg QoS Violation: {perf['avg_qos_violation']:.2f}")
        print(f"  Avg Cost: {perf['avg_cost']:.2f}")
        print(f"  Success Rate: {perf['success_rate']:.1%}\n")

    return results


if __name__ == "__main__":
    from models.network import TopologyBuilder
    from models.task import TaskBuilder, TaskType

    # Create sample topology
    topology = (TopologyBuilder()
        .add_odu('ODU_1', 80)
        .add_odu('ODU_2', 80)
        .add_ric('RIC', 100)
        .add_fiber_link('ODU_1', 'RIC', 10)
        .add_fiber_link('ODU_2', 'RIC', 10)
        .build())

    # Create sample tasks
    task1 = (TaskBuilder()
        .with_id('T1')
        .with_type(TaskType.LATENCY_CRITICAL)
        .with_agents(['a1', 'a2'])
        .with_compute_requirements(5, 10)
        .with_data_transfer(10)
        .with_latency_budget(5.0)
        .with_priority(2.0)
        .with_reward(100)
        .build())

    task2 = (TaskBuilder()
        .with_id('T2')
        .with_type(TaskType.THROUGHPUT_ORIENTED)
        .with_agents(['a3', 'a4'])
        .with_compute_requirements(5, 10)
        .with_data_transfer(50)
        .with_priority(1.0)
        .with_reward(80)
        .build())

    # Compare baselines
    results = compare_baselines([task1, task2], topology, iterations=20)
