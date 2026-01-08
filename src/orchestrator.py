#!/usr/bin/env python3
"""
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time

from core.base import (IOrchestrator, ITask, INetworkTopology, 
                       ResourceAllocation, PerformanceMetrics, TaskState)
from solvers.tsa_solver import GraphAwareTSA, TaskAssignment
from solvers.rar_solver import MILPResourceSolver
from utils.utilities import ResourceUtilities, PerformanceEstimatorEMA, CostModel


@dataclass
class OrchestrationConfig:
    """Configuration for orchestrator"""
    max_iterations: int = 50
    convergence_threshold: float = 1e-4
    ema_alpha: float = 0.3
    w_reward: float = 1.0
    w_qos: float = 1000.0
    noise_level: float = 0.0


class OFLrAppOrchestrator(IOrchestrator):
    """
    Main O-FL rApp Orchestrator

    Coordinates the complete iterative protocol:
    1. TSA: Task and Slice Assignment
    2. RAR: Resource Allocation and Routing
    3. Execution: Distributed FL simulation
    4. Feedback: Performance estimation with EMA
    """

    def __init__(self, config: OrchestrationConfig):
        """
        Initialize orchestrator

        Args:
            config: Orchestration configuration
        """
        self.config = config

        # Components
        self.tsa_solver = GraphAwareTSA(config.w_reward, config.w_qos)
        self.rar_solver = MILPResourceSolver(w_reward=config.w_reward, 
                                            w_qos=config.w_qos)
        self.utilities = ResourceUtilities()
        self.estimator = PerformanceEstimatorEMA(alpha=config.ema_alpha)
        self.cost_model = CostModel()

        # State
        self.tasks: List[ITask] = []
        self.topology: Optional[INetworkTopology] = None
        self.current_assignment: Optional[TaskAssignment] = None
        self.current_allocations: Dict[str, ResourceAllocation] = {}

        # Metrics
        self.metrics_history: List[PerformanceMetrics] = []
        self.iteration = 0
        self.prev_objective = float('inf')

    def initialize(self, tasks: List[ITask], topology: INetworkTopology) -> None:
        """Initialize orchestrator with tasks and topology"""
        self.tasks = tasks
        self.topology = topology
        self.iteration = 0
        self.metrics_history = []

        # Initialize performance estimates
        for task in tasks:
            task_id = task.get_id()
            self.estimator.initialize(f'reward_{task_id}', 100.0)
            self.estimator.initialize(f'qos_vio_{task_id}', 0.0)

    def run_iteration(self) -> PerformanceMetrics:
        """
        Run one complete orchestration iteration

        Returns:
            PerformanceMetrics for this iteration
        """
        self.iteration += 1
        start_time = time.time()

        # Stage 1: TSA
        performance_estimates = self.estimator.get_all_estimates()
        self.current_assignment = self.tsa_solver.solve(
            self.tasks, self.topology, performance_estimates
        )

        # Stage 2: RAR
        self.current_allocations = self.rar_solver.solve(
            self.tasks, self.current_assignment, 
            self.topology, performance_estimates
        )

        # Stage 3: Execute FL tasks
        task_metrics = self._execute_fl_tasks()

        # Stage 4: Update estimates
        for task_id, metrics in task_metrics.items():
            self.estimator.update(f'reward_{task_id}', metrics.global_reward)
            self.estimator.update(f'qos_vio_{task_id}', metrics.qos_violation)

        # Compute convergence
        current_obj = sum(m.resource_cost - self.config.w_reward * m.global_reward +
                         self.config.w_qos * m.qos_violation
                         for m in task_metrics.values())
        convergence = abs(current_obj - self.prev_objective) / (abs(self.prev_objective) + 1e-6)
        self.prev_objective = current_obj

        # Aggregate metrics
        aggregate_metrics = PerformanceMetrics(
            task_id='aggregate',
            global_reward=sum(m.global_reward for m in task_metrics.values()),
            qos_violation=sum(m.qos_violation for m in task_metrics.values()),
            resource_cost=sum(m.resource_cost for m in task_metrics.values()),
            convergence_rate=convergence,
            success=all(m.success for m in task_metrics.values())
        )

        self.metrics_history.append(aggregate_metrics)

        return aggregate_metrics

    def _execute_fl_tasks(self) -> Dict[str, PerformanceMetrics]:
    
        task_metrics = {}

        for task in self.tasks:
            if task.get_state() != TaskState.ACTIVE:
                continue

            task_id = task.get_id()
            allocation = self.current_allocations.get(task_id)

            if not allocation:
                continue

            # Compute performance
            actual_performance = self._simulate_task_execution(task, allocation)

            # Compute reward
            reward = task.compute_reward(allocation, actual_performance)

            # Add noise
            if self.config.noise_level > 0:
                import numpy as np
                noise = np.random.normal(0, self.config.noise_level * abs(reward))
                reward += noise

            # Compute costs
            links = self.topology.get_links()
            link_costs = {lid: props['cost'] * props['bandwidth'] 
                         for lid, props in links.items()}
            costs = self.cost_model.compute_cost(allocation, link_costs)

            # Create metrics
            metrics = PerformanceMetrics(
                task_id=task_id,
                global_reward=max(0, reward),
                qos_violation=actual_performance.get('qos_violation', 0),
                resource_cost=costs['total'],
                latency=actual_performance.get('latency'),
                success=actual_performance.get('success', False)
            )

            task_metrics[task_id] = metrics

        return task_metrics

    def _simulate_task_execution(self, task: ITask, 
                                 allocation: ResourceAllocation) -> Dict:
        
        requirements = task.get_requirements()

        # Compute utilities
        agg_util = self.utilities.compute_aggregator_utility(
            allocation.compute_aggregator,
            requirements.compute_aggregator
        )

        comm_util = self.utilities.compute_communication_utility(
            allocation.get_total_bandwidth(),
            requirements.data_transfer / 1000.0
        )

        # Compute latencies
        comm_latency = self._compute_communication_latency(allocation)
        comp_latency = self._compute_computation_latency(allocation, requirements)
        total_latency = comm_latency + comp_latency

        # Check QoS
        qos_violation = 0.0
        success = True

        if requirements.latency_max is not None:
            qos_violation = max(0, total_latency - requirements.latency_max)
            success = (qos_violation == 0)

        return {
            'latency': total_latency,
            'qos_violation': qos_violation,
            'success': success,
            'agg_utility': agg_util,
            'comm_utility': comm_util
        }

    def _compute_communication_latency(self, allocation: ResourceAllocation) -> float:
        """Compute communication latency"""
        if not allocation.routing:
            return 0.0

        links = self.topology.get_links()
        latencies = [links[link_id]['latency'] 
                    for link_id in allocation.routing.values()
                    if link_id in links]

        return max(latencies) if latencies else 0.0

    def _compute_computation_latency(self, allocation: ResourceAllocation,
                                    requirements) -> float:
        """Compute computation latency"""
        if allocation.compute_aggregator > 1e-6:
            return 5.0 * (requirements.compute_aggregator / allocation.compute_aggregator)
        return float('inf')

    def check_convergence(self) -> bool:
        """Check if orchestration has converged"""
        if len(self.metrics_history) < 2:
            return False

        return self.metrics_history[-1].convergence_rate < self.config.convergence_threshold

    def get_current_allocation(self) -> Dict[str, ResourceAllocation]:
        return self.current_allocations.copy()

    def run_until_convergence(self) -> List[PerformanceMetrics]:
        """Run orchestration until convergence"""
        print(f"\n{'='*70}")
        print("O-FL rApp Orchestration (Object-Oriented)")
        print(f"{'='*70}")
        print(f"Tasks: {[t.get_id() for t in self.tasks]}")
        print(f"Max iterations: {self.config.max_iterations}\n")

        for i in range(self.config.max_iterations):
            metrics = self.run_iteration()

            print(f"Iteration {i+1}:")
            print(f"  Total Reward: {metrics.global_reward:.2f}")
            print(f"  Total Cost: {metrics.resource_cost:.2f}")
            print(f"  QoS Violations: {metrics.qos_violation:.2f}")
            print(f"  Success: {metrics.success}")
            print(f"  Convergence: {metrics.convergence_rate:.6f}\n")

            if self.check_convergence():
                print(f"✓ Converged after {i+1} iterations!")
                break

        return self.metrics_history


if __name__ == "__main__":
    print("OFL-rApp Orchestrator (Object-Oriented)")
    print("Use main.py to run complete simulations")
