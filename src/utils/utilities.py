#!/usr/bin/env python3
"""
utils/utilities.py - Utility Functions
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks
"""

import numpy as np
from typing import Dict, Optional
from core.base import IUtilityFunction, IQoSModel, ITask, ResourceAllocation


class LogarithmicUtility(IUtilityFunction):
    """
    Logarithmic utility function
    U(x) = k * log(1 + x/x_req)
    """

    def __init__(self, scaling_factor: float = 1.0):
        self.scaling_factor = scaling_factor

    def compute_utility(self, allocated_resource: float, 
                       required_resource: float) -> float:
        if required_resource <= 0:
            return 0.0
        return self.scaling_factor * np.log(1.0 + allocated_resource / required_resource)

    def get_derivative(self, allocated_resource: float, 
                      required_resource: float) -> float:
        if required_resource <= 0:
            return 0.0
        return self.scaling_factor / (required_resource + allocated_resource)


class SquareRootUtility(IUtilityFunction):
    """
    Square root utility function
    U(x) = k * sqrt(x/x_req)
    """

    def __init__(self, scaling_factor: float = 1.0):
        self.scaling_factor = scaling_factor

    def compute_utility(self, allocated_resource: float, 
                       required_resource: float) -> float:
        if required_resource <= 0:
            return 0.0
        return self.scaling_factor * np.sqrt(allocated_resource / required_resource)

    def get_derivative(self, allocated_resource: float, 
                      required_resource: float) -> float:
        if required_resource <= 0 or allocated_resource <= 0:
            return 0.0
        return 0.5 * self.scaling_factor / np.sqrt(allocated_resource * required_resource)


class ResourceUtilities:
    """Manager for different resource utility functions"""

    def __init__(self, k_u: float = 1.0, k_v: float = 1.0, 
                 k_f: float = 1.0, k_p: float = 1.0):
        """
        Initialize utility functions

        Args:
            k_u: Edge compute utility scaling
            k_v: Aggregator compute utility scaling
            k_f: Communication utility scaling
            k_p: Policy performance scaling
        """
        self.edge_compute_utility = LogarithmicUtility(k_u)
        self.aggregator_compute_utility = LogarithmicUtility(k_v)
        self.communication_utility = SquareRootUtility(k_f)
        self.policy_scaling = k_p

    def compute_edge_utility(self, allocated: float, required: float) -> float:
        """Compute edge compute utility"""
        return self.edge_compute_utility.compute_utility(allocated, required)

    def compute_aggregator_utility(self, allocated: float, required: float) -> float:
        """Compute aggregator compute utility"""
        return self.aggregator_compute_utility.compute_utility(allocated, required)

    def compute_communication_utility(self, allocated: float, required: float) -> float:
        """Compute communication utility"""
        return self.communication_utility.compute_utility(allocated, required)

    def compute_policy_performance(self, agg_utility: float, comm_utility: float) -> float:
        """
        Compute policy performance metric
        P_task = k_p * RV * RF_total
        """
        return self.policy_scaling * agg_utility * comm_utility


class LatencyBasedQoSModel(IQoSModel):
    """
    QoS model for latency-critical tasks
    """

    def __init__(self, lambda_policy: float = 0.4, lambda_res_u: float = 0.2,
                 lambda_res_v: float = 0.2, lambda_res_f: float = 0.2):
        """
        Initialize QoS model with resource weights

        Args:
            lambda_policy: Weight for policy performance
            lambda_res_u: Weight for edge compute utility
            lambda_res_v: Weight for aggregator compute utility
            lambda_res_f: Weight for communication utility
        """
        self.lambda_policy = lambda_policy
        self.lambda_res_u = lambda_res_u
        self.lambda_res_v = lambda_res_v
        self.lambda_res_f = lambda_res_f

    def compute_qos(self, task: ITask, allocation: ResourceAllocation, 
                   utilities: Dict[str, float]) -> float:
        """
        Compute achieved QoS
        QoS_act = QoS_base + improvements from resource allocation
        """
        qos_base = 10.0  # Baseline QoS

        policy_util = utilities.get('policy', 0.0)
        edge_util = utilities.get('edge', 0.0)
        agg_util = utilities.get('aggregator', 0.0)
        comm_util = utilities.get('communication', 0.0)

        improvement = (self.lambda_policy * policy_util +
                      self.lambda_res_u * edge_util +
                      self.lambda_res_v * agg_util +
                      self.lambda_res_f * comm_util)

        return qos_base + improvement

    def compute_violation(self, task: ITask, achieved_qos: float) -> float:
        """
        Compute QoS violation penalty
        For latency-critical tasks, violation is based on latency budget
        """
        requirements = task.get_requirements()

        if requirements.latency_max is not None:
            # For latency-critical tasks, violation is latency excess
            return max(0.0, achieved_qos - requirements.latency_max)

        # For other tasks, no violation
        return 0.0


class ThroughputQoSModel(IQoSModel):
    """QoS model for throughput-oriented tasks"""

    def __init__(self, min_throughput: float = 1.0):
        self.min_throughput = min_throughput

    def compute_qos(self, task: ITask, allocation: ResourceAllocation, 
                   utilities: Dict[str, float]) -> float:
        """Compute achieved throughput as QoS metric"""
        total_bandwidth = allocation.get_total_bandwidth()
        return total_bandwidth

    def compute_violation(self, task: ITask, achieved_qos: float) -> float:
        """Compute violation if throughput below minimum"""
        return max(0.0, self.min_throughput - achieved_qos)


class PerformanceEstimatorEMA:
    """
    Exponential Moving Average Performance Estimator
    E^{k+1} = (1-α)E^k + αM^k
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA estimator

        Args:
            alpha: Smoothing parameter (0 < alpha < 1)
        """
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be in (0, 1)")

        self.alpha = alpha
        self._estimates: Dict[str, float] = {}
        self._history: Dict[str, list] = {}

    def initialize(self, key: str, initial_value: float) -> None:
        self._estimates[key] = initial_value
        self._history[key] = [initial_value]

    def update(self, key: str, measured_value: float) -> None:
        if key not in self._estimates:
            self._estimates[key] = measured_value
        else:
            self._estimates[key] = ((1 - self.alpha) * self._estimates[key] +
                                   self.alpha * measured_value)

        if key not in self._history:
            self._history[key] = []
        self._history[key].append(self._estimates[key])

    def get_estimate(self, key: str, default: float = 0.0) -> float:
        return self._estimates.get(key, default)

    def get_all_estimates(self) -> Dict[str, float]:
        return self._estimates.copy()

    def get_history(self, key: str) -> list:
        """Get estimation history for a metric"""
        return self._history.get(key, [])

    def reset(self):
        """Reset all estimates"""
        self._estimates.clear()
        self._history.clear()


class CostModel:
    """Cost model for compute and communication resources"""

    def __init__(self, cost_compute_ric: float = 0.5, 
                 cost_compute_du: float = 0.4):
        """
        Initialize cost model

        Args:
            cost_compute_ric: Cost per TOPS at RIC (μ/TOPS)
            cost_compute_du: Cost per TOPS at DU (μ/TOPS)
        """
        self.cost_compute_ric = cost_compute_ric
        self.cost_compute_du = cost_compute_du

    def compute_cost(self, allocation: ResourceAllocation,
                    link_costs: Dict[str, float]) -> Dict[str, float]:
        """
        Compute total cost for an allocation

        Returns:
            Dictionary with cost breakdown
        """
        # Compute costs
        compute_cost_ric = self.cost_compute_ric * allocation.compute_aggregator
        compute_cost_du = sum(self.cost_compute_du * alloc 
                             for alloc in allocation.compute_edge.values())

        # Communication costs
        comm_cost = sum(allocation.bandwidth.get(link_id, 0) * cost
                       for link_id, cost in link_costs.items())

        return {
            'compute_ric': compute_cost_ric,
            'compute_du': compute_cost_du,
            'communication': comm_cost,
            'total': compute_cost_ric + compute_cost_du + comm_cost
        }


if __name__ == "__main__":
    # Test utility functions
    print("Testing Utility Functions:\n")

    utilities = ResourceUtilities(k_u=1.0, k_v=1.0, k_f=1.0, k_p=1.0)

    edge_util = utilities.compute_edge_utility(allocated=10.0, required=5.0)
    print(f"Edge utility (10/5): {edge_util:.4f}")

    agg_util = utilities.compute_aggregator_utility(allocated=15.0, required=10.0)
    print(f"Aggregator utility (15/10): {agg_util:.4f}")

    comm_util = utilities.compute_communication_utility(allocated=5.0, required=500.0)
    print(f"Communication utility (5/500): {comm_util:.4f}")

    policy_perf = utilities.compute_policy_performance(agg_util, comm_util)
    print(f"Policy performance: {policy_perf:.4f}")

    # Test EMA estimator
    print("\nTesting EMA Estimator:\n")
    estimator = PerformanceEstimatorEMA(alpha=0.3)
    estimator.initialize('reward', 100.0)

    for i, measurement in enumerate([110, 115, 120, 118, 122], 1):
        estimator.update('reward', measurement)
        print(f"Iteration {i}: Measured={measurement}, Estimate={estimator.get_estimate('reward'):.2f}")
