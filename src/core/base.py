#!/usr/bin/env python3
"""
core/base.py - Abstract Base Classes and Interfaces
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks

Defines the core abstractions for modular, extensible architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """Enumeration of task types"""
    THROUGHPUT_ORIENTED = "throughput"
    LATENCY_CRITICAL = "latency_critical"
    MIXED = "mixed"
    CUSTOM = "custom"


class TaskState(Enum):
    """Task lifecycle states"""
    PENDING = "pending"
    ACTIVE = "active"
    PREEMPTED = "preempted"
    TERMINATED = "terminated"


@dataclass
class ResourceRequirements:
    """Resource requirements for a task"""
    compute_agent: float  # TOPS per agent
    compute_aggregator: float  # TOPS for aggregator
    data_transfer: float  # Mbits per agent per period
    bandwidth_min: Optional[float] = None  # Minimum bandwidth requirement
    latency_max: Optional[float] = None  # Maximum latency constraint


@dataclass
class ResourceAllocation:
    """Resource allocation decision variables"""
    task_id: str
    compute_edge: Dict[str, float] = field(default_factory=dict)  # node -> allocation
    compute_aggregator: float = 0.0
    bandwidth: Dict[str, float] = field(default_factory=dict)  # link -> allocation
    routing: Dict[str, str] = field(default_factory=dict)  # agent -> link

    def get_total_bandwidth(self) -> float:
        """Get total allocated bandwidth"""
        return sum(self.bandwidth.values())


@dataclass
class PerformanceMetrics:
    """Performance metrics for a task"""
    task_id: str
    global_reward: float = 0.0
    qos_violation: float = 0.0
    resource_cost: float = 0.0
    latency: Optional[float] = None
    convergence_rate: Optional[float] = None
    success: bool = False


class ITask(ABC):
    """Interface for Federated MARL tasks"""

    @abstractmethod
    def get_id(self) -> str:
        """Get task identifier"""
        pass

    @abstractmethod
    def get_type(self) -> TaskType:
        """Get task type"""
        pass

    @abstractmethod
    def get_agents(self) -> List[str]:
        """Get list of agent IDs for this task"""
        pass

    @abstractmethod
    def get_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        pass

    @abstractmethod
    def get_priority(self) -> float:
        """Get task priority (higher = more important)"""
        pass

    @abstractmethod
    def get_state(self) -> TaskState:
        """Get current task state"""
        pass

    @abstractmethod
    def set_state(self, state: TaskState) -> None:
        """Set task state"""
        pass

    @abstractmethod
    def compute_reward(self, allocation: ResourceAllocation, 
                      actual_performance: Dict) -> float:
        """Compute global reward given allocation and performance"""
        pass


class INetworkTopology(ABC):
    """Interface for network topology"""

    @abstractmethod
    def get_nodes(self) -> List[str]:
        """Get list of network nodes"""
        pass

    @abstractmethod
    def get_links(self) -> Dict[str, Dict]:
        """Get network links with properties"""
        pass

    @abstractmethod
    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """Get physical location of agent"""
        pass

    @abstractmethod
    def get_path_properties(self, src: str, dst: str, 
                          link_type: str) -> Optional[Dict]:
        """Get properties of a path (bandwidth, latency, cost)"""
        pass

    @abstractmethod
    def check_capacity_constraint(self, link_id: str, 
                                  allocated: float) -> bool:
        """Check if allocation satisfies link capacity constraint"""
        pass


class IUtilityFunction(ABC):
    """Interface for utility functions"""

    @abstractmethod
    def compute_utility(self, allocated_resource: float, 
                       required_resource: float) -> float:
        """Compute utility for resource allocation"""
        pass

    @abstractmethod
    def get_derivative(self, allocated_resource: float, 
                      required_resource: float) -> float:
        """Compute derivative of utility function"""
        pass


class IQoSModel(ABC):
    """Interface for QoS models"""

    @abstractmethod
    def compute_qos(self, task: ITask, allocation: ResourceAllocation, 
                   utilities: Dict[str, float]) -> float:
        """Compute achieved QoS"""
        pass

    @abstractmethod
    def compute_violation(self, task: ITask, achieved_qos: float) -> float:
        """Compute QoS violation penalty"""
        pass


class IAssignmentSolver(ABC):
    """Interface for task assignment solvers"""

    @abstractmethod
    def solve(self, tasks: List[ITask], topology: INetworkTopology,
             performance_estimates: Dict) -> Dict[str, any]:
        """
        Solve task assignment problem

        Returns:
            Dictionary with assignment decisions
        """
        pass

    @abstractmethod
    def validate_assignment(self, assignment: Dict, 
                          tasks: List[ITask]) -> bool:
        """Validate assignment satisfies constraints"""
        pass


class IResourceSolver(ABC):
    """Interface for resource allocation solvers"""

    @abstractmethod
    def solve(self, tasks: List[ITask], assignment: Dict,
             topology: INetworkTopology, 
             performance_estimates: Dict) -> Dict[str, ResourceAllocation]:
        """
        Solve resource allocation problem

        Returns:
            Dictionary mapping task_id to ResourceAllocation
        """
        pass

    @abstractmethod
    def validate_allocation(self, allocations: Dict[str, ResourceAllocation],
                          topology: INetworkTopology) -> bool:
        """Validate allocation satisfies resource constraints"""
        pass


class IPerformanceEstimator(ABC):
    """Interface for performance estimation"""

    @abstractmethod
    def initialize(self, key: str, initial_value: float) -> None:
        """Initialize estimate for a metric"""
        pass

    @abstractmethod
    def update(self, key: str, measured_value: float) -> None:
        """Update estimate with new measurement"""
        pass

    @abstractmethod
    def get_estimate(self, key: str, default: float = 0.0) -> float:
        """Get current estimate"""
        pass

    @abstractmethod
    def get_all_estimates(self) -> Dict[str, float]:
        """Get all estimates"""
        pass


class IOrchestrator(ABC):
    """Interface for orchestration coordinator"""

    @abstractmethod
    def initialize(self, tasks: List[ITask], topology: INetworkTopology) -> None:
        """Initialize orchestrator with tasks and topology"""
        pass

    @abstractmethod
    def run_iteration(self) -> PerformanceMetrics:
        """Run one orchestration iteration"""
        pass

    @abstractmethod
    def check_convergence(self) -> bool:
        """Check if orchestration has converged"""
        pass

    @abstractmethod
    def get_current_allocation(self) -> Dict[str, ResourceAllocation]:
        """Get current resource allocation"""
        pass


class IBaselineAlgorithm(ABC):
    """Interface for baseline algorithms"""

    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name"""
        pass

    @abstractmethod
    def allocate_resources(self, tasks: List[ITask], 
                          topology: INetworkTopology) -> Dict[str, ResourceAllocation]:
        """Allocate resources for tasks"""
        pass


# Factory pattern for creating components
class ComponentFactory:
    """Factory for creating system components"""

    _task_registry: Dict[str, type] = {}
    _solver_registry: Dict[str, type] = {}
    _baseline_registry: Dict[str, type] = {}

    @classmethod
    def register_task_type(cls, task_type: str, task_class: type):
        """Register a task implementation"""
        cls._task_registry[task_type] = task_class

    @classmethod
    def create_task(cls, task_type: str, **kwargs) -> ITask:
        """Create a task instance"""
        if task_type not in cls._task_registry:
            raise ValueError(f"Unknown task type: {task_type}")
        return cls._task_registry[task_type](**kwargs)

    @classmethod
    def register_solver(cls, solver_name: str, solver_class: type):
        """Register a solver implementation"""
        cls._solver_registry[solver_name] = solver_class

    @classmethod
    def create_solver(cls, solver_name: str, **kwargs):
        """Create a solver instance"""
        if solver_name not in cls._solver_registry:
            raise ValueError(f"Unknown solver: {solver_name}")
        return cls._solver_registry[solver_name](**kwargs)

    @classmethod
    def register_baseline(cls, baseline_name: str, baseline_class: type):
        """Register a baseline algorithm"""
        cls._baseline_registry[baseline_name] = baseline_class

    @classmethod
    def create_baseline(cls, baseline_name: str, **kwargs) -> IBaselineAlgorithm:
        """Create a baseline algorithm instance"""
        if baseline_name not in cls._baseline_registry:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        return cls._baseline_registry[baseline_name](**kwargs)


if __name__ == "__main__":
    print("Core base classes and interfaces defined")
    print(f"Task types: {[t.value for t in TaskType]}")
    print(f"Task states: {[s.value for s in TaskState]}")
