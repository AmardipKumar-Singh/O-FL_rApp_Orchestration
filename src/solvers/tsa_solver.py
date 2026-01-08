#!/usr/bin/env python3
"""
solvers/tsa_solver.py - Task and Slice Assignment Solver
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks

Algorithm 1: Task and Slice Assignment
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from core.base import (IAssignmentSolver, ITask, INetworkTopology, 
                       TaskState, ComponentFactory)


@dataclass
class TaskAssignment:
    """Result of task assignment"""
    task_agent_map: Dict[Tuple[str, str], bool] = field(default_factory=dict)  # (task, agent) -> assigned
    task_slice_map: Dict[Tuple[str, str, str], bool] = field(default_factory=dict)  # (task, node, slice)
    task_active: Dict[str, bool] = field(default_factory=dict)  # task -> active

    def is_task_active(self, task_id: str) -> bool:
        return self.task_active.get(task_id, False)

    def get_assigned_agents(self, task_id: str) -> List[str]:
        return [agent for (tid, agent), assigned in self.task_agent_map.items()
                if tid == task_id and assigned]


class AssignmentScorer:
    """Computes assignment scores for task-agent pairs"""

    def __init__(self, w_reward: float = 1.0, w_qos: float = 1000.0):
        self.w_reward = w_reward
        self.w_qos = w_qos

    def compute_score(self, task: ITask, agent: str,
                     topology: INetworkTopology,
                     performance_estimates: Dict) -> float:
        """
        Compute assignment score
        AScore = E[R_global] - E[TotalCost] - W_QoS * E[QoS_vio]
        """
        task_id = task.get_id()

        # Expected reward
        E_reward = performance_estimates.get(f'reward_{task_id}', 100.0)

        # Expected cost
        requirements = task.get_requirements()
        agent_node = topology.get_agent_location(agent)

        if not agent_node:
            return -float('inf')

        # Compute cost
        compute_cost = (requirements.compute_agent * 0.4 +  # DU cost
                       requirements.compute_aggregator * 0.5)  # RIC cost

        # Communication cost - choose minimum cost path
        data_gbits = requirements.data_transfer / 1000.0

        fiber_props = topology.get_path_properties(agent_node, 'RIC', 'fiber')
        microwave_props = topology.get_path_properties(agent_node, 'RIC', 'microwave')

        comm_costs = []
        if fiber_props:
            comm_costs.append(data_gbits * fiber_props['cost'])
        if microwave_props:
            comm_costs.append(data_gbits * microwave_props['cost'])

        comm_cost = min(comm_costs) if comm_costs else float('inf')
        E_total_cost = compute_cost + comm_cost

        # Expected QoS violation
        E_qos_vio = performance_estimates.get(f'qos_vio_{task_id}', 0.0)

        # Assignment score
        score = E_reward - E_total_cost - self.w_qos * E_qos_vio

        return score


class GraphAwareTSA(IAssignmentSolver):
    

    def __init__(self, w_reward: float = 1.0, w_qos: float = 1000.0):
        """
        Initialize TSA solver

        Args:
            w_reward: Weight for reward in objective
            w_qos: Weight for QoS violations
        """
        self.scorer = AssignmentScorer(w_reward, w_qos)
        self._agent_available: Dict[str, bool] = {}

    def solve(self, tasks: List[ITask], topology: INetworkTopology,
             performance_estimates: Dict) -> TaskAssignment:
        """
        Solve task assignment problem (Algorithm 1)

        Phase 1: Topology-Aware Task Prioritization
        Phase 2: Iterative Assignment with Communication Awareness

        Args:
            tasks: List of tasks
            topology: Network topology
            performance_estimates: Performance estimates

        Returns:
            TaskAssignment object
        """
        assignment = TaskAssignment()

        # Initialize agent availability
        self._agent_available = {agent: True 
                                for task in tasks 
                                for agent in task.get_agents()}

        # Phase 1: Compute priority scores
        task_priorities = []
        for task in tasks:
            priority = self._compute_priority_score(task, topology, 
                                                    performance_estimates)
            task_priorities.append((priority, task))

        # Sort by descending priority
        task_priorities.sort(key=lambda x: x[0], reverse=True)

        # Phase 2: Iterative assignment
        for priority, task in task_priorities:
            best_score = -float('inf')
            best_agent = None
            best_node = None

            # Find best agent for this task
            for agent in task.get_agents():
                if not self._agent_available.get(agent, True):
                    continue

                score = self.scorer.compute_score(task, agent, topology,
                                                 performance_estimates)

                if score > best_score:
                    best_score = score
                    best_agent = agent
                    best_node = topology.get_agent_location(agent)

            # Make assignment if beneficial
            if best_score > 0 and best_agent and best_node:
                task_id = task.get_id()
                assignment.task_agent_map[(task_id, best_agent)] = True
                assignment.task_slice_map[(task_id, best_node, 'default')] = True
                assignment.task_active[task_id] = True
                self._agent_available[best_agent] = False
                task.set_state(TaskState.ACTIVE)
            else:
                assignment.task_active[task.get_id()] = False
                task.set_state(TaskState.PENDING)

        return assignment

    def _compute_priority_score(self, task: ITask, topology: INetworkTopology,
                               performance_estimates: Dict) -> float:
        """
        Compute priority score for task
        PriorityScore = max over agents of assignment score
        """
        max_score = -float('inf')

        for agent in task.get_agents():
            if not self._agent_available.get(agent, True):
                continue

            score = self.scorer.compute_score(task, agent, topology,
                                             performance_estimates)
            max_score = max(max_score, score)

        return max_score if max_score != -float('inf') else 0.0

    def validate_assignment(self, assignment: TaskAssignment, 
                          tasks: List[ITask]) -> bool:
        """
        Validate assignment satisfies constraints
        """
        # Check Constraint: agent uniqueness
        agent_counts: Dict[str, int] = {}
        for (task_id, agent), assigned in assignment.task_agent_map.items():
            if assigned:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
                if agent_counts[agent] > 1:
                    return False

        # Check Constraint: active tasks have agents
        for task in tasks:
            task_id = task.get_id()
            has_agents = any(assignment.task_agent_map.get((task_id, a), False)
                           for a in task.get_agents())
            is_active = assignment.task_active.get(task_id, False)

            if is_active and not has_agents:
                return False

        return True


# Register with factory
ComponentFactory.register_solver('tsa', GraphAwareTSA)


if __name__ == "__main__":
    print("TSA Solver Module")
    print("Implements Algorithm 1: Graph-Aware Task and Slice Assignment")
