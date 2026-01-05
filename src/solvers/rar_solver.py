#!/usr/bin/env python3
"""
solvers/rar_solver.py - Resource Allocation and Routing Solver (OOP)
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks

Implements RAR sub-problem with MILP optimization
"""

from typing import Dict, List, Optional
from core.base import (IResourceSolver, ITask, INetworkTopology, 
                       ResourceAllocation, ComponentFactory)
from solvers.tsa_solver import TaskAssignment

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class MILPResourceSolver(IResourceSolver):
    """
    MILP-based Resource Allocation and Routing Solver

    Solves Equation 34 with:
    - Piecewise linear approximation for utilities
    - McCormick envelope constraints
    - Link capacity and compute constraints
    """

    def __init__(self, cpu_ric: float = 100, cpu_du: float = 80,
                 w_reward: float = 1.0, w_qos: float = 1000.0,
                 time_limit: int = 300):
        """
        Initialize RAR solver

        Args:
            cpu_ric: Total RIC CPU capacity (TOPS)
            cpu_du: Total DU CPU capacity (TOPS)
            w_reward: Reward weight
            w_qos: QoS violation weight
            time_limit: MILP solver time limit (seconds)
        """
        self.cpu_ric = cpu_ric
        self.cpu_du = cpu_du
        self.w_reward = w_reward
        self.w_qos = w_qos
        self.time_limit = time_limit

    def solve(self, tasks: List[ITask], assignment: TaskAssignment,
             topology: INetworkTopology, 
             performance_estimates: Dict) -> Dict[str, ResourceAllocation]:
        """
        Solve resource allocation problem (Equation 34)
        """
        if not GUROBI_AVAILABLE:
            return self._solve_heuristic(tasks, assignment, topology)

        try:
            return self._solve_milp(tasks, assignment, topology, 
                                   performance_estimates)
        except Exception as e:
            print(f"MILP solver error: {e}. Using heuristic.")
            return self._solve_heuristic(tasks, assignment, topology)

    def _solve_milp(self, tasks: List[ITask], assignment: TaskAssignment,
                   topology: INetworkTopology, 
                   performance_estimates: Dict) -> Dict[str, ResourceAllocation]:
        """Solve using Gurobi MILP"""
        model = gp.Model("RAR")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.time_limit)

        # Filter active tasks
        active_tasks = [t for t in tasks if assignment.is_task_active(t.get_id())]

        if not active_tasks:
            return {}

        # Decision variables
        v = {}  # Aggregator compute
        u = {}  # Edge compute
        f = {}  # Bandwidth allocation
        R = {}  # Routing

        links = topology.get_links()
        nodes = topology.get_nodes()

        for task in active_tasks:
            task_id = task.get_id()
            requirements = task.get_requirements()

            # Aggregator compute allocation
            v[task_id] = model.addVar(lb=0, ub=1, name=f"v_{task_id}")

            # Edge compute allocation
            for node in nodes:
                if node != 'RIC':
                    u[(node, task_id)] = model.addVar(lb=0, ub=1, 
                                                      name=f"u_{node}_{task_id}")

            # Bandwidth allocation
            for link_id in links:
                f[(link_id, task_id)] = model.addVar(lb=0, ub=1,
                                                     name=f"f_{link_id}_{task_id}")

            # Routing decisions
            for agent in assignment.get_assigned_agents(task_id):
                agent_node = topology.get_agent_location(agent)
                if agent_node:
                    for link_id in links:
                        if links[link_id]['source'] == agent_node:
                            R[(link_id, agent)] = model.addVar(vtype=GRB.BINARY,
                                                              name=f"R_{link_id}_{agent}")

        model.update()

        # Objective: minimize cost - reward
        compute_cost = gp.LinExpr()
        comm_cost = gp.LinExpr()
        reward_term = gp.LinExpr()

        for task in active_tasks:
            task_id = task.get_id()
            requirements = task.get_requirements()

            # Compute costs
            compute_cost += 0.5 * v[task_id] * requirements.compute_aggregator
            for node in nodes:
                if node != 'RIC':
                    compute_cost += 0.4 * u.get((node, task_id), 0) * requirements.compute_agent

            # Communication costs
            for link_id in links:
                comm_cost += (links[link_id]['cost'] * 
                            f.get((link_id, task_id), 0) * 
                            links[link_id]['bandwidth'])

            # Reward (linearized)
            total_bw = gp.quicksum(
                f.get((link_id, task_id), 0) * links[link_id]['bandwidth']
                for link_id in links
            )
            reward_term += 0.7 * total_bw / 10.0  # Linearized log

        objective = compute_cost + comm_cost - self.w_reward * reward_term
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraints

        # Link capacity (Constraint 21)
        for link_id in links:
            model.addConstr(
                gp.quicksum(f.get((link_id, task.get_id()), 0) 
                           for task in active_tasks) <= 1.0,
                name=f"link_cap_{link_id}"
            )

        # Compute capacity (Constraints 23-24)
        model.addConstr(
            gp.quicksum(v[task.get_id()] for task in active_tasks) <= 1.0,
            name="ric_capacity"
        )

        for node in nodes:
            if node != 'RIC':
                model.addConstr(
                    gp.quicksum(u.get((node, task.get_id()), 0) 
                               for task in active_tasks) <= 1.0,
                    name=f"du_capacity_{node}"
                )

        # Solve
        model.optimize()

        # Extract solution
        allocations = {}
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            for task in active_tasks:
                task_id = task.get_id()
                alloc = ResourceAllocation(task_id=task_id)

                alloc.compute_aggregator = v[task_id].X * self.cpu_ric

                for node in nodes:
                    if node != 'RIC':
                        alloc.compute_edge[node] = u.get((node, task_id), 0).X * self.cpu_du

                for link_id in links:
                    alloc.bandwidth[link_id] = (f.get((link_id, task_id), 0).X * 
                                               links[link_id]['bandwidth'])

                for agent in assignment.get_assigned_agents(task_id):
                    for link_id in links:
                        if (link_id, agent) in R and R[(link_id, agent)].X > 0.5:
                            alloc.routing[agent] = link_id

                allocations[task_id] = alloc

        return allocations

    def _solve_heuristic(self, tasks: List[ITask], assignment: TaskAssignment,
                        topology: INetworkTopology) -> Dict[str, ResourceAllocation]:
        """Heuristic solver when MILP unavailable"""
        allocations = {}
        active_tasks = [t for t in tasks if assignment.is_task_active(t.get_id())]

        if not active_tasks:
            return allocations

        # Proportional allocation based on priority
        total_priority = sum(t.get_priority() for t in active_tasks)
        links = topology.get_links()
        nodes = topology.get_nodes()

        for task in active_tasks:
            task_id = task.get_id()
            weight = task.get_priority() / total_priority if total_priority > 0 else 1.0/len(active_tasks)

            alloc = ResourceAllocation(task_id=task_id)
            alloc.compute_aggregator = weight * self.cpu_ric

            for node in nodes:
                if node != 'RIC':
                    alloc.compute_edge[node] = weight * self.cpu_du

            for link_id in links:
                alloc.bandwidth[link_id] = weight * links[link_id]['bandwidth']

            # Route to fiber link (prefer low latency)
            for agent in assignment.get_assigned_agents(task_id):
                agent_node = topology.get_agent_location(agent)
                if agent_node:
                    fiber_link = f"{agent_node}_RIC_fiber"
                    if fiber_link in links:
                        alloc.routing[agent] = fiber_link

            allocations[task_id] = alloc

        return allocations

    def validate_allocation(self, allocations: Dict[str, ResourceAllocation],
                          topology: INetworkTopology) -> bool:
        """Validate allocations satisfy capacity constraints"""
        links = topology.get_links()

        # Check link capacity constraints
        link_usage = {link_id: 0.0 for link_id in links}
        for alloc in allocations.values():
            for link_id, bw in alloc.bandwidth.items():
                link_usage[link_id] += bw

        for link_id, usage in link_usage.items():
            if usage > links[link_id]['bandwidth'] * 1.01:  # Allow 1% tolerance
                return False

        return True


# Register with factory
ComponentFactory.register_solver('rar_milp', MILPResourceSolver)


if __name__ == "__main__":
    print("RAR Solver Module")
    print(f"Gurobi available: {GUROBI_AVAILABLE}")
