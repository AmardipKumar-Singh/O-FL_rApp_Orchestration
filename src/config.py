#!/usr/bin/env python3
"""
config.py - System Configuration Parameters
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks
"""

import numpy as np

class SystemConfig:
    """Configuration for O-RAN network topology and resources"""

    # Network Topology
    NUM_GNBS = 2  # gNB-A, gNB-B
    NUM_ODUS = 2  # O-DUs per gNB
    NUM_SLICES = 3  # Network slices

    # CPU Capacities (TOPS)
    CPU_TOTAL_RIC = 100  # Near-RT-RIC capacity
    CPU_MAX_DU = 80  # Per O-DU capacity

    # Communication Links
    LINKS = {
        'fiber': {
            'bandwidth': 10.0,  # Gbps
            'latency': 0.5,      # ms
            'cost': 10.0         # μ/Gbps
        },
        'microwave': {
            'bandwidth': 1.0,    # Gbps
            'latency': 5.0,      # ms
            'cost': 2.0          # μ/Gbps
        }
    }

    # Compute Costs (μ/TOPS)
    COST_COMPUTE_RIC = 0.5
    COST_COMPUTE_DU = 0.4

    # Control Parameters
    CONTROL_PERIOD = 100  # ms
    MAX_ITERATIONS = 50
    CONVERGENCE_THRESHOLD = 1e-4

    # Objective Weights
    W_REWARD = 1.0
    W_QOS = 1000.0

    # EMA Parameters
    EMA_ALPHA = 0.3  # Smoothing parameter for feedback loop

    # PWL Approximation
    NUM_BREAKPOINTS = 10  # For piecewise linear approximation

    # Utility Function Parameters
    K_U = 1.0  # Edge compute utility scaling
    K_V = 1.0  # Aggregator compute utility scaling
    K_F = 1.0  # Communication utility scaling
    K_P = 1.0  # Policy performance scaling


class TaskConfig:
    """Configuration for Federated MARL tasks"""

    TASKS = {
        'T1': {  # Throughput-Oriented (eMBB)
            'name': 'eMBB_Task',
            'type': 'throughput',
            'agents': ['a1', 'a2'],
            'data_transfer': 500,  # Mbits per agent per period
            'compute_agent': 5.0,  # TOPS per agent
            'compute_xapp': 10.0,  # TOPS for aggregator
            'reward_params': {'base': 200, 'type': 'log'},
            'qos_requirement': None,  # Best-effort
            'latency_budget': None,
            'priority': 1.0
        },
        'T2': {  # Latency-Critical (uRLLC)
            'name': 'uRLLC_Task',
            'type': 'latency_critical',
            'agents': ['a3', 'a4'],
            'data_transfer': 10,  # Mbits per agent per period
            'compute_agent': 8.0,  # TOPS per agent
            'compute_xapp': 15.0,  # TOPS for aggregator
            'reward_params': {'base': 50, 'type': 'fixed'},
            'qos_requirement': 'latency',
            'latency_budget': 6.0,  # ms
            'priority': 2.0
        },
        'T3': {  # Additional task for dynamic scenarios
            'name': 'MixedTask',
            'type': 'mixed',
            'agents': ['a5', 'a6'],
            'data_transfer': 100,  # Mbits per agent per period
            'compute_agent': 6.0,  # TOPS per agent
            'compute_xapp': 12.0,  # TOPS for aggregator
            'reward_params': {'base': 150, 'type': 'log'},
            'qos_requirement': None,
            'latency_budget': None,
            'priority': 1.5
        }
    }

    # Agent to Node Mapping (O_{r,a})
    AGENT_NODE_MAPPING = {
        'a1': 'ODU_1',
        'a2': 'ODU_2',
        'a3': 'ODU_1',
        'a4': 'ODU_2',
        'a5': 'ODU_1',
        'a6': 'ODU_2'
    }

    # Agent to Slice Mapping (for network slicing)
    AGENT_SLICE_MAPPING = {
        'a1': 'slice_1',
        'a2': 'slice_1',
        'a3': 'slice_2',
        'a4': 'slice_2',
        'a5': 'slice_3',
        'a6': 'slice_3'
    }


if __name__ == "__main__":
    # Test configuration
    sys_config = SystemConfig()
    task_config = TaskConfig()

    print("System Configuration:")
    print(f"  RIC CPU: {sys_config.CPU_TOTAL_RIC} TOPS")
    print(f"  DU CPU: {sys_config.CPU_MAX_DU} TOPS")
    print(f"  Fiber: {sys_config.LINKS['fiber']['bandwidth']} Gbps, {sys_config.LINKS['fiber']['latency']} ms")
    print(f"  Microwave: {sys_config.LINKS['microwave']['bandwidth']} Gbps, {sys_config.LINKS['microwave']['latency']} ms")

    print("\nTask Configuration:")
    for task_id, task_data in task_config.TASKS.items():
        print(f"  {task_id}: {task_data['name']} - {len(task_data['agents'])} agents, {task_data['data_transfer']} Mbits/agent")
