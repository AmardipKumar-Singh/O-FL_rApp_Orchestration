#!/usr/bin/env python3
"""
main_with_datasets.py - Complete Example with O-RAN Datasets
O-FL rApp: Network Traffic + Cell Load Data

Demonstrates:
1. Network Traffic QoS Prediction (uRLLC task)
2. Cell Load Balancing (eMBB task)
3. Real federated learning training
4. Resource orchestration with actual ML performance
"""

import numpy as np
from typing import Dict

# Core imports
from models.task import TaskBuilder, TaskType
from models.network import TopologyBuilder

# Dataset imports
from data.network_traffic_dataset import NetworkTrafficDataset, NetworkTrafficConfig
from data.cell_load_dataset import CellLoadDataset, CellLoadConfig

# Training imports
from orchestrator_with_training import IntegratedOFLrApp, IntegratedConfig
from orchestrator import OrchestrationConfig
from training.fl_trainer import TrainingConfig


def run_network_traffic_qos_scenario():
    """
    Scenario 1: Network Traffic QoS Prediction

    Task: Predict QoS class for incoming traffic flows
    Type: Latency-critical (uRLLC)
    Dataset: Network traffic traces with throughput, latency, jitter
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Network Traffic QoS Prediction (uRLLC)")
    print("="*70 + "\n")

    # 1. Create and load dataset
    print("Step 1: Loading Network Traffic Dataset...")
    traffic_config = NetworkTrafficConfig(
        num_samples=5000,
        num_odus=2,
        noise_level=0.1
    )
    traffic_dataset = NetworkTrafficDataset(traffic_config)
    traffic_dataset.load_data()

    # Print dataset statistics
    stats = traffic_dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Feature dimension: {stats['feature_dim']}")
    print(f"  QoS classes: {stats['num_classes']}")
    print(f"  QoS Distribution:")
    for qos_class, percentage in stats['qos_percentages'].items():
        print(f"    {qos_class}: {percentage:.1f}%")

    # 2. Create agents and partition data
    print("\nStep 2: Partitioning data across agents...")
    agents = ['agent_1', 'agent_2', 'agent_3', 'agent_4']
    partitions = traffic_dataset.partition_data(agents, strategy='non-iid')

    print(f"  Partitioning strategy: non-iid (different traffic patterns)")
    for agent_id, partition in partitions.items():
        print(f"    {agent_id}: {len(partition)} samples at {partition.node_id}")

    # 3. Build O-RAN topology
    print("\nStep 3: Building O-RAN topology...")
    topology = (TopologyBuilder()
        .add_odu('ODU_1', capacity=80.0)
        .add_odu('ODU_2', capacity=80.0)
        .add_ric('RIC', capacity=100.0)
        .add_fiber_link('ODU_1', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_2', 'RIC', bandwidth=10.0)
        .add_microwave_link('ODU_1', 'RIC', bandwidth=1.0)
        .add_microwave_link('ODU_2', 'RIC', bandwidth=1.0)
        .assign_agent('agent_1', 'ODU_1')
        .assign_agent('agent_2', 'ODU_1')
        .assign_agent('agent_3', 'ODU_2')
        .assign_agent('agent_4', 'ODU_2')
        .build())

    print(f"  Topology: 2 O-DUs, 1 RIC, 4 agents")

    # 4. Create latency-critical task
    print("\nStep 4: Creating latency-critical task...")
    task_qos = (TaskBuilder()
        .with_id('T_QoS')
        .with_type(TaskType.LATENCY_CRITICAL)
        .with_agents(agents)
        .with_compute_requirements(agent=8.0, aggregator=15.0)
        .with_data_transfer(10.0)  # Low data transfer for uRLLC
        .with_latency_budget(6.0)  # 6ms latency budget
        .with_priority(2.0)  # High priority
        .with_reward(100.0)
        .build())

    print(f"  Task: {task_qos.get_id()}")
    print(f"  Type: {task_qos.get_type().value}")
    print(f"  Latency budget: {task_qos.get_requirements().latency_max} ms")

    # 5. Configure and initialize orchestrator
    print("\nStep 5: Initializing integrated orchestrator...")
    config = IntegratedConfig(
        orchestration=OrchestrationConfig(
            max_iterations=20,
            convergence_threshold=1e-4,
            ema_alpha=0.3,
            w_reward=1.0,
            w_qos=1000.0  # High penalty for QoS violations
        ),
        training=TrainingConfig(
            local_epochs=5,
            batch_size=32,
            learning_rate=0.01
        ),
        enable_real_training=True
    )

    orchestrator = IntegratedOFLrApp(config)
    orchestrator.register_dataset('T_QoS', traffic_dataset)
    orchestrator.initialize([task_qos], topology)

    # 6. Run orchestration
    print("\nStep 6: Running orchestration with FL training...")
    print("-" * 70)
    metrics_history = orchestrator.run_until_convergence()

    return metrics_history


def run_cell_load_balancing_scenario():
    """
    Scenario 2: Cell Load Balancing

    Task: Predict load balancing actions for cell management
    Type: Throughput-oriented (eMBB)
    Dataset: Cell load data with PRB utilization, handovers, etc.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Cell Load Balancing (eMBB)")
    print("="*70 + "\n")

    # 1. Create and load dataset
    print("Step 1: Loading Cell Load Dataset...")
    cell_config = CellLoadConfig(
        num_samples=5000,
        num_cells=7,
        max_users_per_cell=100
    )
    cell_dataset = CellLoadDataset(cell_config)
    cell_dataset.load_data()

    # Print dataset statistics
    stats = cell_dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Feature dimension: {stats['feature_dim']}")
    print(f"  Action classes: {stats['num_classes']}")
    print(f"  Action Distribution:")
    for action, percentage in stats['action_percentages'].items():
        print(f"    {action}: {percentage:.1f}%")

    # 2. Create agents and partition data
    print("\nStep 2: Partitioning data across cells...")
    agents = ['agent_1', 'agent_2', 'agent_3']
    partitions = cell_dataset.partition_data(agents, strategy='cell-based')

    print(f"  Partitioning strategy: cell-based (geographic)")
    for agent_id, partition in partitions.items():
        print(f"    {agent_id}: {len(partition)} samples")

    # 3. Build O-RAN topology
    print("\nStep 3: Building O-RAN topology...")
    topology = (TopologyBuilder()
        .add_odu('ODU_1', capacity=80.0)
        .add_odu('ODU_2', capacity=80.0)
        .add_ric('RIC', capacity=100.0)
        .add_fiber_link('ODU_1', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_2', 'RIC', bandwidth=10.0)
        .add_microwave_link('ODU_1', 'RIC', bandwidth=1.0)
        .add_microwave_link('ODU_2', 'RIC', bandwidth=1.0)
        .assign_agent('agent_1', 'ODU_1')
        .assign_agent('agent_2', 'ODU_1')
        .assign_agent('agent_3', 'ODU_2')
        .build())

    # 4. Create throughput-oriented task
    print("\nStep 4: Creating throughput-oriented task...")
    task_load = (TaskBuilder()
        .with_id('T_Load')
        .with_type(TaskType.THROUGHPUT_ORIENTED)
        .with_agents(agents)
        .with_compute_requirements(agent=5.0, aggregator=10.0)
        .with_data_transfer(500.0)  # High data transfer for eMBB
        .with_priority(1.0)
        .with_reward(200.0)
        .build())

    print(f"  Task: {task_load.get_id()}")
    print(f"  Type: {task_load.get_type().value}")

    # 5. Configure and initialize orchestrator
    print("\nStep 5: Initializing integrated orchestrator...")
    config = IntegratedConfig(
        orchestration=OrchestrationConfig(
            max_iterations=20,
            convergence_threshold=1e-4,
            ema_alpha=0.3,
            w_reward=1.0,
            w_qos=100.0
        ),
        training=TrainingConfig(
            local_epochs=3,
            batch_size=64,
            learning_rate=0.02
        ),
        enable_real_training=True
    )

    orchestrator = IntegratedOFLrApp(config)
    orchestrator.register_dataset('T_Load', cell_dataset)
    orchestrator.initialize([task_load], topology)

    # 6. Run orchestration
    print("\nStep 6: Running orchestration with FL training...")
    print("-" * 70)
    metrics_history = orchestrator.run_until_convergence()

    return metrics_history


def run_combined_scenario():
    """
    Scenario 3: Combined Multi-Task

    Both tasks running concurrently with resource competition
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Combined Multi-Task (QoS + Load Balancing)")
    print("="*70 + "\n")

    # Load both datasets
    print("Step 1: Loading both datasets...")

    traffic_config = NetworkTrafficConfig(num_samples=3000)
    traffic_dataset = NetworkTrafficDataset(traffic_config)
    traffic_dataset.load_data()

    cell_config = CellLoadConfig(num_samples=3000)
    cell_dataset = CellLoadDataset(cell_config)
    cell_dataset.load_data()

    print(f"  ✓ Network Traffic: {len(traffic_dataset)} samples")
    print(f"  ✓ Cell Load: {len(cell_dataset)} samples")

    # Partition data
    print("\nStep 2: Partitioning data...")
    qos_agents = ['agent_1', 'agent_2', 'agent_3', 'agent_4']
    load_agents = ['agent_5', 'agent_6', 'agent_7']

    traffic_partitions = traffic_dataset.partition_data(qos_agents, strategy='iid')
    cell_partitions = cell_dataset.partition_data(load_agents, strategy='cell-based')

    print(f"  QoS task: {len(qos_agents)} agents")
    print(f"  Load task: {len(load_agents)} agents")

    # Build larger topology
    print("\nStep 3: Building extended topology...")
    topology = (TopologyBuilder()
        .add_odu('ODU_1', capacity=100.0)
        .add_odu('ODU_2', capacity=100.0)
        .add_odu('ODU_3', capacity=100.0)
        .add_ric('RIC', capacity=150.0)
        .add_fiber_link('ODU_1', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_2', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_3', 'RIC', bandwidth=10.0)
        .add_microwave_link('ODU_1', 'RIC', bandwidth=1.0)
        .add_microwave_link('ODU_2', 'RIC', bandwidth=1.0)
        .add_microwave_link('ODU_3', 'RIC', bandwidth=1.0))

    for i, agent in enumerate(qos_agents):
        topology.assign_agent(agent, f'ODU_{i % 2 + 1}')
    for i, agent in enumerate(load_agents):
        topology.assign_agent(agent, f'ODU_{i % 3 + 1}')

    topology = topology.build()

    # Create both tasks
    print("\nStep 4: Creating tasks...")
    task_qos = (TaskBuilder()
        .with_id('T_QoS')
        .with_type(TaskType.LATENCY_CRITICAL)
        .with_agents(qos_agents)
        .with_compute_requirements(agent=8.0, aggregator=15.0)
        .with_data_transfer(10.0)
        .with_latency_budget(6.0)
        .with_priority(2.0)
        .with_reward(100.0)
        .build())

    task_load = (TaskBuilder()
        .with_id('T_Load')
        .with_type(TaskType.THROUGHPUT_ORIENTED)
        .with_agents(load_agents)
        .with_compute_requirements(agent=5.0, aggregator=10.0)
        .with_data_transfer(500.0)
        .with_priority(1.0)
        .with_reward(200.0)
        .build())

    print(f"  ✓ {task_qos.get_id()}: {task_qos.get_type().value}")
    print(f"  ✓ {task_load.get_id()}: {task_load.get_type().value}")

    # Configure orchestrator
    print("\nStep 5: Initializing orchestrator for multi-task...")
    config = IntegratedConfig(
        orchestration=OrchestrationConfig(
            max_iterations=25,
            convergence_threshold=1e-4,
            ema_alpha=0.3,
            w_reward=1.0,
            w_qos=1000.0
        ),
        training=TrainingConfig(
            local_epochs=4,
            batch_size=32,
            learning_rate=0.015
        ),
        enable_real_training=True
    )

    orchestrator = IntegratedOFLrApp(config)
    orchestrator.register_dataset('T_QoS', traffic_dataset)
    orchestrator.register_dataset('T_Load', cell_dataset)
    orchestrator.initialize([task_qos, task_load], topology)

    # Run orchestration
    print("\nStep 6: Running multi-task orchestration...")
    print("-" * 70)
    metrics_history = orchestrator.run_until_convergence()

    return metrics_history


def main():
    """Run all scenarios"""
    print("\n" + "="*70)
    print("O-FL rApp: Integrated FL Training with O-RAN Datasets")
    print("="*70)
    print("\nScenarios:")
    print("  1. Network Traffic QoS Prediction (uRLLC)")
    print("  2. Cell Load Balancing (eMBB)")
    print("  3. Combined Multi-Task")
    print("\n" + "="*70)

    # Run scenarios
    try:
        metrics1 = run_network_traffic_qos_scenario()
        metrics2 = run_cell_load_balancing_scenario()
        metrics3 = run_combined_scenario()

        print("\n" + "="*70)
        print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
        print("="*70)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
