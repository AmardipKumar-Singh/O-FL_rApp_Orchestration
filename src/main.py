#!/usr/bin/env python3
"""
main.py - Simulation-Based Entry Point
O-FL rApp: Distributed Orchestration Framework
"""

import numpy as np
from typing import List

from models.task import TaskBuilder, TaskType
from models.network import TopologyBuilder
from orchestrator import OFLrApp, OrchestrationConfig
from baselines.baseline_algorithms import compare_baselines
from config import SystemConfig


def create_sample_topology():
    """Create a sample O-RAN topology"""
    print("Creating O-RAN network topology...")

    topology = (TopologyBuilder()
        # Edge nodes (O-DUs)
        .add_odu('ODU_1', capacity=80.0)
        .add_odu('ODU_2', capacity=80.0)
        .add_odu('ODU_3', capacity=80.0)

        # Aggregation (Near-RT-RIC)
        .add_ric('RIC', capacity=150.0)

        # Communication links
        .add_fiber_link('ODU_1', 'RIC', bandwidth=10.0, latency=2.0, cost=1.0)
        .add_fiber_link('ODU_2', 'RIC', bandwidth=10.0, latency=2.0, cost=1.0)
        .add_fiber_link('ODU_3', 'RIC', bandwidth=8.0, latency=3.0, cost=0.8)

        # Agent assignments
        .assign_agent('agent_1', 'ODU_1')
        .assign_agent('agent_2', 'ODU_1')
        .assign_agent('agent_3', 'ODU_2')
        .assign_agent('agent_4', 'ODU_2')
        .assign_agent('agent_5', 'ODU_3')
        .assign_agent('agent_6', 'ODU_3')

        .build())

    print(f"  ✓ {len(topology.get_nodes())} nodes")
    print(f"  ✓ {len(topology.get_links())} links")
    print(f"  ✓ 6 agents\n")

    return topology


def create_heterogeneous_tasks() -> List:
    """Create heterogeneous FL/MARL tasks"""
    print("Creating heterogeneous tasks...")

    # Task 1: Latency-Critical (uRLLC) - Mobility Management
    task1 = (TaskBuilder()
        .with_id('T_Mobility')
        .with_type(TaskType.LATENCY_CRITICAL)
        .with_agents(['agent_1', 'agent_2'])
        .with_compute_requirements(agent=8.0, aggregator=15.0)
        .with_data_transfer(10.0)  # Small model
        .with_latency_budget(5.0)  # Strict latency
        .with_priority(2.0)  # High priority
        .with_reward(100.0)
        .build())

    # Task 2: Throughput-Oriented (eMBB) - Traffic Prediction
    task2 = (TaskBuilder()
        .with_id('T_Traffic')
        .with_type(TaskType.THROUGHPUT_ORIENTED)
        .with_agents(['agent_3', 'agent_4'])
        .with_compute_requirements(agent=10.0, aggregator=20.0)
        .with_data_transfer(50.0)  # Large model
        .with_priority(1.0)  # Medium priority
        .with_reward(80.0)
        .build())

    # Task 3: Mixed - Load Balancing
    task3 = (TaskBuilder()
        .with_id('T_LoadBalance')
        .with_type(TaskType.MIXED)
        .with_agents(['agent_5', 'agent_6'])
        .with_compute_requirements(agent=6.0, aggregator=12.0)
        .with_data_transfer(20.0)
        .with_latency_budget(10.0)
        .with_priority(1.5)
        .with_reward(90.0)
        .build())

    print(f"  ✓ Task 1: {task1.get_id()} (uRLLC) - 2 agents")
    print(f"  ✓ Task 2: {task2.get_id()} (eMBB) - 2 agents")
    print(f"  ✓ Task 3: {task3.get_id()} (Mixed) - 2 agents\n")

    return [task1, task2, task3]


def run_proposed_algorithm(tasks, topology):
    """Run the proposed O-FL rApp algorithm"""
    print(f"\n{'='*70}")
    print("PROPOSED ALGORITHM: O-FL rApp with TSA + RAR")
    print(f"{'='*70}\n")

    # Configure orchestrator
    config = OrchestrationConfig(
        max_iterations=50,
        convergence_threshold=1e-4,
        ema_alpha=0.3,
        w_reward=1.0,
        w_qos=1000.0,
        tsa_priority_weight=2.0,
        rar_use_gurobi=True
    )

    # Create and run orchestrator
    orchestrator = OFLrApp(config)
    orchestrator.initialize(tasks, topology)

    print("Running orchestration...")
    metrics_history = orchestrator.run_until_convergence()

    # Analyze results
    print(f"\n{'='*70}")
    print("RESULTS: Proposed O-FL rApp")
    print(f"{'='*70}\n")

    if metrics_history:
        # Overall metrics
        total_reward = sum(m.global_reward for m in metrics_history)
        total_qos_vio = sum(m.qos_violation for m in metrics_history)
        total_cost = sum(m.resource_cost for m in metrics_history)
        success_rate = np.mean([m.success for m in metrics_history])

        print(f"Total Iterations: {len(metrics_history) // len(tasks)}")
        print(f"Total Global Reward: {total_reward:.2f}")
        print(f"Total QoS Violations: {total_qos_vio:.2f}")
        print(f"Total Resource Cost: {total_cost:.2f}")
        print(f"Success Rate: {success_rate:.1%}\n")

        # Per-task metrics
        print("Per-Task Performance:")
        for task in tasks:
            task_metrics = [m for m in metrics_history if m.task_id == task.get_id()]
            if task_metrics:
                avg_reward = np.mean([m.global_reward for m in task_metrics])
                avg_qos = np.mean([m.qos_violation for m in task_metrics])
                avg_latency = np.mean([m.latency for m in task_metrics])
                print(f"  {task.get_id()}:")
                print(f"    Avg Reward: {avg_reward:.2f}")
                print(f"    Avg QoS Violation: {avg_qos:.2f}")
                print(f"    Avg Latency: {avg_latency:.2f} ms")

        return {
            'avg_reward': total_reward / len(metrics_history),
            'avg_qos_violation': total_qos_vio / len(metrics_history),
            'avg_cost': total_cost / len(metrics_history),
            'success_rate': success_rate,
            'convergence_iterations': len(metrics_history) // len(tasks)
        }

    return None


def run_baseline_comparison(tasks, topology):
    """Run and compare all baseline algorithms"""
    print(f"\n{'='*70}")
    print("BASELINE ALGORITHMS COMPARISON")
    print(f"{'='*70}\n")

    # Import tasks for each baseline
    from copy import deepcopy

    baseline_results = compare_baselines(
        [deepcopy(t) for t in tasks],
        topology,
        iterations=50
    )

    return baseline_results


def compare_all_algorithms(tasks, topology):
    """Compare proposed algorithm with all baselines"""
    print(f"\n{'='*70}")
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print(f"{'='*70}\n")

    # Run proposed algorithm
    proposed_results = run_proposed_algorithm(tasks, topology)

    # Run baselines
    baseline_results = run_baseline_comparison(tasks, topology)

    # Print comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}\n")

    print(f"{'Algorithm':<25} {'Avg Reward':>12} {'QoS Vio':>10} {'Cost':>10} {'Success':>10}")
    print("-" * 70)

    # Proposed
    if proposed_results:
        print(f"{'O-FL rApp (Proposed)':<25} "
              f"{proposed_results['avg_reward']:>12.2f} "
              f"{proposed_results['avg_qos_violation']:>10.2f} "
              f"{proposed_results['avg_cost']:>10.2f} "
              f"{proposed_results['success_rate']:>9.1%}")

    # Baselines
    for name, results in baseline_results.items():
        display_name = name.replace('_', ' ')
        print(f"{display_name:<25} "
              f"{results['avg_reward']:>12.2f} "
              f"{results['avg_qos_violation']:>10.2f} "
              f"{results['avg_cost']:>10.2f} "
              f"{results['success_rate']:>9.1%}")

    print("\n")

    # Calculate improvements
    if proposed_results and baseline_results:
        best_baseline_reward = max(r['avg_reward'] for r in baseline_results.values())
        improvement = (proposed_results['avg_reward'] - best_baseline_reward) / best_baseline_reward * 100
        print(f"Improvement over best baseline: {improvement:+.1f}%\n")


def run_scalability_experiment():
    """Test scalability with increasing number of tasks"""
    print(f"\n{'='*70}")
    print("SCALABILITY EXPERIMENT")
    print(f"{'='*70}\n")

    topology = create_sample_topology()

    for num_tasks in [2, 3, 5, 8]:
        print(f"\nTesting with {num_tasks} tasks...")

        # Create tasks
        tasks = []
        for i in range(num_tasks):
            task_type = TaskType.LATENCY_CRITICAL if i % 2 == 0 else TaskType.THROUGHPUT_ORIENTED
            agents = [f'agent_{i*2+1}', f'agent_{i*2+2}']

            task = (TaskBuilder()
                .with_id(f'T_{i+1}')
                .with_type(task_type)
                .with_agents(agents[:2])  # Max 2 agents per task
                .with_compute_requirements(agent=5.0, aggregator=10.0)
                .with_data_transfer(20.0)
                .with_priority(float(num_tasks - i))
                .with_reward(100.0)
                .build())

            tasks.append(task)

        # Run orchestration
        config = OrchestrationConfig(max_iterations=20)
        orchestrator = OFLrApp(config)
        orchestrator.initialize(tasks, topology)
        metrics = orchestrator.run_until_convergence()

        # Report
        if metrics:
            avg_reward = np.mean([m.global_reward for m in metrics])
            success_rate = np.mean([m.success for m in metrics])
            print(f"  Results: Avg Reward = {avg_reward:.2f}, Success Rate = {success_rate:.1%}")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("O-FL rApp: Distributed Orchestration Framework")
    print("Simulation-Based Experiments (No Real Training)")
    print("="*70 + "\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create topology and tasks
    topology = create_sample_topology()
    tasks = create_heterogeneous_tasks()

    # Option 1: Run proposed algorithm only
    print("\n[1] Running Proposed Algorithm...")
    proposed_results = run_proposed_algorithm(tasks, topology)

    # Option 2: Run baselines only
    # print("\n[2] Running Baseline Comparison...")
    # baseline_results = run_baseline_comparison(tasks, topology)

    # Option 3: Compare all algorithms
    # print("\n[3] Comprehensive Comparison...")
    # compare_all_algorithms(tasks, topology)

    # Option 4: Scalability experiment
    # print("\n[4] Scalability Experiment...")
    # run_scalability_experiment()

    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}\n")
    print("To run with real FL training: python main_with_datasets.py")
    print("To run with MARL training: python main_with_marl.py\n")


if __name__ == "__main__":
    main()
