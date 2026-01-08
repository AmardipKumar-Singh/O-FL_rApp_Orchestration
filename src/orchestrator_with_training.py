#!/usr/bin/env python3
"""
orchestrator_with_training.py - O-FL rApp with Real FL Training
Extends the base orchestrator to include actual federated learning training
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from orchestrator import OFLrAppOrchestrator, OrchestrationConfig
from core.base import ITask, INetworkTopology, PerformanceMetrics, TaskState
from training.fl_trainer import FederatedLearningTrainer, TrainingConfig
from data.base_dataset import ORANDataset


@dataclass
class IntegratedConfig:
    """Configuration for integrated orchestrator"""
    orchestration: OrchestrationConfig
    training: TrainingConfig
    enable_real_training: bool = True


class IntegratedOFLrApp(OFLrAppOrchestrator):

    def __init__(self, config: IntegratedConfig):
        super().__init__(config.orchestration)
        self.training_config = config.training
        self.enable_real_training = config.enable_real_training

        # FL trainers per task
        self.trainers: Dict[str, FederatedLearningTrainer] = {}

        # Datasets per task
        self.datasets: Dict[str, ORANDataset] = {}

    def register_dataset(self, task_id: str, dataset: ORANDataset) -> None:
        """
        Register dataset for a task

        Args:
            task_id: Task identifier
            dataset: O-RAN dataset (loaded and partitioned)
        """
        self.datasets[task_id] = dataset

        # Create FL trainer for this task
        if task_id not in self.trainers:
            # Get dimensions from dataset
            sample = dataset._data[0] if dataset._data else None
            if sample:
                input_dim = len(sample.features)
                output_dim = len(sample.label)

                self.trainers[task_id] = FederatedLearningTrainer(
                    input_dim, output_dim, self.training_config
                )

                print(f"Registered dataset '{dataset.name}' for task {task_id}")
                print(f"  Input dim: {input_dim}, Output dim: {output_dim}")

    def _execute_fl_tasks(self) -> Dict[str, PerformanceMetrics]:
        """
        Execute FL tasks with real training

        Overrides base method to include actual model training
        """
        task_metrics = {}

        for task in self.tasks:
            if task.get_state() != TaskState.ACTIVE:
                continue

            task_id = task.get_id()
            allocation = self.current_allocations.get(task_id)

            if not allocation:
                continue

            # Execute training if enabled and dataset available
            if self.enable_real_training and task_id in self.datasets:
                metrics = self._execute_real_fl_training(task, allocation)
            else:
                # Fall back to simulation
                metrics = self._execute_simulated_fl(task, allocation)

            task_metrics[task_id] = metrics

        return task_metrics

    def _execute_real_fl_training(self, task: ITask,
                                  allocation) -> PerformanceMetrics:
        """Execute real federated learning training"""
        task_id = task.get_id()
        dataset = self.datasets[task_id]
        trainer = self.trainers[task_id]

        # Get assigned agents from current assignment
        assigned_agents = self.current_assignment.get_assigned_agents(task_id)

        # Get data partitions for assigned agents
        all_partitions = dataset.get_all_partitions()
        active_partitions = {
            agent: all_partitions[agent]
            for agent in assigned_agents
            if agent in all_partitions
        }

        if not active_partitions:
            # No partitions available, fall back to simulation
            return self._execute_simulated_fl(task, allocation)

        # Train one federated round
        training_results = trainer.train_federated_round(
            active_partitions, allocation, task
        )

        # Compute actual performance from training
        actual_performance = {
            'latency': self._compute_communication_latency(allocation) + 
                      self._compute_training_latency(allocation),
            'qos_violation': self._compute_qos_violation(task, training_results),
            'success': training_results['global_accuracy'] > 0.5,
            'accuracy': training_results['global_accuracy'],
            'loss': training_results['global_loss']
        }

        # Compute reward based on training performance
        reward = training_results['reward']

        # Compute costs
        links = self.topology.get_links()
        link_costs = {lid: props['cost'] * props['bandwidth'] 
                     for lid, props in links.items()}
        costs = self.cost_model.compute_cost(allocation, link_costs)

        # Create metrics
        metrics = PerformanceMetrics(
            task_id=task_id,
            global_reward=max(0, reward),
            qos_violation=actual_performance['qos_violation'],
            resource_cost=costs['total'],
            latency=actual_performance['latency'],
            success=actual_performance['success']
        )

        return metrics

    def _execute_simulated_fl(self, task: ITask, allocation) -> PerformanceMetrics:
    
        actual_performance = self._simulate_task_execution(task, allocation)

        # Compute reward
        reward = task.compute_reward(allocation, actual_performance)

        # Add noise if configured
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

        return metrics

    def _compute_training_latency(self, allocation) -> float:
        """Compute latency from training computation"""
        if allocation.compute_aggregator > 1e-6:
            # Training time increases with model complexity
            return 10.0 / allocation.compute_aggregator
        return 50.0

    def _compute_qos_violation(self, task: ITask, 
                              training_results: Dict) -> float:
        """Compute QoS violation for training task"""
        requirements = task.get_requirements()

        if requirements.latency_max is not None:
            total_latency = (training_results['aggregation_time'] + 
                           self._compute_communication_latency(
                               self.current_allocations.get(task.get_id())
                           ))
            violation = max(0, total_latency - requirements.latency_max)
            return violation

        # For accuracy-based QoS
        target_accuracy = 0.7  # Target 70% accuracy
        accuracy_gap = max(0, target_accuracy - training_results['global_accuracy'])
        return accuracy_gap * 10.0  # Scale to match latency violations

    def get_training_summary(self) -> Dict[str, Dict]:
        """Get training summary for all tasks"""
        summaries = {}
        for task_id, trainer in self.trainers.items():
            summaries[task_id] = trainer.get_training_summary()
        return summaries

    def run_until_convergence(self) -> List[PerformanceMetrics]:
        """Run with training summary at the end"""
        print(f"\n{'='*70}")
        print("Integrated O-FL rApp with Real FL Training")
        print(f"{'='*70}")
        print(f"Tasks: {[t.get_id() for t in self.tasks]}")
        print(f"Datasets: {list(self.datasets.keys())}")
        print(f"Real training: {self.enable_real_training}")
        print(f"Max iterations: {self.config.max_iterations}\n")

        # Run base orchestration
        metrics_history = super().run_until_convergence()

        # Print training summaries
        if self.enable_real_training:
            print(f"\n{'='*70}")
            print("TRAINING SUMMARY")
            print(f"{'='*70}\n")

            for task_id, summary in self.get_training_summary().items():
                print(f"Task {task_id}:")
                print(f"  Communication Rounds: {summary['communication_rounds']}")
                print(f"  Final Accuracy: {summary['final_accuracy']:.4f}")
                print(f"  Best Accuracy: {summary['best_accuracy']:.4f}")
                print(f"  Final Loss: {summary['final_loss']:.4f}")
                print(f"  Converged: {summary['convergence']}")
                print(f"  Agents: {summary['num_agents']}\n")

        return metrics_history


if __name__ == "__main__":
    print("Integrated O-FL rApp Orchestrator")
    print("Combines optimization with real FL training")
