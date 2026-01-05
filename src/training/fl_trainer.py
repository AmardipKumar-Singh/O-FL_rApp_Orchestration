#!/usr/bin/env python3
"""
training/fl_trainer.py - Federated Learning Trainer for O-RAN Tasks
O-FL rApp: Model Training Integration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from data.base_dataset import DataPartition, DataLoader
from core.base import ITask, ResourceAllocation


@dataclass
class TrainingConfig:
    """Configuration for FL training"""
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_method: str = 'fedavg'  # 'fedavg', 'fedprox'
    convergence_threshold: float = 1e-3


class SimpleNeuralNetwork:
    """Simple neural network for classification tasks"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass with cache for backprop"""
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._softmax(Z2)

        cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2, cache

    def backward(self, cache: Dict, y: np.ndarray, 
                learning_rate: float) -> float:
        """Backward pass and parameter update"""
        m = y.shape[0]

        # Output layer gradients
        dZ2 = cache['A2'] - y
        dW2 = (cache['A1'].T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache['Z1'] > 0)  # ReLU derivative
        dW1 = (cache['X'].T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        # Compute loss
        loss = -np.mean(np.sum(y * np.log(cache['A2'] + 1e-8), axis=1))
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output, _ = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate accuracy and loss"""
        output, cache = self.forward(X)
        predictions = np.argmax(output, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
        return accuracy, loss

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters"""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set model parameters"""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


class FederatedLearningTrainer:
    """
    Federated Learning Trainer for O-RAN Tasks

    Integrates with O-FL rApp orchestrator to train models
    on distributed network data
    """

    def __init__(self, input_dim: int, output_dim: int, 
                 config: Optional[TrainingConfig] = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or TrainingConfig()

        # Global model
        hidden_dim = max(32, (input_dim + output_dim) // 2)
        self.global_model = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)

        # Local models per agent
        self.local_models: Dict[str, SimpleNeuralNetwork] = {}

        # Training history
        self.history = {
            'global_loss': [],
            'global_accuracy': [],
            'local_losses': {},
            'communication_rounds': 0
        }

    def train_federated_round(self, partitions: Dict[str, DataPartition],
                             allocation: ResourceAllocation,
                             task: ITask) -> Dict:
        """
        Execute one federated learning round

        Args:
            partitions: Data partitions for each agent
            allocation: Resource allocation from RAR solver
            task: Task being trained

        Returns:
            Training metrics for this round
        """
        self.history['communication_rounds'] += 1

        # Phase 1: Local training at each agent
        local_updates = {}
        local_metrics = {}

        for agent_id, partition in partitions.items():
            if agent_id not in self.local_models:
                hidden_dim = max(32, (self.input_dim + self.output_dim) // 2)
                self.local_models[agent_id] = SimpleNeuralNetwork(
                    self.input_dim, hidden_dim, self.output_dim
                )

            # Download global model
            self.local_models[agent_id].set_parameters(
                self.global_model.get_parameters()
            )

            # Compute available training epochs based on compute allocation
            compute_alloc = allocation.compute_edge.get(
                partition.node_id, 5.0
            )
            adjusted_epochs = self._compute_to_epochs(compute_alloc)

            # Local training
            metrics = self._train_local_model(
                self.local_models[agent_id],
                partition,
                adjusted_epochs
            )

            local_updates[agent_id] = self.local_models[agent_id].get_parameters()
            local_metrics[agent_id] = metrics

            if agent_id not in self.history['local_losses']:
                self.history['local_losses'][agent_id] = []
            self.history['local_losses'][agent_id].append(metrics['loss'])

        # Phase 2: Aggregation at Near-RT-RIC
        aggregation_start = time.time()

        # Compute aggregation based on allocated bandwidth
        bandwidth_weights = self._compute_aggregation_weights(
            allocation, list(local_updates.keys())
        )

        aggregated_params = self._aggregate_parameters(
            local_updates, bandwidth_weights
        )

        aggregation_time = time.time() - aggregation_start

        # Update global model
        self.global_model.set_parameters(aggregated_params)

        # Phase 3: Evaluate global model
        all_samples = []
        for partition in partitions.values():
            all_samples.extend(partition.samples)

        if all_samples:
            X = np.array([s.features for s in all_samples])
            y = np.array([s.label for s in all_samples])
            global_acc, global_loss = self.global_model.evaluate(X, y)
        else:
            global_acc, global_loss = 0.0, float('inf')

        self.history['global_accuracy'].append(global_acc)
        self.history['global_loss'].append(global_loss)

        # Compute reward based on performance
        reward = self._compute_training_reward(
            global_acc, global_loss, allocation
        )

        return {
            'global_accuracy': global_acc,
            'global_loss': global_loss,
            'local_metrics': local_metrics,
            'aggregation_time': aggregation_time,
            'bandwidth_used': sum(allocation.bandwidth.values()),
            'compute_used': sum(allocation.compute_edge.values()),
            'reward': reward,
            'communication_round': self.history['communication_rounds']
        }

    def _train_local_model(self, model: SimpleNeuralNetwork,
                          partition: DataPartition,
                          epochs: int) -> Dict:
        """Train local model on agent's data"""
        data_loader = DataLoader(partition, self.config.batch_size, shuffle=True)

        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for X_batch, y_batch in data_loader:
                _, cache = model.forward(X_batch)
                loss = model.backward(cache, y_batch, self.config.learning_rate)
                epoch_losses.append(loss)
            losses.append(np.mean(epoch_losses))

        # Evaluate on local data
        X = partition.get_features()
        y = partition.get_labels()
        accuracy, final_loss = model.evaluate(X, y)

        return {
            'loss': final_loss,
            'accuracy': accuracy,
            'epochs': epochs,
            'samples': len(partition)
        }

    def _compute_to_epochs(self, compute_allocation: float) -> int:
        """Convert compute allocation to training epochs"""
        # More compute = more epochs (linear relationship)
        base_epochs = self.config.local_epochs
        scaling_factor = compute_allocation / 10.0  # Normalize by typical allocation
        adjusted_epochs = int(base_epochs * min(2.0, max(0.5, scaling_factor)))
        return max(1, adjusted_epochs)

    def _compute_aggregation_weights(self, allocation: ResourceAllocation,
                                    agent_ids: List[str]) -> Dict[str, float]:
        """Compute aggregation weights based on bandwidth allocation"""
        # Weight by allocated bandwidth (better communication = higher weight)
        weights = {}
        total_bandwidth = 0.0

        for agent_id in agent_ids:
            # Get bandwidth for agent's routing
            agent_bandwidth = sum(
                bw for link_id, bw in allocation.bandwidth.items()
                if agent_id in allocation.routing.values()
            )
            weights[agent_id] = max(0.1, agent_bandwidth)
            total_bandwidth += weights[agent_id]

        # Normalize weights
        if total_bandwidth > 0:
            weights = {k: v / total_bandwidth for k, v in weights.items()}
        else:
            # Equal weights if no bandwidth info
            equal_weight = 1.0 / len(agent_ids)
            weights = {k: equal_weight for k in agent_ids}

        return weights

    def _aggregate_parameters(self, local_updates: Dict[str, Dict],
                             weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        """FedAvg aggregation"""
        aggregated = {}

        # Get parameter names from first model
        param_names = list(next(iter(local_updates.values())).keys())

        for param_name in param_names:
            weighted_sum = None
            for agent_id, params in local_updates.items():
                weight = weights.get(agent_id, 1.0 / len(local_updates))

                if weighted_sum is None:
                    weighted_sum = weight * params[param_name]
                else:
                    weighted_sum += weight * params[param_name]

            aggregated[param_name] = weighted_sum

        return aggregated

    def _compute_training_reward(self, accuracy: float, loss: float,
                                allocation: ResourceAllocation) -> float:
        """
        Compute reward for training performance

        Reward = accuracy - loss_penalty - resource_cost
        """
        # Accuracy component (0-100 scale)
        accuracy_reward = accuracy * 100.0

        # Loss penalty
        loss_penalty = min(50.0, loss * 10.0)

        # Resource cost
        compute_cost = sum(allocation.compute_edge.values()) * 0.1
        comm_cost = sum(allocation.bandwidth.values()) * 0.05
        resource_cost = compute_cost + comm_cost

        reward = accuracy_reward - loss_penalty - resource_cost
        return max(0.0, reward)

    def get_training_summary(self) -> Dict:
        """Get summary of training progress"""
        return {
            'communication_rounds': self.history['communication_rounds'],
            'final_accuracy': self.history['global_accuracy'][-1] if self.history['global_accuracy'] else 0.0,
            'final_loss': self.history['global_loss'][-1] if self.history['global_loss'] else float('inf'),
            'best_accuracy': max(self.history['global_accuracy']) if self.history['global_accuracy'] else 0.0,
            'convergence': self._check_convergence(),
            'num_agents': len(self.local_models)
        }

    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.history['global_loss']) < 3:
            return False

        recent_losses = self.history['global_loss'][-3:]
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        return loss_change < self.config.convergence_threshold


if __name__ == "__main__":
    print("Federated Learning Trainer for O-RAN")
    print("Supports: Network Traffic QoS, Cell Load Balancing")
