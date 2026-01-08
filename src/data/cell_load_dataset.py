#!/usr/bin/env python3
"""
data/cell_load_dataset.py - Cell Load Data for Load Balancing
O-FL rApp: O-RAN Dataset
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from data.base_dataset import ORANDataset, DataSample, DataPartition


@dataclass
class CellLoadConfig:
    """Configuration for cell load dataset"""
    num_samples: int = 10000
    num_cells: int = 7  # Hexagonal layout
    max_users_per_cell: int = 100
    prb_capacity: int = 100  # Physical Resource Blocks
    noise_level: float = 0.1

    # Cell characteristics
    cell_radius: float = 500.0  # meters
    interference_factor: float = 0.3
    handover_threshold: float = 0.75  # Load threshold for handover

    # Traffic dynamics
    mobility_rate: float = 0.05  # User mobility rate
    arrival_rate: float = 10.0  # New users per time unit
    departure_rate: float = 8.0


class CellLoadDataset(ORANDataset):
    """
    Cell Load Dataset for Load Balancing

    Features (per cell):
    - Active users count
    - PRB (Physical Resource Block) utilization (%)
    - Average throughput per user (Mbps)
    - Handover rate (users/time)
    - Cell edge users ratio
    - Inter-cell interference (dBm)
    - Buffer occupancy (%)
    - Neighboring cell loads (average)
    - User distribution (spatial)
    - Historical load trend

    Labels:
    - Action: 0 (No action), 1 (Offload users), 2 (Accept handover), 3 (Increase capacity)
    """

    def __init__(self, config: Optional[CellLoadConfig] = None):
        super().__init__("CellLoad")
        self.config = config or CellLoadConfig()
        self.feature_dim = 10
        self.num_classes = 4  # Load balancing actions
        self._feature_names = [
            'active_users', 'prb_utilization', 'avg_throughput',
            'handover_rate', 'cell_edge_ratio', 'interference',
            'buffer_occupancy', 'neighbor_load', 'user_distribution',
            'load_trend'
        ]

        # Cell topology (hexagonal)
        self._cell_neighbors = self._create_cell_topology()

    def _create_cell_topology(self) -> Dict[int, List[int]]:
        """Create hexagonal cell topology"""
        # Center cell (0) with 6 neighbors (1-6)
        neighbors = {
            0: [1, 2, 3, 4, 5, 6],  # Center
            1: [0, 2, 6],
            2: [0, 1, 3],
            3: [0, 2, 4],
            4: [0, 3, 5],
            5: [0, 4, 6],
            6: [0, 5, 1]
        }
        return neighbors

    def load_data(self, data_path: Optional[str] = None) -> None:
        """
        Load or generate cell load data

        Args:
            data_path: Path to real data file (if None, generates synthetic)
        """
        if data_path and self._load_from_file(data_path):
            self._loaded = True
            return

        print(f"Generating synthetic cell load data ({self.config.num_samples} samples)...")
        self._data = self._generate_synthetic_data()
        self._loaded = True
        print(f"✓ Generated {len(self._data)} cell load samples")

    def _generate_synthetic_data(self) -> List[DataSample]:
        """Generate realistic synthetic cell load data"""
        samples = []

        # Initialize cell states
        cell_loads = {cell_id: 0.5 for cell_id in range(self.config.num_cells)}
        cell_users = {cell_id: self.config.max_users_per_cell // 2 
                     for cell_id in range(self.config.num_cells)}

        for i in range(self.config.num_samples):
            # Randomly select a cell to generate sample for
            cell_id = i % self.config.num_cells

            # Update cell state with dynamics
            prev_load = cell_loads[cell_id]

            # User arrivals and departures
            arrivals = np.random.poisson(self.config.arrival_rate * 0.1)
            departures = np.random.poisson(self.config.departure_rate * 0.1)
            cell_users[cell_id] = max(0, min(self.config.max_users_per_cell,
                                            cell_users[cell_id] + arrivals - departures))

            # Active users (normalized)
            active_users = cell_users[cell_id]
            active_users_norm = active_users / self.config.max_users_per_cell

            # PRB utilization (increases with user count)
            base_prb = (active_users / self.config.max_users_per_cell) * 100
            prb_utilization = base_prb + np.random.normal(0, 5 * self.config.noise_level)
            prb_utilization = np.clip(prb_utilization, 0, 100)

            # Average throughput (decreases with load)
            congestion_factor = 1.0 - (prb_utilization / 150.0)
            avg_throughput = 50.0 * max(0.1, congestion_factor) + np.random.normal(0, 5)
            avg_throughput = max(1.0, avg_throughput)

            # Handover rate (increases with high load)
            if prb_utilization > 70:
                handover_rate = 0.5 + 0.3 * (prb_utilization - 70) / 30 + np.random.exponential(0.1)
            else:
                handover_rate = np.random.exponential(0.2)
            handover_rate = min(2.0, handover_rate)

            # Cell edge users ratio (users with poor signal)
            cell_edge_ratio = 0.2 + 0.3 * (prb_utilization / 100) + np.random.uniform(-0.1, 0.1)
            cell_edge_ratio = np.clip(cell_edge_ratio, 0, 1)

            # Inter-cell interference
            neighbor_loads = [cell_loads[n] for n in self._cell_neighbors[cell_id]]
            avg_neighbor_load = np.mean(neighbor_loads)
            interference = -90.0 + 20 * avg_neighbor_load * self.config.interference_factor
            interference += np.random.normal(0, 2)

            # Buffer occupancy
            buffer_occupancy = min(100, prb_utilization * 1.2 + np.random.uniform(0, 10))

            # User spatial distribution (uniformity measure)
            user_distribution = np.random.beta(2, 2)  # Tends towards uniform

            # Historical load trend (change from previous)
            load_trend = (prb_utilization / 100) - prev_load
            cell_loads[cell_id] = prb_utilization / 100

            # Features vector
            features = np.array([
                active_users_norm,
                prb_utilization / 100.0,
                avg_throughput / 100.0,
                handover_rate / 2.0,
                cell_edge_ratio,
                (interference + 110) / 40.0,  # Normalize -110 to -70 dBm
                buffer_occupancy / 100.0,
                avg_neighbor_load,
                user_distribution,
                (load_trend + 0.5) / 1.0  # Normalize -0.5 to 0.5
            ], dtype=np.float32)

            # Determine load balancing action
            action = self._determine_load_balancing_action(
                prb_utilization, active_users, avg_neighbor_load,
                handover_rate, buffer_occupancy, load_trend
            )

            # One-hot encoded label
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[action] = 1.0

            sample = DataSample(
                features=features,
                label=label,
                timestamp=i * 1.0,  # 1 second intervals
                metadata={
                    'cell_id': cell_id,
                    'active_users': active_users,
                    'prb_utilization': prb_utilization,
                    'avg_throughput': avg_throughput,
                    'neighbor_loads': neighbor_loads
                }
            )

            samples.append(sample)

        return samples

    def _determine_load_balancing_action(self, prb_util: float, 
                                        active_users: int,
                                        neighbor_load: float,
                                        handover_rate: float,
                                        buffer_occ: float,
                                        load_trend: float) -> int:
        """
        Determine optimal load balancing action

        Returns:
            0: No action - Cell operating normally
            1: Offload users - Cell overloaded, push users to neighbors
            2: Accept handover - Cell underloaded, can accept more users
            3: Increase capacity - Need capacity expansion
        """
        # Critical overload: Offload users to neighbors
        if prb_util > 85 and neighbor_load < 0.7:
            return 1  # Offload

        # Severe congestion with no offload option: Increase capacity
        elif prb_util > 90 or buffer_occ > 95:
            return 3  # Increase capacity

        # Underloaded and neighbors are overloaded: Accept handover
        elif prb_util < 50 and neighbor_load > 0.75:
            return 2  # Accept handover

        # Normal operation
        else:
            return 0  # No action

    def _load_from_file(self, data_path: str) -> bool:
        """Load real cell load data from file"""
        try:
            import pandas as pd
            df = pd.read_csv(data_path)

            required_cols = ['active_users', 'prb_utilization', 'action']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing columns in {data_path}")
                return False

            self._data = []
            for _, row in df.iterrows():
                features = np.array([
                    row['active_users'] / 100.0,
                    row['prb_utilization'] / 100.0,
                    row.get('avg_throughput', 50) / 100.0,
                    row.get('handover_rate', 0.5) / 2.0,
                    row.get('cell_edge_ratio', 0.3),
                    row.get('interference', -90) / 40.0,
                    row.get('buffer_occupancy', 50) / 100.0,
                    row.get('neighbor_load', 0.5),
                    row.get('user_distribution', 0.5),
                    row.get('load_trend', 0) / 1.0
                ], dtype=np.float32)

                label = np.zeros(self.num_classes, dtype=np.float32)
                label[int(row['action'])] = 1.0

                sample = DataSample(features=features, label=label)
                self._data.append(sample)

            print(f"✓ Loaded {len(self._data)} samples from {data_path}")
            return True

        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
            return False

    def partition_data(self, agents: List[str], 
                      strategy: str = 'iid') -> Dict[str, DataPartition]:
        """
        Partition data across O-DU agents (cells)

        Strategies:
        - 'iid': Random distribution
        - 'cell-based': Each agent gets data from specific cells
        - 'load-based': Partition by load levels
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        self._partitions = {}

        if strategy == 'iid':
            self._partitions = self._partition_iid(agents)
        elif strategy == 'cell-based':
            self._partitions = self._partition_cell_based(agents)
        elif strategy == 'load-based':
            self._partitions = self._partition_load_based(agents)
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")

        return self._partitions

    def _partition_iid(self, agents: List[str]) -> Dict[str, DataPartition]:
        """IID partitioning"""
        indices = np.random.permutation(len(self._data))
        splits = np.array_split(indices, len(agents))

        partitions = {}
        for agent_id, split_indices in zip(agents, splits):
            samples = [self._data[i] for i in split_indices]
            partitions[agent_id] = DataPartition(
                agent_id=agent_id,
                node_id=agent_id.replace('agent', 'ODU'),
                samples=samples
            )

        return partitions

    def _partition_cell_based(self, agents: List[str]) -> Dict[str, DataPartition]:
        """Each agent gets data from specific cells"""
        partitions = {}
        cells_per_agent = self.config.num_cells // len(agents)

        for i, agent_id in enumerate(agents):
            assigned_cells = list(range(i * cells_per_agent, 
                                       (i + 1) * cells_per_agent))

            samples = [s for s in self._data 
                      if s.metadata.get('cell_id') in assigned_cells]

            partitions[agent_id] = DataPartition(
                agent_id=agent_id,
                node_id=agent_id.replace('agent', 'ODU'),
                samples=samples
            )

        return partitions

    def _partition_load_based(self, agents: List[str]) -> Dict[str, DataPartition]:
        """Partition by load levels (non-IID)"""
        # Sort by load level
        sorted_data = sorted(self._data,
                           key=lambda s: s.metadata.get('prb_utilization', 50))

        partitions = {}
        samples_per_agent = len(sorted_data) // len(agents)

        for i, agent_id in enumerate(agents):
            start_idx = i * samples_per_agent
            end_idx = start_idx + samples_per_agent if i < len(agents) - 1 else len(sorted_data)

            samples = sorted_data[start_idx:end_idx]
            partitions[agent_id] = DataPartition(
                agent_id=agent_id,
                node_id=agent_id.replace('agent', 'ODU'),
                samples=samples
            )

        return partitions

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self._loaded:
            return {}

        features = np.array([s.features for s in self._data])
        labels = np.array([np.argmax(s.label) for s in self._data])

        action_counts = {
            'No action': np.sum(labels == 0),
            'Offload users': np.sum(labels == 1),
            'Accept handover': np.sum(labels == 2),
            'Increase capacity': np.sum(labels == 3)
        }

        stats = {
            'dataset': self.name,
            'total_samples': len(self._data),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'action_distribution': action_counts,
            'action_percentages': {k: v/len(labels)*100 for k, v in action_counts.items()},
            'feature_means': {name: features[:, i].mean() 
                            for i, name in enumerate(self._feature_names)},
            'feature_stds': {name: features[:, i].std() 
                           for i, name in enumerate(self._feature_names)},
            'num_cells': self.config.num_cells,
            'partitions': len(self._partitions)
        }

        return stats


if __name__ == "__main__":
    # Test the dataset
    config = CellLoadConfig(num_samples=1000)
    dataset = CellLoadDataset(config)
    dataset.load_data()

    print(f"\nDataset: {dataset}")
    print(f"\nStatistics:")
    import json
    print(json.dumps(dataset.get_statistics(), indent=2, default=str))

    # Test partitioning
    agents = ['agent_1', 'agent_2', 'agent_3']
    partitions = dataset.partition_data(agents, strategy='cell-based')

    print(f"\nPartitions:")
    for agent_id, partition in partitions.items():
        print(f"  {agent_id}: {len(partition)} samples")
