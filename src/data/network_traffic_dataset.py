#!/usr/bin/env python3
"""
data/network_traffic_dataset.py - Network Traffic Traces for QoS Prediction
O-FL rApp: O-RAN Dataset Implementation

Dataset characteristics:
- Time-series data from O-DU network interfaces
- Features: Throughput, latency, jitter, packet loss, etc.
- Labels: QoS class (0: Critical, 1: High, 2: Medium, 3: Best-effort)
- Use case: Predict QoS requirements for incoming traffic flows
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from data.base_dataset import ORANDataset, DataSample, DataPartition


@dataclass
class NetworkTrafficConfig:
    """Configuration for network traffic dataset"""
    num_samples: int = 10000
    window_size: int = 10  # Time window for features
    num_odus: int = 2
    noise_level: float = 0.1
    temporal_correlation: float = 0.8  # Temporal dependency

    # Traffic patterns
    peak_hours: Tuple[int, int] = (8, 20)  # 8 AM to 8 PM
    base_throughput: float = 100.0  # Mbps
    peak_multiplier: float = 3.0


class NetworkTrafficDataset(ORANDataset):
    """
    Network Traffic Dataset for QoS Prediction

    Features (per time window):
    - Throughput (Mbps)
    - Latency (ms)
    - Jitter (ms)
    - Packet loss rate (%)
    - Active connections
    - Traffic burst index
    - Time of day (encoded)
    - Day of week (encoded)
    - Historical QoS (previous window)
    - Resource utilization (%)

    Labels:
    - QoS Class: 0 (Critical/uRLLC), 1 (High/eMBB), 2 (Medium), 3 (Best-effort)
    """

    def __init__(self, config: Optional[NetworkTrafficConfig] = None):
        super().__init__("NetworkTraffic")
        self.config = config or NetworkTrafficConfig()
        self.feature_dim = 10
        self.num_classes = 4
        self._feature_names = [
            'throughput', 'latency', 'jitter', 'packet_loss',
            'connections', 'burst_index', 'time_of_day', 
            'day_of_week', 'prev_qos', 'resource_util'
        ]

    def load_data(self, data_path: Optional[str] = None) -> None:
        """
        Load or generate network traffic data

        Args:
            data_path: Path to real data file (if None, generates synthetic)
        """
        if data_path and self._load_from_file(data_path):
            self._loaded = True
            return

        print(f"Generating synthetic network traffic data ({self.config.num_samples} samples)...")
        self._data = self._generate_synthetic_data()
        self._loaded = True
        print(f"✓ Generated {len(self._data)} network traffic samples")

    def _generate_synthetic_data(self) -> List[DataSample]:
        """Generate realistic synthetic network traffic data"""
        samples = []

        # Initialize state for temporal correlation
        prev_throughput = self.config.base_throughput
        prev_latency = 10.0
        prev_qos = 1

        for i in range(self.config.num_samples):
            # Time context (24-hour cycle, 7-day week)
            hour = (i * 0.1) % 24
            day = (i // 240) % 7

            # Peak hour effect
            is_peak = self.config.peak_hours[0] <= hour <= self.config.peak_hours[1]
            load_factor = self.config.peak_multiplier if is_peak else 1.0

            # Generate correlated throughput
            target_throughput = self.config.base_throughput * load_factor
            prev_throughput = (self.config.temporal_correlation * prev_throughput +
                             (1 - self.config.temporal_correlation) * target_throughput)
            throughput = prev_throughput + np.random.normal(0, 10 * self.config.noise_level)
            throughput = max(0, throughput)

            # Latency inversely related to available capacity
            capacity_ratio = throughput / (self.config.base_throughput * self.config.peak_multiplier)
            base_latency = 5.0 + 15.0 * capacity_ratio  # Higher load = higher latency
            prev_latency = (self.config.temporal_correlation * prev_latency +
                          (1 - self.config.temporal_correlation) * base_latency)
            latency = prev_latency + np.random.normal(0, 2 * self.config.noise_level)
            latency = max(1.0, latency)

            # Jitter increases with load
            jitter = 0.5 + 2.0 * capacity_ratio + np.random.exponential(0.5)

            # Packet loss increases non-linearly with load
            packet_loss = 0.01 * (capacity_ratio ** 2) + np.random.exponential(0.005)
            packet_loss = min(1.0, packet_loss)

            # Active connections
            connections = int(50 + 200 * load_factor + np.random.normal(0, 20))
            connections = max(1, connections)

            # Traffic burst index (variance in recent throughput)
            burst_index = np.random.gamma(2, 0.3)

            # Resource utilization
            resource_util = capacity_ratio + np.random.uniform(-0.1, 0.1)
            resource_util = np.clip(resource_util, 0, 1)

            # Features vector
            features = np.array([
                throughput / 300.0,  # Normalized
                latency / 50.0,
                jitter / 5.0,
                packet_loss,
                connections / 300.0,
                burst_index,
                np.sin(2 * np.pi * hour / 24),  # Cyclic encoding
                np.cos(2 * np.pi * day / 7),
                prev_qos / 3.0,  # Previous QoS class
                resource_util
            ], dtype=np.float32)

            # Determine QoS class based on characteristics
            qos_class = self._determine_qos_class(
                latency, jitter, packet_loss, throughput, burst_index
            )

            # One-hot encoded label
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[qos_class] = 1.0

            sample = DataSample(
                features=features,
                label=label,
                timestamp=i * 0.1,  # 0.1 second intervals
                metadata={
                    'hour': hour,
                    'day': day,
                    'is_peak': is_peak,
                    'raw_throughput': throughput,
                    'raw_latency': latency
                }
            )

            samples.append(sample)
            prev_qos = qos_class

        return samples

    def _determine_qos_class(self, latency: float, jitter: float, 
                            packet_loss: float, throughput: float,
                            burst_index: float) -> int:
        """
        Determine QoS class based on network characteristics

        Returns:
            0: Critical (uRLLC) - Low latency, low jitter, low loss
            1: High (eMBB) - High throughput required
            2: Medium - Moderate requirements
            3: Best-effort - No strict requirements
        """
        # Critical (uRLLC): Strict latency requirements
        if latency < 6.0 and jitter < 1.0 and packet_loss < 0.001:
            return 0

        # High priority (eMBB): High throughput needs
        elif throughput > 200.0 or burst_index > 0.8:
            return 1

        # Medium priority: Moderate requirements
        elif latency < 20.0 and packet_loss < 0.01:
            return 2

        # Best-effort: Everything else
        else:
            return 3

    def _load_from_file(self, data_path: str) -> bool:
        """Load real network traffic data from file"""
        try:
            import pandas as pd
            df = pd.read_csv(data_path)

            # Expected columns in real dataset
            required_cols = ['throughput', 'latency', 'jitter', 'packet_loss', 'qos_class']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing columns in {data_path}")
                return False

            self._data = []
            for _, row in df.iterrows():
                features = np.array([
                    row['throughput'] / 300.0,
                    row['latency'] / 50.0,
                    row['jitter'] / 5.0,
                    row['packet_loss'],
                    row.get('connections', 100) / 300.0,
                    row.get('burst_index', 0.5),
                    np.sin(2 * np.pi * row.get('hour', 12) / 24),
                    np.cos(2 * np.pi * row.get('day', 1) / 7),
                    row.get('prev_qos', 1) / 3.0,
                    row.get('resource_util', 0.5)
                ], dtype=np.float32)

                label = np.zeros(self.num_classes, dtype=np.float32)
                label[int(row['qos_class'])] = 1.0

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
        Partition data across O-DU agents

        Strategies:
        - 'iid': Independent and identically distributed
        - 'non-iid': Non-IID based on time patterns
        - 'geographic': Geographic/cell-based partitioning
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        self._partitions = {}

        if strategy == 'iid':
            self._partitions = self._partition_iid(agents)
        elif strategy == 'non-iid':
            self._partitions = self._partition_non_iid(agents)
        elif strategy == 'geographic':
            self._partitions = self._partition_geographic(agents)
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")

        return self._partitions

    def _partition_iid(self, agents: List[str]) -> Dict[str, DataPartition]:
        """IID partitioning: Random uniform split"""
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

    def _partition_non_iid(self, agents: List[str]) -> Dict[str, DataPartition]:
        """Non-IID partitioning: Different time patterns per agent"""
        # Sort by QoS class
        sorted_data = sorted(self._data, 
                           key=lambda s: np.argmax(s.label))

        # Each agent gets data from specific QoS classes
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

    def _partition_geographic(self, agents: List[str]) -> Dict[str, DataPartition]:
        """Geographic partitioning: Different cells have different patterns"""
        # Assign different load patterns to different geographic areas
        partitions = {}

        for i, agent_id in enumerate(agents):
            # Each agent represents a different cell with unique characteristics
            cell_samples = []

            for sample in self._data:
                # Probabilistic assignment based on cell characteristics
                if np.random.random() < (1.0 / len(agents) * 1.2):  # Slight overlap
                    cell_samples.append(sample)

            partitions[agent_id] = DataPartition(
                agent_id=agent_id,
                node_id=agent_id.replace('agent', 'ODU'),
                samples=cell_samples[:len(self._data) // len(agents)]  # Balance sizes
            )

        return partitions

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self._loaded:
            return {}

        features = np.array([s.features for s in self._data])
        labels = np.array([np.argmax(s.label) for s in self._data])

        qos_counts = {
            'Critical (uRLLC)': np.sum(labels == 0),
            'High (eMBB)': np.sum(labels == 1),
            'Medium': np.sum(labels == 2),
            'Best-effort': np.sum(labels == 3)
        }

        stats = {
            'dataset': self.name,
            'total_samples': len(self._data),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'qos_distribution': qos_counts,
            'qos_percentages': {k: v/len(labels)*100 for k, v in qos_counts.items()},
            'feature_means': {name: features[:, i].mean() 
                            for i, name in enumerate(self._feature_names)},
            'feature_stds': {name: features[:, i].std() 
                           for i, name in enumerate(self._feature_names)},
            'partitions': len(self._partitions)
        }

        return stats


if __name__ == "__main__":
    # Test the dataset
    config = NetworkTrafficConfig(num_samples=1000)
    dataset = NetworkTrafficDataset(config)
    dataset.load_data()

    print(f"\nDataset: {dataset}")
    print(f"\nStatistics:")
    import json
    print(json.dumps(dataset.get_statistics(), indent=2, default=str))

    # Test partitioning
    agents = ['agent_1', 'agent_2', 'agent_3', 'agent_4']
    partitions = dataset.partition_data(agents, strategy='iid')

    print(f"\nPartitions:")
    for agent_id, partition in partitions.items():
        print(f"  {agent_id}: {len(partition)} samples")
