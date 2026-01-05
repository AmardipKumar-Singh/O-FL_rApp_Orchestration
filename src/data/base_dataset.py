#!/usr/bin/env python3
"""
data/base_dataset.py - Abstract Dataset Interface
O-FL rApp: O-RAN Dataset Management
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class DataSample:
    """Single data sample with features and labels"""
    features: np.ndarray
    label: np.ndarray
    timestamp: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class DataPartition:
    """Data partition for a specific agent/node"""
    agent_id: str
    node_id: str
    samples: List[DataSample]

    def get_features(self) -> np.ndarray:
        """Get all features as array"""
        return np.array([s.features for s in self.samples])

    def get_labels(self) -> np.ndarray:
        """Get all labels as array"""
        return np.array([s.label for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)


class ORANDataset(ABC):
    """Abstract base class for O-RAN datasets"""

    def __init__(self, name: str):
        self.name = name
        self._data: List[DataSample] = []
        self._partitions: Dict[str, DataPartition] = {}
        self._loaded = False

    @abstractmethod
    def load_data(self, data_path: Optional[str] = None) -> None:
        """Load dataset from file or generate synthetic data"""
        pass

    @abstractmethod
    def partition_data(self, agents: List[str], 
                      strategy: str = 'iid') -> Dict[str, DataPartition]:
        """
        Partition data across agents

        Args:
            agents: List of agent IDs
            strategy: Partitioning strategy ('iid', 'non-iid', 'geographic')

        Returns:
            Dictionary mapping agent_id to DataPartition
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        pass

    def get_partition(self, agent_id: str) -> Optional[DataPartition]:
        """Get data partition for specific agent"""
        return self._partitions.get(agent_id)

    def get_all_partitions(self) -> Dict[str, DataPartition]:
        """Get all data partitions"""
        return self._partitions.copy()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"samples={len(self._data)}, partitions={len(self._partitions)})")


class DataLoader:
    """Data loader for batching during training"""

    def __init__(self, partition: DataPartition, batch_size: int = 32, 
                 shuffle: bool = True):
        self.partition = partition
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = list(range(len(partition)))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self._indices)

        for i in range(0, len(self._indices), self.batch_size):
            batch_indices = self._indices[i:i + self.batch_size]
            batch_samples = [self.partition.samples[j] for j in batch_indices]

            features = np.array([s.features for s in batch_samples])
            labels = np.array([s.label for s in batch_samples])

            yield features, labels

    def __len__(self) -> int:
        return (len(self.partition) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    print("O-RAN Dataset Base Classes")
