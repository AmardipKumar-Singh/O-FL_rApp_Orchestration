#!/usr/bin/env python3
"""
O-FL rApp: Distributed Orchestration of Concurrent Federated MARL Tasks
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from core.base import INetworkTopology


@dataclass
class Link:
    """Network link properties"""
    link_id: str
    source: str
    destination: str
    bandwidth: float  # Gbps
    latency: float  # ms
    cost: float  # μ/Gbps
    available_bandwidth: float = field(init=False)

    def __post_init__(self):
        self.available_bandwidth = self.bandwidth

    def allocate(self, amount: float) -> bool:
        """Allocate bandwidth on this link"""
        if amount <= self.available_bandwidth:
            self.available_bandwidth -= amount
            return True
        return False

    def release(self, amount: float):
        """Release bandwidth on this link"""
        self.available_bandwidth = min(self.bandwidth, 
                                      self.available_bandwidth + amount)

    def reset(self):
        """Reset link to full capacity"""
        self.available_bandwidth = self.bandwidth


@dataclass
class Node:
    """Network node (O-DU or RIC)"""
    node_id: str
    node_type: str  # 'DU' or 'RIC'
    compute_capacity: float  # TOPS
    available_capacity: float = field(init=False)
    hosted_agents: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.available_capacity = self.compute_capacity

    def allocate_compute(self, amount: float) -> bool:
        """Allocate compute resources"""
        if amount <= self.available_capacity:
            self.available_capacity -= amount
            return True
        return False

    def release_compute(self, amount: float):
        """Release compute resources"""
        self.available_capacity = min(self.compute_capacity,
                                     self.available_capacity + amount)

    def reset(self):
        """Reset node to full capacity"""
        self.available_capacity = self.compute_capacity


class ORANTopology(INetworkTopology):
    """
    O-RAN Network Topology Implementation

    Hierarchical structure:
    - O-DUs (edge nodes) with agents
    - Near-RT-RIC (aggregation controller)
    - Multiple transport links (fiber, microwave)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize O-RAN topology

        Args:
            config: Configuration dictionary with network parameters
        """
        self._nodes: Dict[str, Node] = {}
        self._links: Dict[str, Link] = {}
        self._agent_locations: Dict[str, str] = {}
        self._config = config or {}

        self._initialize_from_config()

    def _initialize_from_config(self):
        """Initialize topology from configuration"""
        if not self._config:
            self._create_default_topology()

    def _create_default_topology(self):
        """Create default O-RAN topology"""
        # Create O-DUs
        self.add_node('ODU_1', 'DU', compute_capacity=80.0)
        self.add_node('ODU_2', 'DU', compute_capacity=80.0)

        # Create RIC
        self.add_node('RIC', 'RIC', compute_capacity=100.0)

        # Create fiber links
        self.add_link('ODU_1_RIC_fiber', 'ODU_1', 'RIC', 
                     bandwidth=10.0, latency=0.5, cost=10.0)
        self.add_link('ODU_2_RIC_fiber', 'ODU_2', 'RIC',
                     bandwidth=10.0, latency=0.5, cost=10.0)

        # Create microwave links
        self.add_link('ODU_1_RIC_microwave', 'ODU_1', 'RIC',
                     bandwidth=1.0, latency=5.0, cost=2.0)
        self.add_link('ODU_2_RIC_microwave', 'ODU_2', 'RIC',
                     bandwidth=1.0, latency=5.0, cost=2.0)

    def add_node(self, node_id: str, node_type: str, 
                compute_capacity: float) -> Node:
        """Add a node to the topology"""
        node = Node(node_id, node_type, compute_capacity)
        self._nodes[node_id] = node
        return node

    def add_link(self, link_id: str, source: str, destination: str,
                bandwidth: float, latency: float, cost: float) -> Link:
        """Add a link to the topology"""
        link = Link(link_id, source, destination, bandwidth, latency, cost)
        self._links[link_id] = link
        return link

    def assign_agent_to_node(self, agent_id: str, node_id: str):
        """Assign an agent to a node"""
        if node_id in self._nodes:
            self._agent_locations[agent_id] = node_id
            self._nodes[node_id].hosted_agents.append(agent_id)
        else:
            raise ValueError(f"Node {node_id} not found")

    def get_nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node object"""
        return self._nodes.get(node_id)

    def get_links(self) -> Dict[str, Dict]:
        """Get links as dictionary of properties"""
        return {
            link_id: {
                'source': link.source,
                'destination': link.destination,
                'bandwidth': link.bandwidth,
                'latency': link.latency,
                'cost': link.cost,
                'available': link.available_bandwidth
            }
            for link_id, link in self._links.items()
        }

    def get_link(self, link_id: str) -> Optional[Link]:
        """Get link object"""
        return self._links.get(link_id)

    def get_agent_location(self, agent_id: str) -> Optional[str]:
        return self._agent_locations.get(agent_id)

    def get_path_properties(self, src: str, dst: str, 
                          link_type: str) -> Optional[Dict]:
        """Get properties of a path"""
        link_id = f"{src}_{dst}_{link_type}"
        link = self._links.get(link_id)

        if link:
            return {
                'bandwidth': link.bandwidth,
                'latency': link.latency,
                'cost': link.cost,
                'available': link.available_bandwidth
            }
        return None

    def check_capacity_constraint(self, link_id: str, 
                                  allocated: float) -> bool:
        """Check if allocation satisfies link capacity constraint"""
        link = self._links.get(link_id)
        if link:
            return allocated <= link.bandwidth
        return False

    def get_links_from_node(self, node_id: str) -> List[str]:
        """Get all links originating from a node"""
        return [link_id for link_id, link in self._links.items()
                if link.source == node_id]

    def get_links_to_node(self, node_id: str) -> List[str]:
        """Get all links terminating at a node"""
        return [link_id for link_id, link in self._links.items()
                if link.destination == node_id]

    def reset_resources(self):
        """Reset all resource allocations"""
        for node in self._nodes.values():
            node.reset()
        for link in self._links.values():
            link.reset()

    def get_topology_summary(self) -> Dict:
        """Get summary of topology"""
        return {
            'nodes': {
                node_id: {
                    'type': node.node_type,
                    'capacity': node.compute_capacity,
                    'available': node.available_capacity,
                    'agents': len(node.hosted_agents)
                }
                for node_id, node in self._nodes.items()
            },
            'links': {
                link_id: {
                    'src': link.source,
                    'dst': link.destination,
                    'bandwidth': link.bandwidth,
                    'latency': link.latency,
                    'cost': link.cost
                }
                for link_id, link in self._links.items()
            },
            'agents': len(self._agent_locations)
        }


class TopologyBuilder:
    """Builder pattern for creating network topologies"""

    def __init__(self):
        self._topology = ORANTopology()

    def add_odu(self, odu_id: str, capacity: float = 80.0) -> 'TopologyBuilder':
        """Add an O-DU node"""
        self._topology.add_node(odu_id, 'DU', capacity)
        return self

    def add_ric(self, ric_id: str = 'RIC', capacity: float = 100.0) -> 'TopologyBuilder':
        """Add RIC node"""
        self._topology.add_node(ric_id, 'RIC', capacity)
        return self

    def add_fiber_link(self, src: str, dst: str, 
                      bandwidth: float = 10.0) -> 'TopologyBuilder':
        """Add fiber link"""
        link_id = f"{src}_{dst}_fiber"
        self._topology.add_link(link_id, src, dst, bandwidth, 0.5, 10.0)
        return self

    def add_microwave_link(self, src: str, dst: str,
                          bandwidth: float = 1.0) -> 'TopologyBuilder':
        """Add microwave link"""
        link_id = f"{src}_{dst}_microwave"
        self._topology.add_link(link_id, src, dst, bandwidth, 5.0, 2.0)
        return self

    def assign_agent(self, agent_id: str, node_id: str) -> 'TopologyBuilder':
        """Assign agent to node"""
        self._topology.assign_agent_to_node(agent_id, node_id)
        return self

    def build(self) -> ORANTopology:
        """Build the topology"""
        return self._topology


if __name__ == "__main__":
    # Test topology creation
    builder = TopologyBuilder()
    topology = (builder
                .add_odu('ODU_1', 80.0)
                .add_odu('ODU_2', 80.0)
                .add_ric('RIC', 100.0)
                .add_fiber_link('ODU_1', 'RIC')
                .add_fiber_link('ODU_2', 'RIC')
                .add_microwave_link('ODU_1', 'RIC')
                .add_microwave_link('ODU_2', 'RIC')
                .assign_agent('a1', 'ODU_1')
                .assign_agent('a2', 'ODU_2')
                .build())

    print("Topology Summary:")
    import json
    print(json.dumps(topology.get_topology_summary(), indent=2))
