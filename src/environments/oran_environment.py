#!/usr/bin/env python3
"""
environments/oran_environment.py - O-RAN MARL Environment
O-FL rApp: Multi-Agent Environment for Network Control

"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from training.marl_base import (MARLEnvironment, MARLState, MARLAction,
                                MARLReward, MARLTransition)
from models.network import ORANTopology


@dataclass
class ORANEnvConfig:
    """Configuration for O-RAN MARL environment"""
    num_odus: int = 3
    num_users_per_odu: int = 50
    max_prb: int = 100  # Physical Resource Blocks
    max_bandwidth: float = 10.0  # Gbps
    episode_length: int = 100
    reward_type: str = 'throughput'  # 'throughput', 'fairness', 'qos'


class ORANNetworkEnvironment(MARLEnvironment):

    def __init__(self, topology: ORANTopology, config: Optional[ORANEnvConfig] = None):
        self.topology = topology
        self.config = config or ORANEnvConfig()

        # Get O-DU agents
        self.agent_ids = [node_id for node_id in topology.get_nodes() 
                         if node_id != 'RIC']

        # State dimensions
        self.obs_dim = 7  # Per agent observation
        self.global_state_dim = len(self.agent_ids) * self.obs_dim

        # Action dimensions
        self.action_dim = 3  # [prb_allocation, handover_threshold, power_level]

        # Environment state
        self.current_step = 0
        self.prb_utilization: Dict[str, float] = {}
        self.active_users: Dict[str, int] = {}
        self.buffer_occupancy: Dict[str, float] = {}
        self.throughput_history: Dict[str, List[float]] = {}

        # Initialize
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize environment state"""
        for agent_id in self.agent_ids:
            self.prb_utilization[agent_id] = np.random.uniform(0.3, 0.7)
            self.active_users[agent_id] = np.random.randint(
                10, self.config.num_users_per_odu
            )
            self.buffer_occupancy[agent_id] = np.random.uniform(0.2, 0.5)
            self.throughput_history[agent_id] = [50.0]  # Initial throughput

    def reset(self) -> MARLState:
        """Reset environment and return initial state"""
        self.current_step = 0
        self._initialize_state()
        return self._get_state()

    def step(self, action: MARLAction) -> Tuple[MARLState, MARLReward, bool, Dict]:
        """
        Execute action and return next state, reward, done, info

        Actions:
        - action[0]: PRB allocation ratio (0-1)
        - action[1]: Handover threshold (0-1)
        - action[2]: Power control level (0-1)
        """
        self.current_step += 1

        # Execute actions for each agent
        for agent_id in self.agent_ids:
            agent_action = action.get_agent_action(agent_id)
            self._execute_agent_action(agent_id, agent_action)

        # Update environment dynamics
        self._update_dynamics()

        # Compute rewards
        reward = self._compute_reward()

        # Get next state
        next_state = self._get_state()

        # Check if episode is done
        done = self.current_step >= self.config.episode_length

        # Additional info
        info = {
            'step': self.current_step,
            'avg_throughput': np.mean([self.throughput_history[a][-1] 
                                      for a in self.agent_ids]),
            'avg_prb_util': np.mean(list(self.prb_utilization.values())),
            'total_users': sum(self.active_users.values())
        }

        return next_state, reward, done, info

    def _execute_agent_action(self, agent_id: str, action: np.ndarray) -> None:
        """Execute action for single agent"""
        prb_alloc = np.clip(action[0], 0, 1)
        handover_thresh = np.clip(action[1], 0, 1)
        power_level = np.clip(action[2], 0, 1)

        # Update PRB utilization based on allocation
        current_users = self.active_users[agent_id]
        target_util = prb_alloc * (current_users / self.config.num_users_per_odu)
        self.prb_utilization[agent_id] = 0.7 * self.prb_utilization[agent_id] + 0.3 * target_util

        # Handover effects
        if self.prb_utilization[agent_id] > 0.8 and handover_thresh > 0.6:
            # Try to handover users to less loaded neighbors
            neighbors = self._get_neighbor_loads(agent_id)
            if neighbors:
                min_neighbor = min(neighbors, key=neighbors.get)
                if neighbors[min_neighbor] < 0.6:
                    # Successful handover
                    users_to_handover = max(1, int(current_users * 0.1))
                    self.active_users[agent_id] -= users_to_handover
                    self.active_users[min_neighbor] += users_to_handover

        # Power control affects interference and throughput
        # Higher power = better signal but more interference
        # This is simplified - actual implementation would model SINR

    def _update_dynamics(self) -> None:
        """Update environment dynamics (user arrivals, departures, etc.)"""
        for agent_id in self.agent_ids:
            # User arrivals (Poisson process)
            arrivals = np.random.poisson(2.0)
            departures = np.random.poisson(1.5)

            # Update active users
            self.active_users[agent_id] = np.clip(
                self.active_users[agent_id] + arrivals - departures,
                0, self.config.num_users_per_odu
            )

            # Update buffer occupancy based on load
            load = self.prb_utilization[agent_id]
            self.buffer_occupancy[agent_id] = 0.8 * self.buffer_occupancy[agent_id] + 0.2 * load

            # Compute throughput
            throughput = self._compute_throughput(agent_id)
            self.throughput_history[agent_id].append(throughput)

    def _compute_throughput(self, agent_id: str) -> float:
        """Compute throughput for agent based on utilization and interference"""
        util = self.prb_utilization[agent_id]
        users = self.active_users[agent_id]

        # Shannon capacity (simplified)
        # Throughput = bandwidth * log(1 + SNR)
        # SNR decreases with utilization and interference

        neighbors = self._get_neighbor_loads(agent_id)
        interference = np.mean(list(neighbors.values())) if neighbors else 0.0

        # Effective SNR (simplified model)
        snr = 20.0 * (1 - util) / (1 + interference)

        # Throughput in Mbps
        throughput = self.config.max_bandwidth * 1000 * np.log2(1 + snr)
        throughput = throughput / max(1, users)  # Per-user throughput

        return max(1.0, throughput)

    def _compute_reward(self) -> MARLReward:
        """Compute rewards for all agents"""
        individual_rewards = {}

        if self.config.reward_type == 'throughput':
            # Reward based on throughput
            for agent_id in self.agent_ids:
                throughput = self.throughput_history[agent_id][-1]
                individual_rewards[agent_id] = throughput / 100.0  # Normalize

        elif self.config.reward_type == 'fairness':
            # Reward based on fairness (Jain's index)
            throughputs = [self.throughput_history[a][-1] for a in self.agent_ids]
            jain_index = (sum(throughputs) ** 2) / (len(throughputs) * sum(t ** 2 for t in throughputs))

            for agent_id in self.agent_ids:
                individual_rewards[agent_id] = jain_index * 10.0

        elif self.config.reward_type == 'qos':
            # Reward based on QoS satisfaction
            for agent_id in self.agent_ids:
                util = self.prb_utilization[agent_id]
                buffer = self.buffer_occupancy[agent_id]

                # Penalty for high utilization and buffer occupancy
                qos_penalty = max(0, util - 0.8) * 10 + max(0, buffer - 0.8) * 10
                throughput_reward = self.throughput_history[agent_id][-1] / 100.0

                individual_rewards[agent_id] = throughput_reward - qos_penalty

        # Team reward (system-wide objective)
        team_reward = sum(individual_rewards.values())

        return MARLReward(individual_rewards, team_reward)

    def _get_state(self) -> MARLState:
        """Get current state for all agents"""
        observations = {}

        for agent_id in self.agent_ids:
            # Get neighbor loads
            neighbor_loads = self._get_neighbor_loads(agent_id)
            avg_neighbor_load = np.mean(list(neighbor_loads.values())) if neighbor_loads else 0.0

            # Agent observation
            obs = np.array([
                self.prb_utilization[agent_id],
                self.active_users[agent_id] / self.config.num_users_per_odu,
                self.buffer_occupancy[agent_id],
                avg_neighbor_load,
                self.throughput_history[agent_id][-1] / 100.0,  # Normalized
                len(neighbor_loads) / 3.0,  # Number of neighbors (normalized)
                self.current_step / self.config.episode_length  # Time progress
            ], dtype=np.float32)

            observations[agent_id] = obs

        # Global state (concatenate all observations)
        global_state = np.concatenate(list(observations.values()))

        return MARLState(observations, global_state)

    def _get_neighbor_loads(self, agent_id: str) -> Dict[str, float]:
        """Get loads of neighboring O-DUs"""
        # Get connected neighbors through links
        neighbors = {}
        links = self.topology.get_links()

        for link_id, link_props in links.items():
            if link_props['source'] == agent_id:
                dst = link_props['destination']
                if dst != 'RIC' and dst in self.prb_utilization:
                    neighbors[dst] = self.prb_utilization[dst]
            elif link_props['destination'] == agent_id:
                src = link_props['source']
                if src != 'RIC' and src in self.prb_utilization:
                    neighbors[src] = self.prb_utilization[src]

        return neighbors

    def get_observation_space(self, agent_id: str) -> Tuple[int, ...]:
        """Get observation space shape for agent"""
        return (self.obs_dim,)

    def get_action_space(self, agent_id: str) -> Tuple[int, ...]:
        """Get action space shape for agent"""
        return (self.action_dim,)

    def get_agent_ids(self) -> List[str]:
        """Get list of all agent IDs"""
        return self.agent_ids.copy()

    def render(self) -> str:
        """Render current environment state"""
        lines = [f"\n{'='*60}"]
        lines.append(f"O-RAN MARL Environment - Step {self.current_step}")
        lines.append(f"{'='*60}")

        for agent_id in self.agent_ids:
            lines.append(f"\n{agent_id}:")
            lines.append(f"  PRB Utilization: {self.prb_utilization[agent_id]:.2%}")
            lines.append(f"  Active Users: {self.active_users[agent_id]}")
            lines.append(f"  Buffer: {self.buffer_occupancy[agent_id]:.2%}")
            lines.append(f"  Throughput: {self.throughput_history[agent_id][-1]:.1f} Mbps")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)


if __name__ == "__main__":
    from models.network import TopologyBuilder

    # Create topology
    topology = (TopologyBuilder()
        .add_odu('ODU_1', 80)
        .add_odu('ODU_2', 80)
        .add_odu('ODU_3', 80)
        .add_ric('RIC', 100)
        .add_fiber_link('ODU_1', 'RIC')
        .add_fiber_link('ODU_2', 'RIC')
        .add_fiber_link('ODU_3', 'RIC')
        .build())

    # Create environment
    env = ORANNetworkEnvironment(topology)

    print(f"Environment created with {len(env.get_agent_ids())} agents")
    print(f"Observation space: {env.get_observation_space('ODU_1')}")
    print(f"Action space: {env.get_action_space('ODU_1')}")

    # Test episode
    state = env.reset()
    print(env.render())
