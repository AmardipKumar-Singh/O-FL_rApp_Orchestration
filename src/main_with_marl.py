#!/usr/bin/env python3
"""
main_with_marl.py - Complete O-RAN MARL Training Example
O-FL rApp: Integrated MAPPO Training for Network Control

Demonstrates:
1. O-RAN MARL environment setup
2. MAPPO multi-agent training
3. Resource orchestration with MARL tasks
4. Performance evaluation
"""

import numpy as np
from typing import Dict, List

from models.network import TopologyBuilder
from models.task import TaskBuilder, TaskType
from environments.oran_environment import ORANNetworkEnvironment, ORANEnvConfig
from training.mappo_trainer import MAPPOTrainer, MAPPOConfig
from training.marl_base import MARLTransition
from orchestrator_with_training import IntegratedOFLrApp, IntegratedConfig
from orchestrator import OrchestrationConfig
from training.fl_trainer import TrainingConfig


def train_mappo_agents(env: ORANNetworkEnvironment, 
                      trainer: MAPPOTrainer,
                      num_episodes: int = 100) -> Dict:
    """
    Train MAPPO agents in O-RAN environment

    Args:
        env: O-RAN MARL environment
        trainer: MAPPO trainer
        num_episodes: Number of training episodes

    Returns:
        Training statistics
    """
    print(f"\n{'='*70}")
    print("MAPPO Training: Multi-Agent Network Control")
    print(f"{'='*70}")
    print(f"Agents: {env.get_agent_ids()}")
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {env.config.episode_length}")
    print(f"{'='*70}\n")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_transitions = []

        for step in range(env.config.episode_length):
            # Select actions
            action = trainer.select_actions(state, explore=True)

            # Execute in environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            transition = MARLTransition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info
            )
            episode_transitions.append(transition)
            trainer.buffer.push(transition)

            episode_reward += reward.team_reward
            state = next_state

            if done:
                break

        # Train after episode (on-policy)
        if len(trainer.buffer) >= trainer.config.mini_batch_size:
            # Get last episode data
            batch = trainer.buffer.sample_last(len(episode_transitions))
            metrics = trainer.train_step(batch)

            # Clear buffer (on-policy)
            trainer.buffer.clear()
        else:
            metrics = {}

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Length (last 10): {avg_length:.1f}")

            if metrics:
                print(f"  Actor Loss: {metrics.get('actor_loss', 0):.4f}")
                print(f"  Critic Loss: {metrics.get('critic_loss', 0):.4f}")
                print(f"  Entropy: {metrics.get('entropy', 0):.4f}")

            print(f"  {info}")

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"{'='*70}\n")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_avg_reward': np.mean(episode_rewards[-10:]),
        'best_reward': max(episode_rewards)
    }


def evaluate_mappo_agents(env: ORANNetworkEnvironment,
                         trainer: MAPPOTrainer,
                         num_episodes: int = 10) -> Dict:
    """
    Evaluate trained MAPPO agents

    Args:
        env: O-RAN MARL environment
        trainer: Trained MAPPO trainer
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation statistics
    """
    print(f"\n{'='*70}")
    print("MAPPO Evaluation")
    print(f"{'='*70}\n")

    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(env.config.episode_length):
            # Select actions (no exploration)
            action = trainer.select_actions(state, explore=False)

            # Execute in environment
            next_state, reward, done, info = env.step(action)

            episode_reward += reward.team_reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\n{'='*70}")
    print(f"Evaluation Results:")
    print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min Reward: {min(episode_rewards):.2f}")
    print(f"  Max Reward: {max(episode_rewards):.2f}")
    print(f"{'='*70}\n")

    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'min_reward': min(episode_rewards),
        'max_reward': max(episode_rewards)
    }


def run_marl_scenario():
    """
    Scenario 1: Pure MARL Training

    Train MAPPO agents for O-RAN network control without orchestration
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Pure MARL Training (Network Control)")
    print("="*70 + "\n")

    # 1. Build O-RAN topology
    print("Step 1: Building O-RAN topology...")
    topology = (TopologyBuilder()
        .add_odu('ODU_1', capacity=100.0)
        .add_odu('ODU_2', capacity=100.0)
        .add_odu('ODU_3', capacity=100.0)
        .add_ric('RIC', capacity=150.0)
        .add_fiber_link('ODU_1', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_2', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_3', 'RIC', bandwidth=10.0)
        .add_microwave_link('ODU_1', 'ODU_2', bandwidth=1.0)
        .add_microwave_link('ODU_2', 'ODU_3', bandwidth=1.0)
        .add_microwave_link('ODU_1', 'ODU_3', bandwidth=1.0)
        .build())

    print(f"  Topology: 3 O-DUs, 1 RIC, 6 links\n")

    # 2. Create MARL environment
    print("Step 2: Creating O-RAN MARL environment...")
    env_config = ORANEnvConfig(
        num_odus=3,
        num_users_per_odu=50,
        episode_length=100,
        reward_type='qos'
    )
    env = ORANNetworkEnvironment(topology, env_config)

    print(f"  Agents: {env.get_agent_ids()}")
    print(f"  Obs dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim}\n")

    # 3. Create MAPPO trainer
    print("Step 3: Creating MAPPO trainer...")
    mappo_config = MAPPOConfig(
        gamma=0.99,
        lambda_gae=0.95,
        epsilon_clip=0.2,
        ppo_epochs=4,
        mini_batch_size=64,
        lr_actor=3e-4,
        lr_critic=1e-3,
        hidden_dim=64,
        action_type='continuous',
        use_shared_critic=True
    )

    agent_ids = env.get_agent_ids()
    obs_dims = {agent_id: env.obs_dim for agent_id in agent_ids}
    action_dims = {agent_id: env.action_dim for agent_id in agent_ids}

    trainer = MAPPOTrainer(agent_ids, obs_dims, action_dims, mappo_config)
    print(f"  MAPPO configured: {len(agent_ids)} agents\n")

    # 4. Train agents
    print("Step 4: Training MAPPO agents...")
    train_stats = train_mappo_agents(env, trainer, num_episodes=100)

    # 5. Evaluate agents
    print("Step 5: Evaluating trained agents...")
    eval_stats = evaluate_mappo_agents(env, trainer, num_episodes=10)

    # 6. Save model
    print("Step 6: Saving trained model...")
    trainer.save_model('mappo_oran_model.pkl')
    print(f"  Model saved to: mappo_oran_model.pkl\n")

    return train_stats, eval_stats


def run_integrated_marl_orchestration():
    """
    Scenario 2: MARL with O-FL rApp Orchestration

    Integrate MARL training with resource orchestration
    Multiple MARL tasks compete for resources
    """
    print("\n" + "="*70)
    print("SCENARIO 2: MARL with O-FL rApp Orchestration")
    print("="*70 + "\n")

    # Build topology
    topology = (TopologyBuilder()
        .add_odu('ODU_1', capacity=100.0)
        .add_odu('ODU_2', capacity=100.0)
        .add_odu('ODU_3', capacity=100.0)
        .add_ric('RIC', capacity=150.0)
        .add_fiber_link('ODU_1', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_2', 'RIC', bandwidth=10.0)
        .add_fiber_link('ODU_3', 'RIC', bandwidth=10.0)
        .build())

    # Create multiple MARL tasks
    print("Creating multiple MARL tasks...")

    # Task 1: Load balancing (uRLLC)
    task_load = (TaskBuilder()
        .with_id('T_LoadBalance')
        .with_type(TaskType.LATENCY_CRITICAL)
        .with_agents(['ODU_1', 'ODU_2'])
        .with_compute_requirements(agent=10.0, aggregator=15.0)
        .with_data_transfer(20.0)
        .with_latency_budget(5.0)
        .with_priority(2.0)
        .with_reward(150.0)
        .build())

    # Task 2: Spectrum management (eMBB)
    task_spectrum = (TaskBuilder()
        .with_id('T_Spectrum')
        .with_type(TaskType.THROUGHPUT_ORIENTED)
        .with_agents(['ODU_2', 'ODU_3'])
        .with_compute_requirements(agent=8.0, aggregator=12.0)
        .with_data_transfer(100.0)
        .with_priority(1.0)
        .with_reward(120.0)
        .build())

    print(f"  Task 1: {task_load.get_id()} (uRLLC)")
    print(f"  Task 2: {task_spectrum.get_id()} (eMBB)\n")

    # Configure orchestrator
    config = IntegratedConfig(
        orchestration=OrchestrationConfig(
            max_iterations=30,
            convergence_threshold=1e-4,
            w_reward=1.0,
            w_qos=1000.0
        ),
        training=TrainingConfig(),
        enable_real_training=False  # Simulation mode for MARL
    )

    orchestrator = IntegratedOFLrApp(config)
    orchestrator.initialize([task_load, task_spectrum], topology)

    print("Running orchestration with MARL tasks...")
    metrics_history = orchestrator.run_until_convergence()

    print("\nOrchestration completed!")
    print(f"Iterations: {len(metrics_history)}")

    return metrics_history


def main():
    """Run all MARL scenarios"""
    print("\n" + "="*70)
    print("O-FL rApp: MARL Implementation with MAPPO")
    print("="*70)
    print("\nScenarios:")
    print("  1. Pure MARL Training (Network Control)")
    print("  2. MARL with O-FL rApp Orchestration")
    print("\n" + "="*70)

    # Run scenarios
    try:
        # Scenario 1: Pure MARL
        train_stats, eval_stats = run_marl_scenario()

        # Scenario 2: Integrated orchestration
        orch_metrics = run_integrated_marl_orchestration()

        print("\n" + "="*70)
        print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nMAPPO Training:")
        print(f"  Final Avg Reward: {train_stats['final_avg_reward']:.2f}")
        print(f"  Best Reward: {train_stats['best_reward']:.2f}")
        print(f"\nMAPPO Evaluation:")
        print(f"  Avg Reward: {eval_stats['avg_reward']:.2f}")
        print(f"\nOrchestration:")
        print(f"  Iterations: {len(orch_metrics)}")

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
