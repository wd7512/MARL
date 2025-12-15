"""
Generic training loops for multi-agent reinforcement learning.

This module implements training regimes that work with any MARL environment:
1. Iterated Best Response (IBR) - Sequential updates with immediate propagation
2. Simultaneous Best Response (SBR) - Parallel updates with synchronized snapshots

These patterns are domain-agnostic and can be applied to any MARL problem.
"""

import os
import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

from easy_marl.core.agents import BaseAgent, PPOAgent

MAX_ENV_PROCS = psutil.cpu_count(logical=False) or os.cpu_count()
SAFE_MAX_ENV_PROCS = max(1, MAX_ENV_PROCS - 1)


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility across processes.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sequential_train(
    agents: List[PPOAgent],
    env_factory: Callable,
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    update_schedule: str = "SBR",
    update_probability: float = 1.0,
    evaluation_fn: Callable = None,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Sequential training with configurable update schedule (IBR or SBR).

    Update Schedules:
        - "SBR" (Simultaneous Best Response, default):
            All agents train against FROZEN SNAPSHOT of opponent policies.
            Updates synchronized at round boundaries (Jacobi-style).

        - "IBR" (Iterated Best Response):
            Each agent sees CURRENT state of all other agents.
            Updates propagate immediately (Gauss-Seidel-style).

    Args:
        agents: List of PPOAgent objects to train.
        env_factory: Function that creates environment for an agent.
                    Signature: (agent_index: int, agents: List) -> MARLEnvironment
        num_rounds: Number of complete training rounds.
        timesteps_per_agent: PPO training timesteps per agent per round.
        seed: Random seed for reproducibility.
        save_dir: Directory to save trained models.
        verbose: Whether to print training progress.
        update_schedule: "SBR" for simultaneous or "IBR" for iterated updates.
        update_probability: Probability (0-1) that an agent updates in a round.
        evaluation_fn: Optional function to evaluate agents after each training.
                      Signature: (agents: List) -> Dict

    Returns:
        agents: List of trained PPOAgent objects.
        training_info: Dictionary containing training metadata.

    Raises:
        ValueError: If update_schedule is not "IBR" or "SBR".
    """
    # Validate parameters
    update_schedule = update_schedule.upper()
    if update_schedule not in ("IBR", "SBR"):
        raise ValueError(
            f"update_schedule must be 'IBR' or 'SBR', got '{update_schedule}'"
        )

    N = len(agents)
    training_info = {
        "N": N,
        "num_rounds": num_rounds,
        "timesteps_per_agent": timesteps_per_agent,
        "total_timesteps": N * num_rounds * timesteps_per_agent,
        "seed": seed,
        "parallel": False,
        "update_schedule": update_schedule,
        "update_probability": update_probability,
    }

    # Ensure reproducible inertia decisions
    random.seed(seed)
    all_results = []

    for round_idx in range(num_rounds):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Round {round_idx + 1}/{num_rounds} [{update_schedule}]")
            print(f"{'=' * 60}")

        # SBR: Snapshot all agent policies at round start
        if update_schedule == "SBR":
            agents_snapshots = [agent.save_to_bytes() for agent in agents]

        trained_states = {}
        for i in range(N):
            # SBR: Restore all other agents to round-start snapshot
            if update_schedule == "SBR":
                for j, agent in enumerate(agents):
                    if j != i:
                        agent.load_from_bytes(agents_snapshots[j])

            # Inertia check (only for SBR after first round)
            rng = random.Random(seed + round_idx * 100 + i)
            if (
                update_schedule == "SBR"
                and rng.random() > update_probability
                and round_idx > 0
            ):
                if verbose:
                    print(f"  [Inertia] Skipping update for Agent {i}")
                trained_states[i] = agents_snapshots[i]
                continue

            # Set seeds for reproducibility
            training_seed = seed + round_idx * N + i
            set_all_seeds(training_seed)

            if verbose:
                print(f"\n-> Training Agent {i} ({timesteps_per_agent} timesteps)")

            # Create environment for this agent
            env = env_factory(agent_index=i, agents=agents, seed=training_seed)
            agents[i].model.set_env(env)
            agents[i].train(total_timesteps=timesteps_per_agent)

            # Save checkpoint
            if save_dir is not None:
                round_dir = os.path.join(save_dir, f"round_{round_idx + 1}")
                os.makedirs(round_dir, exist_ok=True)
                save_path = os.path.join(round_dir, f"agent_{i}")
                agents[i].save(save_path)
                if verbose:
                    print(f"  [OK] Saved to {save_path}")

            # SBR: Save trained state
            if update_schedule == "SBR":
                trained_states[i] = agents[i].save_to_bytes()

            # Evaluate if function provided
            if evaluation_fn is not None:
                eval_metrics = evaluation_fn(agents)
                all_results.append(
                    {
                        "round": round_idx + 1,
                        "agent": i,
                        "metrics": eval_metrics,
                    }
                )

        # SBR: Synchronize all trained policies at round end
        if update_schedule == "SBR":
            for i in range(N):
                agents[i].load_from_bytes(trained_states[i])
            if verbose:
                print(
                    f"\n[OK] Round {round_idx + 1} complete - all {N} agents synchronized"
                )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Total timesteps: {training_info['total_timesteps']:,}")

    return agents, training_info


def parallel_train(
    agents: List[PPOAgent],
    env_factory: Callable,
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    n_workers: int = SAFE_MAX_ENV_PROCS,
    update_probability: float = 1.0,
    evaluation_fn: Callable = None,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Simultaneous Best Response (SBR) training with parallel execution.

    Trains all agents in parallel within each round. Each agent optimizes
    against a FROZEN SNAPSHOT of opponent policies taken at round start.

    Args:
        agents: List of PPOAgent objects to train.
        env_factory: Function that creates environment for an agent.
        num_rounds: Number of complete training rounds.
        timesteps_per_agent: PPO training timesteps per agent per round.
        seed: Random seed for reproducibility.
        save_dir: Directory to save trained models.
        verbose: Whether to print training progress.
        n_workers: Number of parallel worker processes.
        update_probability: Probability (0-1) that an agent updates in a round.
        evaluation_fn: Optional function to evaluate agents after each training.

    Returns:
        agents: List of trained PPOAgent objects.
        training_info: Dictionary containing training metadata.
    """
    n_workers = min(n_workers, len(agents))
    N = len(agents)

    training_info = {
        "N": N,
        "num_rounds": num_rounds,
        "timesteps_per_agent": timesteps_per_agent,
        "total_timesteps": N * num_rounds * timesteps_per_agent,
        "seed": seed,
        "parallel": True,
        "n_workers": n_workers,
        "update_probability": update_probability,
    }

    random.seed(seed)
    all_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n{'=' * 60}")
                print(
                    f"Round {round_idx + 1}/{num_rounds} [PARALLEL - {n_workers} workers]"
                )
                print(f"{'=' * 60}")

            # Serialize all agent states at round start
            agents_states = [agent.save_to_bytes() for agent in agents]

            futures = []
            active_agent_indices = []

            for i in range(N):
                # Inertia check
                rng = random.Random(seed + round_idx * 100 + i)
                if rng.random() > update_probability and round_idx > 0:
                    if verbose:
                        print(f"  [Inertia] Skipping update for Agent {i}")
                    continue

                active_agent_indices.append(i)
                future = executor.submit(
                    _train_single_agent_worker,
                    agent_index=i,
                    round_idx=round_idx,
                    agents_states=agents_states,
                    env_factory=env_factory,
                    timesteps_per_agent=timesteps_per_agent,
                    save_dir=save_dir,
                    N=N,
                    seed=seed,
                )
                futures.append(future)

            # Collect results
            results_this_round = {}
            for future in as_completed(futures):
                agent_index, trained_state = future.result()
                results_this_round[agent_index] = trained_state

                if verbose:
                    print(f"  [OK] Agent {agent_index} training complete")

            # Update agents with trained states
            for i in active_agent_indices:
                agents[i].load_from_bytes(results_this_round[i])

            # Evaluate if function provided
            if evaluation_fn is not None:
                eval_metrics = evaluation_fn(agents)
                for i in active_agent_indices:
                    all_results.append(
                        {
                            "round": round_idx + 1,
                            "agent": i,
                            "metrics": eval_metrics,
                        }
                    )

            if verbose:
                print(
                    f"\n[OK] Round {round_idx + 1} complete - {len(active_agent_indices)} agents trained"
                )

    if verbose:
        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"{'=' * 60}")
        print(f"Total timesteps: {training_info['total_timesteps']:,}")

    return agents, training_info


def _train_single_agent_worker(
    agent_index: int,
    round_idx: int,
    agents_states: List[bytes],
    env_factory: Callable,
    timesteps_per_agent: int,
    save_dir: str,
    N: int,
    seed: int,
) -> Tuple[int, bytes]:
    """
    Worker function to train a single agent in a separate process.

    Args:
        agent_index: Index of agent to train.
        round_idx: Current round index.
        agents_states: Serialized states of all agents.
        env_factory: Function that creates environment.
        timesteps_per_agent: Number of training timesteps.
        save_dir: Directory for saving outputs.
        N: Number of agents.
        seed: Base random seed.

    Returns:
        Tuple of (agent_index, trained_agent_state).
    """
    # Set seeds for reproducibility
    training_seed = seed + round_idx * N + agent_index
    set_all_seeds(training_seed)

    # Deserialize all agents
    # Note: Agents are created with env=None initially. This is okay because:
    # 1. The agent being trained will get its env set immediately below
    # 2. Other agents are only used for their fixed_act_function(), which doesn't need env
    # 3. The env_factory receives the agents list and can handle environment creation
    from easy_marl.core.agents import PPOAgent

    agents = []
    for i, state in enumerate(agents_states):
        agent = PPOAgent.from_bytes(state, env=None)
        agents.append(agent)

    # Create environment and set it for the agent being trained
    env = env_factory(agent_index=agent_index, agents=agents, seed=training_seed)
    agents[agent_index].model.set_env(env)
    agents[agent_index].train(total_timesteps=timesteps_per_agent)

    # Save checkpoint
    if save_dir is not None:
        round_dir = os.path.join(save_dir, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        save_path = os.path.join(round_dir, f"agent_{agent_index}")
        agents[agent_index].save(save_path)

    # Serialize trained agent
    trained_state = agents[agent_index].save_to_bytes()

    return agent_index, trained_state
