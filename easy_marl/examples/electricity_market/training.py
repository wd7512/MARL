"""
Training utilities for electricity market MARL.

This module provides training functions specific to the electricity market
domain, built on top of the generic training loops in easy_marl.core.training.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Callable

from easy_marl.core.agents import PPOAgent
from easy_marl.core.training import (
    sequential_train as core_sequential_train,
    parallel_train as core_parallel_train,
    SAFE_MAX_ENV_PROCS,
)
from easy_marl.examples.electricity_market.environment import ElectricityMarketEnv
from easy_marl.examples.electricity_market.params import make_default_params

# Default configuration
DEFAULT_OBS = "simple_v3"
DEFAULT_UPDATE_PROB = 0.75

# PPO hyperparameters tuned for electricity market
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 24,  # Match episode length
    "batch_size": 24,
    "hidden_sizes": (32, 32, 32, 32),
    "ent_coef": 0.01,
}


def init_agents(
    N: int,
    params: Dict,
    seed: int,
    observer_name: str = DEFAULT_OBS,
) -> List[PPOAgent]:
    """
    Initialize N agents for electricity market.

    Args:
        N: Number of agents.
        params: Environment parameters.
        seed: Random seed.
        observer_name: Type of observation function.

    Returns:
        List of initialized PPOAgent objects.
    """
    agents = []
    for i in range(N):
        env = ElectricityMarketEnv(
            agents=[],
            params=params,
            seed=seed,
            agent_index=i,
            observer_name=observer_name,
        )
        agents.append(
            PPOAgent(
                env=env,
                seed=seed + i,
                **PPO_PARAMS,
            )
        )
    return agents


def make_env_for_agent(
    agent_index: int,
    agents: List,
    params: Dict,
    seed: int,
    observer_name: str = DEFAULT_OBS,
) -> ElectricityMarketEnv:
    """
    Create environment instance for a specific agent.

    Args:
        agent_index: Index of the agent to train.
        agents: List of all agent objects.
        params: Environment parameters.
        seed: Random seed.
        observer_name: Type of observation function.

    Returns:
        ElectricityMarketEnv configured for the specified agent.
    """
    return ElectricityMarketEnv(
        agents=agents,
        params=params,
        seed=seed,
        agent_index=agent_index,
        observer_name=observer_name,
    )


def convert_np_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_np_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def evaluate_agents(
    agents: List,
    params: Dict,
    num_episodes: int = 1,
    seed: int = None,
    observer_name: str = DEFAULT_OBS,
) -> Dict:
    """
    Evaluate trained agents over multiple episodes.

    Args:
        agents: List of trained agent objects.
        params: Environment parameters.
        num_episodes: Number of evaluation episodes.
        seed: Random seed (only for stochastic environments).
        observer_name: Type of observation function.

    Returns:
        Dictionary containing evaluation metrics.
    """
    N = len(agents)

    # Storage for evaluation metrics
    episode_rewards = np.zeros((num_episodes, N))
    episode_profits = np.zeros((num_episodes, N))
    episode_quantities = np.zeros((num_episodes, N))
    market_prices = []

    for ep in range(num_episodes):
        # Create evaluation environment
        eval_env = ElectricityMarketEnv(
            agents=agents,
            params=params,
            seed=seed + ep if seed else None,
            observer_name=observer_name,
        )

        obs, _ = eval_env.reset()
        done = False

        while not done:
            # Use fixed policies for all agents
            obs, reward, terminated, truncated, _ = eval_env.step(
                None, fixed_evaluation=True
            )
            done = terminated or truncated

        # Extract metrics
        output = eval_env.output

        for i in range(N):
            total_reward = np.sum(output["rewards"][:, i])
            total_profit = np.sum(
                (output["market_prices"] - params["costs"][i])
                * output["q_cleared"][:, i]
            )
            total_quantity = np.sum(output["q_cleared"][:, i])

            episode_rewards[ep, i] = total_reward
            episode_profits[ep, i] = total_profit
            episode_quantities[ep, i] = total_quantity

        market_prices.append(output["market_prices"])

    # Aggregate results
    results = {
        "mean_rewards": np.mean(episode_rewards, axis=0),
        "std_rewards": np.std(episode_rewards, axis=0),
        "mean_profits": np.mean(episode_profits, axis=0),
        "std_profits": np.std(episode_profits, axis=0),
        "mean_quantities": np.mean(episode_quantities, axis=0),
        "mean_market_price": np.mean(market_prices),
        "std_market_price": np.std(market_prices),
        "mean_market_prices_over_time": np.mean(market_prices, axis=0),
    }

    return results


def sequential_train(
    N: int = 5,
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    param_func: Callable = make_default_params,
    update_schedule: str = "SBR",
    T: int = 24,
    observer_name: str = DEFAULT_OBS,
    update_probability: float = DEFAULT_UPDATE_PROB,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Sequential training for electricity market agents.

    This is a wrapper around the generic sequential_train that provides
    electricity market-specific configuration and evaluation.

    Args:
        N: Number of generators.
        num_rounds: Number of training rounds.
        timesteps_per_agent: PPO timesteps per agent per round.
        seed: Random seed.
        save_dir: Directory to save models and metrics.
        verbose: Print training progress.
        param_func: Function to generate environment parameters.
        update_schedule: "SBR" or "IBR".
        T: Time horizon (hours).
        observer_name: Type of observation function.
        update_probability: Probability of agent update per round.

    Returns:
        agents: List of trained agents.
        training_info: Training metadata dictionary.
    """
    # Generate parameters
    params = param_func(N=N, T=T)

    # Initialize agents
    agents = init_agents(N, params, seed, observer_name)

    # Create environment factory
    def env_factory(agent_index: int, agents: List, seed: int):
        return make_env_for_agent(agent_index, agents, params, seed, observer_name)

    # Create evaluation function
    def evaluation_fn(agents: List) -> Dict:
        metrics = evaluate_agents(
            agents, params, num_episodes=1, seed=seed, observer_name=observer_name
        )
        return convert_np_to_python(metrics)

    # Call generic training
    agents, training_info = core_sequential_train(
        agents=agents,
        env_factory=env_factory,
        num_rounds=num_rounds,
        timesteps_per_agent=timesteps_per_agent,
        seed=seed,
        save_dir=save_dir,
        verbose=verbose,
        update_schedule=update_schedule,
        update_probability=update_probability,
        evaluation_fn=evaluation_fn,
    )

    # Save training summary
    if save_dir is not None:
        summary_path = os.path.join(save_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(training_info, f, indent=4)

    return agents, training_info


def parallel_train(
    N: int = 5,
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    param_func: Callable = make_default_params,
    n_workers: int = SAFE_MAX_ENV_PROCS,
    T: int = 24,
    observer_name: str = DEFAULT_OBS,
    update_probability: float = DEFAULT_UPDATE_PROB,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Parallel training for electricity market agents.

    This is a wrapper around the generic parallel_train that provides
    electricity market-specific configuration and evaluation.

    Args:
        N: Number of generators.
        num_rounds: Number of training rounds.
        timesteps_per_agent: PPO timesteps per agent per round.
        seed: Random seed.
        save_dir: Directory to save models and metrics.
        verbose: Print training progress.
        param_func: Function to generate environment parameters.
        n_workers: Number of parallel processes.
        T: Time horizon (hours).
        observer_name: Type of observation function.
        update_probability: Probability of agent update per round.

    Returns:
        agents: List of trained agents.
        training_info: Training metadata dictionary.
    """
    # Generate parameters
    params = param_func(N=N, T=T)

    # Initialize agents
    agents = init_agents(N, params, seed, observer_name)

    # Create environment factory
    def env_factory(agent_index: int, agents: List, seed: int):
        return make_env_for_agent(agent_index, agents, params, seed, observer_name)

    # Create evaluation function
    def evaluation_fn(agents: List) -> Dict:
        metrics = evaluate_agents(
            agents, params, num_episodes=1, seed=seed, observer_name=observer_name
        )
        return convert_np_to_python(metrics)

    # Call generic training
    agents, training_info = core_parallel_train(
        agents=agents,
        env_factory=env_factory,
        num_rounds=num_rounds,
        timesteps_per_agent=timesteps_per_agent,
        seed=seed,
        save_dir=save_dir,
        verbose=verbose,
        n_workers=n_workers,
        update_probability=update_probability,
        evaluation_fn=evaluation_fn,
    )

    # Save training summary
    if save_dir is not None:
        summary_path = os.path.join(save_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(training_info, f, indent=4)

    return agents, training_info


def auto_train(
    N: int = 5,
    num_rounds: int = 3,
    timesteps_per_agent: int = 24 * 4,
    seed: int = 42,
    save_dir: str = "outputs",
    verbose: bool = True,
    param_func: Callable = make_default_params,
    T: int = 24,
    observer_name: str = DEFAULT_OBS,
    update_probability: float = DEFAULT_UPDATE_PROB,
    **kwargs,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Automatically choose between sequential and parallel training.

    Selection criteria:
    - Use sequential for small problems (N < 2 or timesteps < 4000)
    - Use parallel for larger problems to leverage multi-core CPUs

    Args:
        N: Number of generators.
        num_rounds: Number of training rounds.
        timesteps_per_agent: PPO timesteps per agent per round.
        seed: Random seed.
        save_dir: Directory to save models and metrics.
        verbose: Print training progress.
        param_func: Function to generate environment parameters.
        T: Time horizon (hours).
        observer_name: Type of observation function.
        update_probability: Probability of agent update per round.
        **kwargs: Additional arguments passed to parallel_train.

    Returns:
        agents: List of trained agents.
        training_info: Training metadata dictionary.
    """
    if N < 2:
        if verbose:
            print("Auto training: N < 2, using sequential training")
        return sequential_train(
            N,
            num_rounds,
            timesteps_per_agent,
            seed,
            save_dir,
            verbose,
            param_func,
            T=T,
            observer_name=observer_name,
            update_probability=update_probability,
        )

    if timesteps_per_agent < 4000:
        if verbose:
            print(
                "Auto training: timesteps_per_agent < 4000, using sequential training"
            )
        return sequential_train(
            N,
            num_rounds,
            timesteps_per_agent,
            seed,
            save_dir,
            verbose,
            param_func,
            T=T,
            observer_name=observer_name,
            update_probability=update_probability,
        )
    else:
        if verbose:
            print("Auto training: timesteps_per_agent >= 4000, using parallel training")
        return parallel_train(
            N,
            num_rounds,
            timesteps_per_agent,
            seed,
            save_dir,
            verbose,
            param_func,
            T=T,
            observer_name=observer_name,
            update_probability=update_probability,
            **kwargs,
        )
