"""
Training utilities for Snake MARL.

This module provides training functions for the Snake game,
built on top of the generic MARL framework.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple

from easy_marl.core.agents import PPOAgent
from easy_marl.examples.snake.environment import SnakeEnv


# Default configuration
DEFAULT_OBS = "simple"

# PPO hyperparameters tuned for Snake game
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64,
    "hidden_sizes": (64, 64),
    "ent_coef": 0.01,
}


def make_default_params(board_size: int = 15) -> Dict:
    """
    Create default parameters for Snake environment.
    
    Args:
        board_size: Size of the square board.
        
    Returns:
        Dictionary of environment parameters.
    """
    return {
        "board_size": board_size,
    }


def init_agent(
    params: Dict,
    seed: int,
    observer_name: str = DEFAULT_OBS,
) -> PPOAgent:
    """
    Initialize a single agent for Snake game.
    
    Args:
        params: Environment parameters.
        seed: Random seed.
        observer_name: Type of observation function.
        
    Returns:
        Initialized PPOAgent object.
    """
    env = SnakeEnv(
        agents=[],
        params=params,
        seed=seed,
        agent_index=0,
        observer_name=observer_name,
    )
    agent = PPOAgent(
        env=env,
        seed=seed,
        **PPO_PARAMS,
    )
    return agent


def evaluate_agent(
    agent: PPOAgent,
    params: Dict,
    num_episodes: int = 10,
    seed: int = None,
    observer_name: str = DEFAULT_OBS,
) -> Dict:
    """
    Evaluate a trained agent over multiple episodes.
    
    Args:
        agent: Trained agent object.
        params: Environment parameters.
        num_episodes: Number of evaluation episodes.
        seed: Random seed.
        observer_name: Type of observation function.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    # Storage for evaluation metrics
    episode_rewards = []
    episode_lengths = []
    food_collected = []
    max_snake_lengths = []
    
    for ep in range(num_episodes):
        # Create evaluation environment
        eval_env = SnakeEnv(
            agents=[agent],
            params=params,
            seed=seed + ep if seed else None,
            agent_index=0,
            observer_name=observer_name,
        )
        
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        max_snake_length = 0
        
        while not done:
            # Get action from agent (deterministic for evaluation)
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            max_snake_length = max(max_snake_length, info.get("snake_length", 0))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        food_collected.append(eval_env.board.food_points)
        max_snake_lengths.append(max_snake_length)
    
    # Aggregate results
    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_food": float(np.mean(food_collected)),
        "std_food": float(np.std(food_collected)),
        "mean_max_snake_length": float(np.mean(max_snake_lengths)),
        "max_food": int(np.max(food_collected)),
        "max_length": int(np.max(episode_lengths)),
    }
    
    return results


def train(
    timesteps: int = 100000,
    seed: int = 42,
    save_dir: str = "outputs/snake",
    verbose: bool = True,
    board_size: int = 15,
    observer_name: str = DEFAULT_OBS,
    eval_episodes: int = 10,
) -> Tuple[PPOAgent, Dict]:
    """
    Train a single agent to play Snake.
    
    Args:
        timesteps: Total training timesteps.
        seed: Random seed.
        save_dir: Directory to save models and metrics.
        verbose: Print training progress.
        board_size: Size of the game board.
        observer_name: Type of observation function.
        eval_episodes: Number of episodes for evaluation.
        
    Returns:
        agent: Trained agent.
        training_info: Training metadata dictionary.
    """
    if verbose:
        print("=" * 50)
        print("Snake Single-Agent Training")
        print("=" * 50)
        print(f"Board size: {board_size}x{board_size}")
        print(f"Total timesteps: {timesteps}")
        print(f"Random seed: {seed}")
        print("=" * 50)
    
    # Generate parameters
    params = make_default_params(board_size=board_size)
    
    # Initialize agent
    agent = init_agent(params, seed, observer_name)
    
    # Train agent
    if verbose:
        print("\nTraining agent...")
    
    agent.train(total_timesteps=timesteps)
    
    if verbose:
        print("Training complete!")
    
    # Evaluate agent
    if verbose:
        print(f"\nEvaluating agent over {eval_episodes} episodes...")
    
    eval_results = evaluate_agent(
        agent, params, num_episodes=eval_episodes, seed=seed, observer_name=observer_name
    )
    
    if verbose:
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
        print(f"  Mean Food Collected: {eval_results['mean_food']:.1f} ± {eval_results['std_food']:.1f}")
        print(f"  Max Food Collected: {eval_results['max_food']}")
        print(f"  Mean Max Snake Length: {eval_results['mean_max_snake_length']:.1f}")
    
    # Save model and results
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "snake_agent.zip")
        agent.model.save(model_path)
        
        if verbose:
            print(f"\nModel saved to: {model_path}")
        
        # Save training info
        training_info = {
            "params": params,
            "timesteps": timesteps,
            "seed": seed,
            "board_size": board_size,
            "observer_name": observer_name,
            "evaluation": eval_results,
        }
        
        info_path = os.path.join(save_dir, "training_info.json")
        with open(info_path, "w") as f:
            json.dump(training_info, f, indent=4)
        
        if verbose:
            print(f"Training info saved to: {info_path}")
    else:
        training_info = {
            "params": params,
            "timesteps": timesteps,
            "seed": seed,
            "evaluation": eval_results,
        }
    
    return agent, training_info
