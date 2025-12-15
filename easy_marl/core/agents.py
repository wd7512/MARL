"""
Generic agent implementations for multi-agent reinforcement learning.

This module provides base agent classes and implementations that can be used
across different MARL domains.
"""

import torch
from abc import ABC, abstractmethod
from stable_baselines3 import PPO
from typing import Callable, Any
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all agents in a MARL system."""

    def __init__(self, env=None):
        """
        Initialize base agent.
        
        Args:
            env: Optional environment instance for the agent.
        """
        self.env = env

    @abstractmethod
    def act(self, obs: Any, deterministic: bool = True) -> Any:
        """
        Return an action given an observation.
        
        Args:
            obs: Observation from the environment.
            deterministic: Whether to use deterministic policy.
            
        Returns:
            Action to take in the environment.
        """
        pass

    @abstractmethod
    def fixed_act_function(self, deterministic: bool = True) -> Callable:
        """
        Return a frozen action function for use by other agents.
        
        This method should return a function that captures the current policy
        and won't change even if the agent is trained further.
        
        Args:
            deterministic: Whether the returned function should be deterministic.
            
        Returns:
            A callable that maps observation to action.
        """
        pass

    def train(self, *args, **kwargs):
        """
        Train the agent (optional).
        
        Override if the agent supports learning.
        
        Raises:
            NotImplementedError: If the agent type does not support training.
        """
        raise NotImplementedError("This agent type does not support training.")


class SimpleAgent(BaseAgent):
    """
    Simple baseline agent that returns a default action.
    
    This agent is useful for testing and as a baseline opponent.
    By default, it returns a null action [0.0, 0.0].
    """

    def __init__(self, env=None, default_action=None):
        """
        Initialize simple agent.
        
        Args:
            env: Optional environment instance.
            default_action: Action to always return. Defaults to [0.0, 0.0].
        """
        super().__init__(env)
        self.default_action = default_action if default_action is not None else [0.0, 0.0]

    def act(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        """
        Return the default action.
        
        Args:
            obs: Observation (ignored).
            deterministic: Whether to use deterministic policy (ignored).
            
        Returns:
            The default action.
        """
        return np.array(self.default_action, dtype=np.float32)

    def fixed_act_function(self, deterministic: bool = True) -> Callable:
        """
        Return a frozen action function.
        
        The function captures the current default_action by value,
        so changes to self.default_action won't affect it.
        
        Args:
            deterministic: Whether the function should be deterministic (ignored).
            
        Returns:
            A function that always returns the captured action.
        """
        # Capture current action by value
        frozen_action = np.array(self.default_action, dtype=np.float32).copy()

        def act_fn(obs):
            return frozen_action

        return act_fn


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent for reinforcement learning.
    
    This agent uses a neural network policy to learn optimal strategies.
    It can be used in any MARL environment that follows the Gymnasium API.
    """

    def __init__(
        self,
        env,
        seed: int = 42,
        learning_rate: float = 3e-4,
        gamma: float = 0.98,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        batch_size: int = 64,
        hidden_sizes: tuple = (64, 64),
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Initialize PPO agent with configurable hyperparameters.

        Args:
            env: Gymnasium environment instance.
            seed: Random seed for reproducibility.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor for future rewards.
            clip_range: PPO clipping parameter.
            n_steps: Number of steps to collect before update.
            batch_size: Minibatch size for updates.
            hidden_sizes: Tuple of hidden layer sizes for policy network.
            weight_decay: L2 regularization coefficient.
            **kwargs: Additional arguments passed to PPO constructor.
        """
        super().__init__(env)

        # Neural network architecture
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=list(hidden_sizes), vf=list(hidden_sizes)),
        )

        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
            gamma=gamma,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=batch_size,
            seed=seed,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )

        # Add L2 regularization if specified
        if weight_decay > 0:
            self.model.policy.optimizer = torch.optim.Adam(
                self.model.policy.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

    def train(
        self,
        total_timesteps: int = 50000,
        checkpoint_path: str = None,
        checkpoint_freq: int = 1000,
        callbacks: list = None,
    ):
        """
        Train the agent using PPO algorithm.

        Args:
            total_timesteps: Total number of environment steps.
            checkpoint_path: Directory to save checkpoints (optional).
            checkpoint_freq: Frequency of checkpoint saves.
            callbacks: List of callback objects.
        """
        from stable_baselines3.common.callbacks import CheckpointCallback

        callback_list = callbacks if callbacks is not None else []

        if checkpoint_path is not None:
            checkpoint_cb = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_path,
                name_prefix="ppo_agent",
            )
            callback_list.append(checkpoint_cb)

        self.model.learn(total_timesteps=total_timesteps, callback=callback_list)

    def act(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        """
        Return action for given observation.

        Args:
            obs: Environment observation.
            deterministic: If True, use deterministic policy.

        Returns:
            Action to take in the environment.
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def fixed_act_function(self, deterministic: bool = True) -> Callable:
        """
        Return a frozen action function for use by other agents.
        
        This method creates a snapshot of the current policy by serializing
        the model state. The returned function will not change even if the
        agent is trained further.

        Args:
            deterministic: If True, the function uses deterministic policy.

        Returns:
            Callable that maps observation to action.
        """
        # Serialize the current model state
        model_state = self.save_to_bytes()
        
        def act_fn(obs):
            # Create a temporary agent with the frozen model state
            # Note: This is less efficient but ensures true freezing
            # For better performance, consider caching the deserialized model
            import io
            buffer = io.BytesIO(model_state)
            temp_model = PPO.load(buffer)
            action, _ = temp_model.predict(obs, deterministic=deterministic)
            return action

        return act_fn

    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path where model should be saved.
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to saved model.
        """
        self.model = PPO.load(path, env=self.env)

    def save_to_bytes(self) -> bytes:
        """
        Serialize agent state to bytes.

        Use with load_from_bytes() to mutate an existing instance,
        or with from_bytes() to create a new instance.

        Returns:
            Serialized model state including weights, optimizer, and RNG.
        """
        import io

        buffer = io.BytesIO()
        self.model.save(buffer)
        return buffer.getvalue()

    def load_from_bytes(self, data: bytes):
        """
        Load serialized state into this existing instance (in-place mutation).

        Use this when you have a live agent that you want to restore to a
        previous state, e.g., restoring opponent snapshots in SBR training.

        Args:
            data: Bytes from save_to_bytes().
        """
        import io

        buffer = io.BytesIO(data)
        self.model = PPO.load(buffer, env=self.env)

    @classmethod
    def from_bytes(cls, data: bytes, env) -> "PPOAgent":
        """
        Create a new agent instance directly from serialized state.

        Use this instead of __init__ + load_from_bytes when creating agents
        in worker processes. This avoids the overhead and potential RNG
        perturbation of initializing a random model that gets immediately
        discarded.

        Args:
            data: Bytes from save_to_bytes().
            env: Environment instance for the agent.

        Returns:
            New PPOAgent instance with the deserialized model.
        """
        import io

        instance = cls.__new__(cls)
        instance.env = env

        buffer = io.BytesIO(data)
        instance.model = PPO.load(buffer, env=env)

        return instance
