"""
Generic multi-agent reinforcement learning environment base class.

This module provides a base class for MARL environments that follow the
Gymnasium API. Domain-specific environments should inherit from this class
and implement the required methods.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from abc import abstractmethod


class MARLEnvironment(gym.Env):
    """
    Base class for multi-agent reinforcement learning environments.
    
    This class provides the common structure for MARL environments where:
    - Multiple agents interact in a shared environment
    - Each agent observes the environment from its own perspective
    - Agents can have fixed or learning policies
    - Training can be done in sequential or parallel modes
    
    Subclasses should implement:
    - _get_obs(): Build observation for an agent
    - _compute_reward(): Calculate reward for an agent's action
    - _is_terminal(): Check if episode should end
    - _build_observation_space(): Define the observation space
    - _build_action_space(): Define the action space
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        agents: List,
        params: Dict,
        seed: Optional[int] = None,
        agent_index: int = 0,
    ) -> None:
        """
        Initialize MARL environment.
        
        Args:
            agents: List of agent objects in the environment.
            params: Dictionary of environment parameters.
            seed: Random seed for reproducibility.
            agent_index: Index of the agent being trained (perspective).
        """
        super().__init__()
        self.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.agents = agents
        self.agent_index = agent_index
        self.params = params
        
        # Store fixed policies for other agents
        self.fixed_policies = [
            agent.fixed_act_function(deterministic=True) if agent is not None else None
            for agent in self.agents
        ]
        
        # Build spaces (subclasses should implement these)
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()
        
        # Internal state
        self.reset(seed=seed)

    @abstractmethod
    def _build_observation_space(self) -> spaces.Space:
        """
        Build and return the observation space for agents.
        
        Returns:
            Gymnasium space object defining valid observations.
        """
        pass

    @abstractmethod
    def _build_action_space(self) -> spaces.Space:
        """
        Build and return the action space for agents.
        
        Returns:
            Gymnasium space object defining valid actions.
        """
        pass

    @abstractmethod
    def _get_obs(self, agent_index: Optional[int] = None) -> np.ndarray:
        """
        Get observation for a specific agent.
        
        Args:
            agent_index: Index of agent to get observation for.
                        If None, uses self.agent_index.
        
        Returns:
            Observation array for the specified agent.
        """
        pass

    @abstractmethod
    def _compute_reward(
        self,
        agent_index: int,
        actions: Dict[int, np.ndarray],
        state: Dict[str, Any],
    ) -> float:
        """
        Compute reward for an agent given all actions and current state.
        
        Args:
            agent_index: Index of agent to compute reward for.
            actions: Dictionary mapping agent indices to their actions.
            state: Current environment state.
        
        Returns:
            Reward value for the agent.
        """
        pass

    @abstractmethod
    def _is_terminal(self) -> bool:
        """
        Check if the episode should terminate.
        
        Returns:
            True if episode is done, False otherwise.
        """
        pass

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Optional random seed.
        
        Returns:
            Tuple of (initial_observation, info_dict).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Subclasses should override this to reset their specific state
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action taken by the agent being trained.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # This is a template - subclasses should implement specific logic
        raise NotImplementedError("Subclasses must implement step()")

    def render(self) -> None:
        """Render the environment (optional)."""
        pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value.
        
        Returns:
            List containing the seed value.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_metadata(self) -> Dict:
        """
        Return environment metadata as a dictionary.
        
        Returns:
            Dictionary with environment parameters and configuration.
        """
        return self.params.copy()
