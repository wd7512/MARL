"""
Snake game MARL environment.

This module implements a single-agent Snake game using the generic MARL framework.
For now, this is a one-agent game, but the structure allows for future multi-agent
extensions.
"""

from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any

from easy_marl.core.environment import MARLEnvironment
from easy_marl.examples.snake.engine import SnakeBoard
from easy_marl.examples.snake.observators import OBSERVERS


class SnakeEnv(MARLEnvironment):
    """
    Snake game environment for reinforcement learning.
    
    This is a single-agent game where the agent controls a snake that must:
    - Eat food to grow and gain points
    - Avoid walls and its own body
    - Survive as long as possible
    """

    def __init__(
        self,
        agents,
        params=None,
        seed=None,
        agent_index: int = 0,
        observer_name: str = "simple",
    ) -> None:
        """
        Initialize Snake environment.
        
        Args:
            agents: List of agent objects (single agent for now).
            params: Environment parameters including:
                - board_size: Size of the square board (default: 15)
            seed: Random seed for reproducibility.
            agent_index: Index of the agent being trained (always 0 for single agent).
            observer_name: Type of observation function to use.
        """
        # Store snake-specific parameters
        params = params or {}
        self.board_size = params.get("board_size", 15)
        
        # Observer setup
        obs_dim_fn, obs_fn = OBSERVERS[observer_name]
        self.obs_dim = obs_dim_fn()
        self._obs_fn = obs_fn
        
        # Initialize parent class
        super().__init__(
            agents=agents,
            params=params,
            seed=seed,
            agent_index=agent_index,
        )

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space based on selected observer."""
        return spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Space:
        """
        Build action space for snake movement.
        
        Action space is discrete with 4 actions:
        - 0: Left
        - 1: Up
        - 2: Right
        - 3: Down
        """
        return spaces.Discrete(4)

    def _get_obs(self, agent_index: Optional[int] = None) -> np.ndarray:
        """Get observation from current board state."""
        return self._obs_fn(self.board)

    def _compute_reward(
        self,
        agent_index: int,
        actions: Dict[int, np.ndarray],
        state: Dict[str, Any],
    ) -> float:
        """
        Compute reward for the agent's action.
        
        Reward structure:
        - Large positive reward for eating food
        - Small negative reward for each move (encourages efficiency)
        - Large negative reward for dying
        """
        # This is computed in step() for efficiency
        return self.last_reward

    def _is_terminal(self) -> bool:
        """Episode terminates when the game ends."""
        return self.board.end

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Create new board
        self.board = SnakeBoard(size=self.board_size, seed=seed)
        
        # Reset reward tracking
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.steps = 0
        
        return self._get_obs(), {}

    def step(
        self, action, fixed_evaluation: bool = False
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict]:
        """
        Execute one timestep in the game.
        
        Args:
            action: Action from the agent (0=Left, 1=Up, 2=Right, 3=Down).
            fixed_evaluation: If True, use fixed policies for evaluation.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Convert action to integer if it's an array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif not isinstance(action, int):
            action = int(action)
        
        # Store previous state
        prev_food_points = self.board.food_points
        
        # Execute move
        self.board.push(action)
        self.steps += 1
        
        # Calculate reward
        reward = 0.0
        
        # Reward for eating food
        if self.board.food_points > prev_food_points:
            reward += 10.0  # Large positive reward for food
        
        # Small penalty for each move to encourage efficiency
        reward -= 0.01
        
        # Large penalty for dying
        if self.board.end:
            reward -= 1.0
        
        self.last_reward = reward
        self.total_reward += reward
        
        # Get next observation
        done = self._is_terminal()
        obs = self._get_obs() if not done else None
        
        terminated = done
        truncated = False
        
        info = {
            "food_points": self.board.food_points,
            "move_points": self.board.move_points,
            "energy": self.board.energy,
            "snake_length": len(self.board.body_list) + 1,  # body + head
        }
        
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Print current board state for debugging."""
        print(f"\nStep: {self.steps}, Score: {self.board.food_points}")
        print(self.board)

    def get_metadata(self) -> Dict:
        """Return environment metadata as dictionary."""
        return {
            "board_size": self.board_size,
            "observation_space": str(self.observation_space),
            "action_space": str(self.action_space),
        }
