"""
Electricity market MARL environment.

This module implements a specific instance of the generic MARL framework
for electricity market bidding simulation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple

from easy_marl.core.environment import MARLEnvironment
from easy_marl.examples.electricity_market.market import market_clearing
from easy_marl.examples.electricity_market.observators import OBSERVERS

# Penalty per unit of unmet demand - acts as a system stabilizer
UNIT_LOL_PENALTY = 1


class ElectricityMarketEnv(MARLEnvironment):
    """
    Electricity market environment for multi-agent bidding simulation.
    
    This environment simulates proportional bidding for multiple generators
    in a day-ahead market. Generators submit quantity-price bids, and the
    market clears based on merit order.
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
        Initialize electricity market environment.
        
        Args:
            agents: List of agent objects.
            params: Environment parameters including:
                - N_generators: Number of generators
                - T: Time horizon (hours)
                - demand_profile: Hourly demand values
                - capacities: Generator capacities
                - costs: Generator costs (SRMC)
                - max_bid_delta: Maximum bid price deviation
                - lambda_bid_penalty: Bid regularization coefficient
            seed: Random seed for reproducibility.
            agent_index: Index of the agent being trained.
            observer_name: Type of observation function to use.
        """
        # Store market-specific parameters before calling super
        params = params or {}
        self.N = params["N_generators"]
        self.T = params["T"]
        self.max_bid_delta = params.get("max_bid_delta", 50.0)
        self.lambda_bid_penalty = params.get("lambda_bid_penalty", 0.01)

        # Demand & generator parameters
        self.D_profile = np.array(params["demand_profile"], dtype=np.float32)
        self.base_D_profile = self.D_profile.copy()
        self.K = np.array(params["capacities"], dtype=np.float32)
        self.c = np.array(params["costs"], dtype=np.float32)
        
        # Validation
        assert len(self.D_profile) == self.T, (
            f"Demand profile length does not match T: {len(self.D_profile)} != {self.T}"
        )
        assert np.all(self.base_D_profile > 0), "Demand must be positive"

        # Scaling factors for normalization
        self.demand_scale = (
            np.max(self.D_profile) if np.max(self.D_profile) > 0 else 1.0
        )
        self.cost_scale = np.max(self.c) if np.max(self.c) > 0 else 1.0
        self.capacity_scale = np.max(self.K) if np.max(self.K) > 0 else 1.0
        self.system_capacity = np.sum(self.K)

        # Observer setup
        obs_dim_fn, obs_fn = OBSERVERS[observer_name]
        self.obs_dim = obs_dim_fn(self.N)
        self._obs_fn = obs_fn
        self.other_mask = np.arange(self.N) != agent_index
        self.obs_buf = np.empty(self.obs_dim, dtype=np.float32)

        # Stochastic bounds for demand perturbation
        self.lower_stochastic_bound = -0.5 * np.min(self.D_profile)
        self.upper_stochastic_bound = 0.5 * (
            self.system_capacity - np.max(self.D_profile)
        )

        # Initialize parent class
        super().__init__(
            agents=agents,
            params=params,
            seed=seed,
            agent_index=agent_index,
        )

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space based on selected observer."""
        return spaces.Box(
            low=0, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def _build_action_space(self) -> spaces.Space:
        """
        Build action space for generator bidding.
        
        Action space is 2D:
        - action[0]: Quantity fraction (0 to 1, where 0 = full capacity)
        - action[1]: Price parameter (-bound to +bound, transformed via tanh)
        """
        reasonable_bound = 10
        return spaces.Box(
            low=np.array([0.0, -reasonable_bound], dtype=np.float32),
            high=np.array([1.0, reasonable_bound], dtype=np.float32),
        )

    def _get_obs(self, agent_index: Optional[int] = None) -> np.ndarray:
        """Get normalized observation for specified agent."""
        if agent_index is None:
            agent_index = self.agent_index

        obs = self._obs_fn(
            self.D_profile,
            self.K,
            self.c,
            agent_index,
            self.demand_scale,
            self.capacity_scale,
            self.cost_scale,
            np.arange(self.N) != agent_index,
            self.obs_buf,
            self.t,
        )
        return obs

    def _compute_reward(
        self,
        agent_index: int,
        actions: Dict[int, np.ndarray],
        state: Dict,
    ) -> float:
        """
        Compute reward for electricity market participation.
        
        Reward components:
        1. Base revenue: (clearing_price - cost) * quantity_cleared
        2. Bid penalty: -lambda * (bid - cost)^2  [regularization]
        3. Loss of load penalty: Shared penalty for unmet demand
        """
        # This is computed in step() for efficiency
        # Return the scaled reward from output
        return self.output["rewards"][self.t - 1, agent_index]

    def _is_terminal(self) -> bool:
        """Episode terminates after T timesteps."""
        return self.t >= self.T

    def run_stochastics(self) -> None:
        """Apply stochastic perturbations to demand profile in-place."""
        # Generate demand perturbation
        demand_perturbation = self.rng.normal(
            loc=0.0, scale=0.05 * self.demand_scale, size=self.T
        ).astype(np.float32)

        # Clip to avoid negative demand or exceeding capacity
        demand_perturbation = np.clip(
            demand_perturbation,
            a_min=self.lower_stochastic_bound,
            a_max=self.upper_stochastic_bound,
        )

        # Update demand profile
        self.D_profile = self.base_D_profile + demand_perturbation

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment state and return initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.run_stochastics()

        self.t = 0

        # Initialize output tracking
        self.output = {
            "bids": np.zeros((self.T, self.N), dtype=np.float32),
            "q_offered": np.zeros((self.T, self.N), dtype=np.float32),
            "q_cleared": np.zeros((self.T, self.N), dtype=np.float32),
            "market_prices": np.zeros(self.T, dtype=np.float32),
            "rewards": np.zeros((self.T, self.N), dtype=np.float32),
            "penalty": np.zeros((self.T, self.N), dtype=np.float32),
        }

        # Current timestep bids and quantities
        self.b_all = np.zeros(self.N, dtype=np.float32)
        self.q_all = np.zeros(self.N, dtype=np.float32)

        return self._get_obs(), {}

    def update_agent_bid(self, action: np.ndarray, agent_idx: int) -> None:
        """
        Convert agent action into quantity and price bids.
        
        Args:
            action: [quantity_fraction, price_parameter]
            agent_idx: Index of the agent submitting the bid
        """
        # Quantity: action[0] = 0 means full capacity, 1 means zero capacity
        q_t = float(1 - action[0]) * self.K[agent_idx]
        
        # Price: cost + tanh(action[1]) * max_bid_delta
        b_t = float(self.c[agent_idx]) + float(np.tanh(action[1])) * self.max_bid_delta

        self.q_all[agent_idx] = q_t
        self.b_all[agent_idx] = b_t

    def update_all_bids(self, exclude_agent_index: bool = True) -> None:
        """
        Populate bids for all agents using their fixed policies.
        
        Args:
            exclude_agent_index: If True, skip the agent being trained.
        """
        for j in range(self.N):
            if exclude_agent_index and j == self.agent_index:
                continue
            
            fn = self.fixed_policies[j]
            if fn is not None:
                obs_j = self._get_obs(agent_index=j)
                act_j = fn(obs_j)
                act_j = np.asarray(act_j, dtype=np.float32)
                
                if act_j.size >= 2:
                    self.update_agent_bid(act_j, j)
                else:
                    raise ValueError(
                        f"Action function for agent {j} returned invalid action of size {act_j.size}."
                    )

    def step(
        self, action, fixed_evaluation: bool = False
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict]:
        """
        Execute one timestep of market clearing.
        
        Args:
            action: Action from the agent being trained (or None if fixed_evaluation).
            fixed_evaluation: If True, use fixed policies for all agents.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Reset bids/quantities
        self.q_all[:] = np.full(self.N, np.nan)
        self.b_all[:] = np.full(self.N, np.nan)

        if fixed_evaluation:
            # Use fixed policies for all agents
            action = None
            self.update_all_bids(exclude_agent_index=False)
        else:
            # Fill other agents' bids, then apply training agent's action
            self.update_all_bids(exclude_agent_index=True)
            self.update_agent_bid(action, self.agent_index)

        # Run market clearing
        P_t, q_cleared = market_clearing(
            self.b_all, self.q_all, self.D_profile[self.t]
        )

        # Compute rewards
        base_rewards = (P_t - self.c) * q_cleared

        # Bid regularization penalty
        bid_penalties = self.lambda_bid_penalty * (self.b_all - self.c) ** 2

        # Loss of load penalty (shared across all agents)
        demand = self.D_profile[self.t]
        total_cleared = np.sum(q_cleared)
        loss_of_load_penalty = UNIT_LOL_PENALTY * max(0, demand - total_cleared)

        r = base_rewards - bid_penalties - loss_of_load_penalty

        # Scale reward to reasonable range
        r /= self.demand_scale * self.cost_scale * max(1, self.T)
        r *= 20

        # Store outputs
        t_idx = self.t
        self.output["bids"][t_idx] = self.b_all.copy()
        self.output["q_offered"][t_idx] = self.q_all.copy()
        self.output["q_cleared"][t_idx] = q_cleared
        self.output["market_prices"][t_idx] = P_t
        self.output["rewards"][t_idx] = r

        # Advance time
        self.t += 1
        done = self._is_terminal()
        obs = self._get_obs() if not done else None

        terminated = done
        truncated = False
        
        return obs, r[self.agent_index], terminated, truncated, {}

    def render(self) -> None:
        """Print latest timestep data for debugging."""
        t_idx = min(self.t - 1, self.T - 1)
        print(
            f"t={self.t}, Demand={self.D_profile[t_idx]:.2f}, "
            f"Bid={self.output['bids'][t_idx, self.agent_index]:.2f}"
        )

    def get_metadata(self) -> Dict[str, float]:
        """Return environment metadata as dictionary."""
        return {
            "N_generators": self.N,
            "T": self.T,
            "capacities": self.K.tolist(),
            "costs": self.c.tolist(),
            "demand_profile": self.D_profile.tolist(),
            "max_bid_delta": self.max_bid_delta,
            "lambda_bid_penalty": self.lambda_bid_penalty,
        }
