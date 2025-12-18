"""Gymnasium-compatible multi-agent electricity market environment."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from easy_marl.examples.bidding.market import market_clearing
from easy_marl.src.observators import OBSERVERS

# penalty per unit of unmet demand, this acts as a system stabilizer
# reduce this to encourage risk-taking agents
UNIT_LOL_PENALTY = 1


class MARLElectricityMarketEnv(gym.Env):
    """Simulates proportional bidding for multiple generators in a day-ahead market."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        agents,
        params=None,
        seed=None,
        agent_index: int = 0,
        observer_name: str = "simple",
    ) -> None:
        super().__init__()
        self.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.agents = agents
        self.fixed_policies = [
            agent.fixed_act_function(deterministic=True) for agent in self.agents
        ]

        # Parameters
        params = params or {}
        self.agent_index = agent_index
        self.N = params["N_generators"]
        self.T = params["T"]
        self.max_bid_delta = params.get("max_bid_delta", 50.0)
        self.lambda_bid_penalty = params.get("lambda_bid_penalty", 0.01)

        # Demand & generator parameters
        self.D_profile = np.array(params["demand_profile"], dtype=np.float32)
        self.base_D_profile = self.D_profile.copy()
        self.K = np.array(params["capacities"], dtype=np.float32)
        self.c = np.array(params["costs"], dtype=np.float32)
        assert len(self.D_profile) == self.T, (
            f"Demand profile length does not match T: {len(self.D_profile)} != {self.T}"
        )
        assert np.all(self.base_D_profile > 0)

        # Scaling
        self.demand_scale = (
            np.max(self.D_profile) if np.max(self.D_profile) > 0 else 1.0
        )
        self.cost_scale = np.max(self.c) if np.max(self.c) > 0 else 1.0
        self.capacity_scale = np.max(self.K) if np.max(self.K) > 0 else 1.0
        self.system_capacity = np.sum(self.K)

        # Observer setup
        obs_dim_fn, obs_fn = OBSERVERS[observer_name]
        self.obs_dim = obs_dim_fn(self.N)
        self._obs_fn = obs_fn  # bind directly (no dict lookup later)
        self.other_mask = np.arange(self.N) != self.agent_index
        self.obs_buf = np.empty(self.obs_dim, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action space
        reasonable_bound = 10
        self.action_space = spaces.Box(
            low=np.array([0.0, -reasonable_bound], dtype=np.float32),
            high=np.array([1.0, reasonable_bound], dtype=np.float32),
        )

        # misc
        self.system_capacity = np.sum(self.K)
        self.lower_stochastic_bound = -0.5 * np.min(self.D_profile)
        self.upper_stochastic_bound = 0.5 * (
            self.system_capacity - np.max(self.D_profile)
        )

        # place to store other agents' fixed action functions:
        # mapping agent_index -> function(obs) -> action (2-vector)
        self.other_action_fns: dict[int, Callable] = {}

        # outputs / internal state
        self.reset(seed=seed)

    def run_stochastics(self) -> None:
        """Apply stochastic perturbations to the demand profile in-place."""

        # generate demand perturbation
        demand_peturbation = self.rng.normal(
            loc=0.0, scale=0.05 * self.demand_scale, size=self.T
        ).astype(np.float32)

        # clip to avoid negative demand
        demand_peturbation = np.clip(
            demand_peturbation,
            a_min=self.lower_stochastic_bound,
            a_max=self.upper_stochastic_bound,
        )

        # update demand profile with stochastic perturbation
        self.D_profile = self.base_D_profile + demand_peturbation

    def reset(self, seed=None):
        """Reset environment state and return the initial observation tuple."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.run_stochastics()

        self.t = 0

        self.output = {
            "bids": np.zeros((self.T, self.N), dtype=np.float32),
            "q_offered": np.zeros((self.T, self.N), dtype=np.float32),
            "q_cleared": np.zeros((self.T, self.N), dtype=np.float32),
            "market_prices": np.zeros(self.T, dtype=np.float32),
            "rewards": np.zeros((self.T, self.N), dtype=np.float32),
            "penalty": np.zeros((self.T, self.N), dtype=np.float32),
        }

        self.b_all = np.zeros(self.N, dtype=np.float32)
        self.q_all = np.zeros(self.N, dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self, agent_index: int | None = None) -> np.ndarray:
        """Return the normalized observation vector for ``agent_index``."""

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

    def update_agent_bid(self, action: np.ndarray, agent_idx: int) -> None:
        """Project an agent's action into quantity and price bids."""

        q_t = float(1 - action[0]) * self.K[agent_idx]
        b_t = float(self.c[agent_idx]) + float(np.tanh(action[1])) * self.max_bid_delta

        self.q_all[agent_idx] = q_t
        self.b_all[agent_idx] = b_t

    def update_all_bids(self, exclude_agent_index: bool = True) -> None:
        """Populate bids for every agent using their fixed policies when available."""

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
    ) -> tuple[np.ndarray | None, float, bool, bool, dict[str, float]]:
        """Advance the environment by one hour of bidding/clearing."""
        # Reset bids/q to default baseline
        # self.q_all[:] = self.K  # default: offer max capacity
        # self.b_all[:] = self.c  # default: bid at cost
        self.q_all[:] = np.full(self.N, np.nan)  # default: error if not set
        self.b_all[:] = np.full(self.N, np.nan)  # default: error if not set

        if fixed_evaluation:
            action = None  # ignore input action, use fixed policies for all agents
            self.update_all_bids(exclude_agent_index=False)
        else:
            # First, fill other agents using provided fixed policies (if any)
            self.update_all_bids(exclude_agent_index=True)

            # Update with current agent action
            self.update_agent_bid(action, self.agent_index)

        # Run market clearing
        P_t, q_cleared = market_clearing(self.b_all, self.q_all, self.D_profile[self.t])

        # Rewards
        base_rewards = (P_t - self.c) * q_cleared

        # Bid regulariser
        bid_penalties = self.lambda_bid_penalty * (self.b_all - self.c) ** 2

        # Penalty for loss of load
        demand = self.D_profile[self.t]
        total_cleared = np.sum(q_cleared)
        loss_of_load_penalty = UNIT_LOL_PENALTY * max(0, demand - total_cleared)

        r = base_rewards - bid_penalties - loss_of_load_penalty

        # Scale reward
        r /= self.demand_scale * self.cost_scale * max(1, self.T)
        r *= 20  # scale to reasonable range

        # Store outputs
        t_idx = self.t
        self.output["bids"][t_idx] = self.b_all.copy()
        self.output["q_offered"][t_idx] = self.q_all.copy()
        self.output["q_cleared"][t_idx] = q_cleared
        self.output["market_prices"][t_idx] = P_t
        self.output["rewards"][t_idx] = r

        # Step time
        self.t += 1
        done = self.t >= self.T
        obs = self._get_obs() if not done else None
        # gymnasium step returns (obs, reward, terminated, truncated, info)
        # keep compatibility: return (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        return obs, r[self.agent_index], terminated, truncated, {}

    def render(self) -> None:
        """Print latest timestep data (for basic debugging only)."""

        t_idx = min(self.t - 1, self.T - 1)
        print(
            f"t={self.t}, Demand={self.D_profile[t_idx]:.2f}, Bid={self.output['bids'][t_idx, self.agent_index]:.2f}"
        )

    def seed(self, seed=None):
        """Seed Gym's RNG utility and return the resulting seed list."""

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_metadata(self) -> dict[str, float]:
        """Return a JSON-serializable dictionary describing the environment."""

        return {
            "N_generators": self.N,
            "T": self.T,
            "capacities": self.K.tolist(),
            "costs": self.c.tolist(),
            "demand_profile": self.D_profile.tolist(),
            "max_bid_delta": self.max_bid_delta,
            "lambda_bid_penalty": self.lambda_bid_penalty,
        }
