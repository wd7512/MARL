"""Utility observers that project MARL environment state into agent features."""

import numpy as np


def basic_observer(
    D_profile: np.ndarray,
    K: np.ndarray,
    c: np.ndarray,
    agent_index: int,
    demand_scale: float,
    capacity_scale: float,
    cost_scale: float,
    mask: np.ndarray,
    obs_buf: np.ndarray,
    t: int,
) -> np.ndarray:
    """Write a structured observation containing peer costs and capacities."""

    obs_buf[0] = D_profile[t] / demand_scale
    obs_buf[1] = K[agent_index] / capacity_scale
    n_other = len(K) - 1
    obs_buf[2 : 2 + n_other] = c[mask] / cost_scale
    obs_buf[2 + n_other :] = K[mask] / capacity_scale
    return obs_buf


def basic_observer_dim(N: int) -> int:
    """Return the feature dimension for ``basic_observer``."""

    return 2 + (N - 1) * 2


def simple_observer(
    D_profile: np.ndarray,
    K: np.ndarray,
    c: np.ndarray,
    agent_index: int,
    demand_scale: float,
    capacity_scale: float,
    cost_scale: float,
    mask: np.ndarray,
    obs_buf: np.ndarray,
    t: int,
) -> np.ndarray:
    """Write a compact observation with demand, own stats, and hour of day."""

    obs_buf[0] = D_profile[t] / demand_scale  # demand value
    obs_buf[1] = K[agent_index] / capacity_scale  # own capacity ratio
    obs_buf[2] = c[agent_index] / cost_scale  # own cost ratio
    obs_buf[3] = (t % 24) / 24  # hour of day
    return obs_buf


def simple_observer_dim(N: int) -> int:
    """Return the feature dimension for ``simple_observer`` (always 4)."""

    return 4


def simple_observer_v2(
    D_profile: np.ndarray,
    K: np.ndarray,
    c: np.ndarray,
    agent_index: int,
    demand_scale: float,
    capacity_scale: float,
    cost_scale: float,
    mask: np.ndarray,
    obs_buf: np.ndarray,
    t: int,
) -> np.ndarray:
    """Write a compact observation with demand, own stats, and hour of day."""

    obs_buf[0] = D_profile[t] / demand_scale  # demand value
    obs_buf[1] = K[agent_index] / capacity_scale  # own capacity ratio
    obs_buf[2] = c[agent_index] / cost_scale  # own cost ratio
    obs_buf[3] = (t % 24) / 24  # hour of day
    obs_buf[4] = ((t // 24) % 7) / 7  # day of week
    return obs_buf


def simple_observer_v2_dim(N: int) -> int:
    """Return the feature dimension for ``simple_observer_v2`` (always 5)."""

    return 5


def simple_observer_v3(
    D_profile: np.ndarray,
    K: np.ndarray,
    c: np.ndarray,
    agent_index: int,
    demand_scale: float,
    capacity_scale: float,
    cost_scale: float,
    mask: np.ndarray,
    obs_buf: np.ndarray,
    t: int,
) -> np.ndarray:
    """
    Write a compact observation with
     - demand,
     - own stats,
     - hour of day,
     - day of week
    """
    hour_of_day = t % 24
    hour_x = np.sin(2 * np.pi * hour_of_day / 24)
    hour_y = np.cos(2 * np.pi * hour_of_day / 24)

    day_of_week = (t // 24) % 7
    day_x = np.sin(2 * np.pi * day_of_week / 7)
    day_y = np.cos(2 * np.pi * day_of_week / 7)

    obs_buf[0] = D_profile[t] / demand_scale  # demand value
    obs_buf[1] = K[agent_index] / capacity_scale  # own capacity ratio
    obs_buf[2] = c[agent_index] / cost_scale  # own cost ratio
    obs_buf[3] = hour_x  # hour of day x
    obs_buf[4] = hour_y  # hour of day y
    obs_buf[5] = day_x  # day of week x
    obs_buf[6] = day_y  # day of week y
    return obs_buf


def simple_observer_v3_dim(N: int) -> int:
    """Return the feature dimension for ``simple_observer_v3`` (always 7)."""
    return 7


OBSERVERS = {
    "basic": (basic_observer_dim, basic_observer),
    "simple": (simple_observer_dim, simple_observer),
    "simple_v2": (simple_observer_v2_dim, simple_observer_v2),
    "simple_v3": (simple_observer_v3_dim, simple_observer_v3),
}
