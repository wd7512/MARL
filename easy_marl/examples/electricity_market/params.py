"""
Parameter generation functions for electricity market environments.

This module provides utilities to create realistic market parameters
for training and evaluation.
"""

import numpy as np
from typing import Dict

# Try to import scipy for advanced demand profiles, fall back to simple profiles if not available
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def make_default_params(N: int = 5, T: int = 24) -> Dict:
    """
    Create default environment parameters for electricity market.

    Args:
        N: Number of generators.
        T: Time horizon (number of timesteps/hours).

    Returns:
        Dictionary of environment parameters suitable for ElectricityMarketEnv.
    """
    # Sinusoidal demand profile (mimics daily pattern)
    demand_profile = (np.sin(np.linspace(0, 2 * np.pi, T)) + 1.5) * 50.0

    # Uniform capacities
    capacities = np.full(N, 30.0 * 5 / N)

    # Heterogeneous costs (linearly increasing)
    costs = np.linspace(20.0, 40.0, N)

    params = {
        "N_generators": N,
        "T": T,
        "demand_profile": demand_profile.tolist(),
        "capacities": capacities.tolist(),
        "costs": costs.tolist(),
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }
    return params


def make_competitive_params(N: int = 8, T: int = 24, competition_rate: int = 2) -> Dict:
    """
    Create competitive environment parameters with paired generators.

    This creates a more competitive market by having multiple generators
    at similar cost levels, encouraging strategic bidding.

    Args:
        N: Number of generators.
        T: Time horizon (number of timesteps).
        competition_rate: Number of generators at each cost level.

    Returns:
        Dictionary of environment parameters.
    """
    # Sinusoidal demand profile
    demand_profile = np.sin(np.linspace(0, 2 * np.pi, T)) + 1.5

    # Uniform capacities
    capacities = np.full(N, 30.0 * 5 / N)

    # Scale demand profile
    scaled_demand_profile = demand_profile / demand_profile.max()
    demand_profile = scaled_demand_profile * (np.sum(capacities[:-1]) - 1)

    # Heterogeneous costs with duplicates for competition
    costs = np.linspace(20.0, 40.0, int(np.ceil(N / competition_rate))).tolist()
    costs = costs * competition_rate  # Duplicate to create pairs
    costs = np.array(costs[:N])  # Trim to N if needed
    costs = np.sort(costs)

    params = {
        "N_generators": N,
        "T": T,
        "demand_profile": demand_profile.tolist(),
        "capacities": capacities.tolist(),
        "costs": costs.tolist(),
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }
    return params


def make_advanced_params(N: int = 5, T: int = 7 * 24) -> Dict:
    """
    Create advanced environment parameters with weekly demand patterns.

    This generates more realistic demand profiles with weekday/weekend
    differences and can incorporate renewable generators.

    Args:
        N: Number of generators.
        T: Time horizon (default is one week = 168 hours).

    Returns:
        Dictionary of environment parameters.
    """
    # Get plant data (can be extended to load from CSV)
    plant_data = _example_plant_data(N)
    
    # Generate weekly demand profile
    demand_profile = _example_demand_profile()
    
    # Scale demand to match system capacity
    demand_profile = demand_profile * sum(plant_data["capacities"]) * 0.75

    params = {
        "N_generators": N,
        "T": T,
        "demand_profile": demand_profile.tolist(),
        "capacities": plant_data["capacities"].tolist(),
        "costs": plant_data["srmc"].tolist(),
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }
    return params


def _example_plant_data(N: int = 5) -> Dict:
    """
    Generate example plant data including renewable generators.
    
    Args:
        N: Number of generators.
    
    Returns:
        Dictionary with plant characteristics.
    """
    # Default configuration: solar, wind, and conventional generators
    if N == 5:
        plant_data = {
            "capacities": [40, 40, 20, 20, 20],
            "srmc": [0, 0, 10, 20, 30],  # Solar and wind have zero marginal cost
        }
    else:
        # Simple fallback for arbitrary N
        plant_data = {
            "capacities": [30.0 * 5 / N] * N,
            "srmc": list(np.linspace(10, 30, N)),
        }
    
    for key in plant_data:
        plant_data[key] = np.asarray(plant_data[key])
    
    return plant_data


def _example_demand_profile() -> np.ndarray:
    """
    Generate realistic weekly demand profile with different weekday/weekend patterns.
    
    Returns:
        Array of hourly demand values for one week (168 hours).
    """
    if not HAS_SCIPY:
        # Fallback to simple pattern if scipy not available
        return _simple_weekly_profile()

    # Weekday pattern: Low at midnight, peak at 8am and 7pm
    wd_x = [0, 8, 13, 19, 24]
    wd_y = [0.5, 0.85, 0.7, 1.0, 0.5]
    wd_spline = CubicSpline(wd_x, wd_y, bc_type="periodic")

    # Weekend pattern: Lower demand, later peaks
    we_x = [0, 10, 14, 20, 24]
    we_y = [0.45, 0.75, 0.6, 0.85, 0.45]
    we_spline = CubicSpline(we_x, we_y, bc_type="periodic")

    # Generate hourly profiles
    hours = np.arange(24)
    weekday_profile = wd_spline(hours)
    weekend_profile = we_spline(hours)

    # Construct weekly profile (5 weekdays, 2 weekends)
    demand_profile = np.concatenate([weekday_profile] * 5 + [weekend_profile] * 2)
    
    # Normalize
    demand_profile = demand_profile / demand_profile.max()

    return demand_profile


def _simple_weekly_profile() -> np.ndarray:
    """
    Simple weekly demand profile fallback (no scipy required).
    
    Returns:
        Array of hourly demand values for one week.
    """
    # Simple sinusoidal pattern with weekday/weekend variation
    T = 7 * 24
    t = np.arange(T)
    
    # Daily cycle
    daily = np.sin(2 * np.pi * t / 24 + np.pi / 2) * 0.3 + 0.7
    
    # Weekly modulation (lower on weekends)
    weekly = np.where((t // 24) >= 5, 0.85, 1.0)  # Days 5-6 are weekend
    
    demand_profile = daily * weekly
    return demand_profile / demand_profile.max()
