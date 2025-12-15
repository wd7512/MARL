"""
Helper functions for RL electricity market environment.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit(cache=True)
def market_clearing(bids: np.ndarray, quantities: np.ndarray, demand: float):
    """
    Compute market clearing price and accepted quantities.

    Args:
        bids (np.ndarray): Array of bid prices, shape (N,)
        quantities (np.ndarray): Array of offered quantities, shape (N,)
        demand (float): Market demand

    Returns:
        P_t (float): Clearing price
        q_cleared (np.ndarray): Accepted quantities, shape (N,)
    """
    bids = np.asarray(bids)
    quantities = np.asarray(quantities)
    N = len(bids)

    order = np.argsort(bids)
    bids_sorted = bids[order]
    q_sorted = quantities[order]

    cum_supply = np.cumsum(q_sorted)
    # Find the first index where cumulative supply meets/exceeds demand
    m = np.searchsorted(cum_supply, demand, side="left")

    q_cleared = np.zeros_like(q_sorted)
    if m >= N:  # demand exceeds total supply
        q_cleared[:] = q_sorted
        P_t = bids_sorted[-1]
    else:
        q_cleared[:m] = q_sorted[:m]
        q_cleared[m] = demand - cum_supply[m - 1] if m > 0 else demand
        P_t = bids_sorted[m]

    # Reorder q_cleared to original order
    q_cleared_final = np.zeros_like(q_cleared)
    q_cleared_final[order] = q_cleared

    return P_t, q_cleared_final


@njit(cache=True)
def linear_sf_market_clearing(bids: np.ndarray, quantities: np.ndarray, demand: float):
    """Compute market clearing using supply functions"""
    pass
