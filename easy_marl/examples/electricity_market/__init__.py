"""
Electricity market MARL environment example.

This module demonstrates how to use the generic MARL framework for
electricity market bidding simulation.
"""

from easy_marl.examples.electricity_market.environment import ElectricityMarketEnv
from easy_marl.examples.electricity_market.market import market_clearing

__all__ = ["ElectricityMarketEnv", "market_clearing"]
