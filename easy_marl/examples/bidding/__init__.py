"""
DEPRECATED: This module has been renamed to easy_marl.examples.electricity_market.

This module provides backward compatibility imports.
All new code should import from easy_marl.examples.electricity_market.
"""

import warnings

warnings.warn(
    "easy_marl.examples.bidding is deprecated. "
    "Please use easy_marl.examples.electricity_market instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility imports
from easy_marl.examples.electricity_market.environment import ElectricityMarketEnv
from easy_marl.examples.electricity_market.market import market_clearing
from easy_marl.examples.electricity_market.training import (
    sequential_train,
    parallel_train,
    auto_train,
)

# Keep old name for complete compatibility
MARLElectricityMarketEnv = ElectricityMarketEnv

__all__ = [
    "ElectricityMarketEnv",
    "MARLElectricityMarketEnv",
    "market_clearing",
    "sequential_train",
    "parallel_train",
    "auto_train",
]
