"""
DEPRECATED: This module is deprecated. Use easy_marl.examples.electricity_market instead.

This module provides backward compatibility for the old MARLElectricityMarketEnv class.
"""

import warnings

warnings.warn(
    "easy_marl.src.environment is deprecated. "
    "Please use easy_marl.examples.electricity_market.environment instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility import
from easy_marl.examples.electricity_market.environment import ElectricityMarketEnv

# Keep old name for backward compatibility
MARLElectricityMarketEnv = ElectricityMarketEnv

__all__ = ["MARLElectricityMarketEnv", "ElectricityMarketEnv"]
