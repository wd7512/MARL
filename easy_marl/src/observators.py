"""
DEPRECATED: This module is deprecated. Use easy_marl.examples.electricity_market.observators instead.

This module provides backward compatibility for observer functions.
"""

import warnings

warnings.warn(
    "easy_marl.src.observators is deprecated. "
    "Please use easy_marl.examples.electricity_market.observators instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility imports
from easy_marl.examples.electricity_market.observators import (
    basic_observer,
    basic_observer_dim,
    simple_observer,
    simple_observer_dim,
    simple_observer_v2,
    simple_observer_v2_dim,
    simple_observer_v3,
    simple_observer_v3_dim,
    OBSERVERS,
)

__all__ = [
    "basic_observer",
    "basic_observer_dim",
    "simple_observer",
    "simple_observer_dim",
    "simple_observer_v2",
    "simple_observer_v2_dim",
    "simple_observer_v3",
    "simple_observer_v3_dim",
    "OBSERVERS",
]
