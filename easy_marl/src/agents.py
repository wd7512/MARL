"""
DEPRECATED: This module is deprecated. Use easy_marl.core.agents instead.

This module provides backward compatibility for agent classes.
"""

import warnings

warnings.warn(
    "easy_marl.src.agents is deprecated. Please use easy_marl.core.agents instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility imports
from easy_marl.core.agents import BaseAgent, PPOAgent, SimpleAgent

__all__ = ["BaseAgent", "PPOAgent", "SimpleAgent"]
