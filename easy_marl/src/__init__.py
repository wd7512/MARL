"""
DEPRECATED: This module is deprecated. Use easy_marl.core instead.

This module provides backward compatibility imports.
All new code should import from easy_marl.core.
"""

import warnings

warnings.warn(
    "easy_marl.src is deprecated. Please use easy_marl.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility imports
from easy_marl.core.agents import BaseAgent, PPOAgent, SimpleAgent
from easy_marl.core.environment import MARLEnvironment

__all__ = ["BaseAgent", "PPOAgent", "SimpleAgent", "MARLEnvironment"]
