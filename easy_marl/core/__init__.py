"""
Generic Multi-Agent Reinforcement Learning (MARL) framework.

This module provides generic base classes and utilities for building
multi-agent reinforcement learning systems.
"""

from easy_marl.core.agents import BaseAgent, PPOAgent, SimpleAgent
from easy_marl.core.environment import MARLEnvironment

__all__ = ["BaseAgent", "PPOAgent", "SimpleAgent", "MARLEnvironment"]
