"""
Snake game example for the MARL framework.

This example demonstrates a single-agent Snake game implementation
using the easy_marl framework.
"""

from easy_marl.examples.snake.environment import SnakeEnv
from easy_marl.examples.snake.training import train, evaluate_agent

__all__ = ["SnakeEnv", "train", "evaluate_agent"]
