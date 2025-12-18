"""
Observation builders for Snake game.

This module provides functions to convert the Snake game state into
observations suitable for reinforcement learning agents.
"""

import numpy as np
from easy_marl.examples.snake.engine import index


def get_simple_observation(board) -> np.ndarray:
    """
    Get simple observation vector from Snake board state.
    
    This is based on the get_inputs function from the original Snake-Learning code.
    The observation includes:
    - Distance to walls in all 4 directions
    - Distance to walls diagonally
    - Distance to food in cardinal and diagonal directions
    - Distance to body parts in cardinal and diagonal directions
    
    Args:
        board: SnakeBoard instance
        
    Returns:
        24-dimensional observation vector (normalized to [0, 1])
    """
    out = np.zeros(24, dtype=np.float32)

    head = index(board.head)
    food = index(board.food)

    head_loc = (int(head // board.size), int(head % board.size))
    food_loc = (int(food // board.size), int(food % board.size))

    diff_x = head_loc[0] - food_loc[0]
    diff_y = head_loc[1] - food_loc[1]

    # Distance to walls (cardinal directions)
    out[0] = head_loc[0]
    out[1] = head_loc[1]
    out[2] = board.size - out[0] - 1
    out[3] = board.size - out[1] - 1

    # Distance to walls (diagonal directions)
    out[4] = max(0, (out[0] + out[1] - 1) / 2)
    out[5] = max(0, (out[1] + out[2] - 1) / 2)
    out[6] = max(0, (out[2] + out[3] - 1) / 2)
    out[7] = max(0, (out[3] + out[0] - 1) / 2)

    # Distance to food (cardinal directions)
    if head_loc[0] == food_loc[0]:
        if head_loc[1] > food_loc[1]:
            out[8] = head_loc[1] - food_loc[1]
        else:
            out[9] = food_loc[1] - head_loc[1]

    if head_loc[1] == food_loc[1]:
        if head_loc[0] > food_loc[0]:
            out[10] = head_loc[0] - food_loc[0]
        else:
            out[11] = food_loc[0] - head_loc[0]

    # Distance to food (diagonal directions)
    if diff_x == diff_y:
        if diff_x < 0:
            out[12] = abs(diff_x)
        else:
            out[13] = abs(diff_x)

    if diff_x == -diff_y:
        if diff_x < 0:
            out[14] = abs(diff_x)
        else:
            out[15] = abs(diff_x)

    # Distance to body parts
    for body in board.body_list:
        b = index(body)
        loc = (int(b // board.size), int(b % board.size))

        diff_x = head_loc[0] - loc[0]
        diff_y = head_loc[1] - loc[1]

        # Cardinal directions
        if loc[0] == head_loc[0]:
            if loc[1] > head_loc[1]:
                if out[16] == 0:
                    out[16] = loc[1] - head_loc[1]
                else:
                    out[16] = min(out[16], loc[1] - head_loc[1])
            else:
                if out[17] == 0:
                    out[17] = head_loc[1] - loc[1]
                else:
                    out[17] = min(out[17], head_loc[1] - loc[1])

        if loc[1] == head_loc[1]:
            if loc[0] > head_loc[0]:
                if out[18] == 0:
                    out[18] = loc[0] - head_loc[0]
                else:
                    out[18] = min(out[18], loc[0] - head_loc[0])
            else:
                if out[19] == 0:
                    out[19] = head_loc[0] - loc[0]
                else:
                    out[19] = min(out[19], head_loc[0] - loc[0])

        # Diagonal directions
        if diff_x == diff_y:
            if diff_x < 0:
                if out[20] == 0:
                    out[20] = abs(diff_x)
                else:
                    out[20] = min(out[20], abs(diff_x))
            else:
                if out[21] == 0:
                    out[21] = abs(diff_x)
                else:
                    out[21] = min(out[21], abs(diff_x))

        if diff_x == -diff_y:
            if diff_x < 0:
                if out[22] == 0:
                    out[22] = abs(diff_x)
                else:
                    out[22] = min(out[22], abs(diff_x))
            else:
                if out[23] == 0:
                    out[23] = abs(diff_x)
                else:
                    out[23] = min(out[23], abs(diff_x))

    # Normalize all features to [0, 1]
    # Ensure board.size > 2 to avoid division by zero
    if board.size > 2:
        for i in range(len(out)):
            if out[i] != 0:
                out[i] = (board.size - out[i] - 1) / (board.size - 2)
    else:
        # For very small boards, just normalize by board size
        for i in range(len(out)):
            if out[i] != 0:
                out[i] = out[i] / board.size

    return out


def simple_obs_dim() -> int:
    """Return dimension of simple observation."""
    return 24


# Dictionary of available observers
OBSERVERS = {
    "simple": (simple_obs_dim, get_simple_observation),
}
