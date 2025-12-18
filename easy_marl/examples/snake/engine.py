"""
Snake game engine.

Based on the Snake-Learning Engine.py implementation using bitboard representation.
This module provides the core game logic for a single-player Snake game.
"""

import math
import random
import numpy as np


def index(b: int) -> int:
    """Convert bitboard position to index."""
    return math.log(b) / math.log(2)


def bini(idx: float) -> int:
    """Convert index to bitboard position."""
    return 2 ** int(idx)


class SnakeBoard:
    """
    Snake game board using bitboard representation.
    
    The board uses bit operations for efficient state representation:
    - walls: Border walls as a bitmask
    - head: Current head position as a bitmask
    - body: All body segments as a bitmask
    - food: Food position as a bitmask
    """

    def __init__(self, size: int = 15, seed: int = None):
        """
        Initialize snake game board.
        
        Args:
            size: Board dimension (size x size grid)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.full_size = self.size ** 2
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Initialize walls (borders)
        self.walls = 0
        # Top and bottom walls
        for i in range(self.size):
            self.walls += bini(i)
            self.walls += bini(self.size * (self.size - 1) + i)
        # Left and right walls
        for i in range(self.size - 2):
            self.walls += bini(self.size + self.size * i)
            self.walls += bini(self.size * 2 - 1 + self.size * i)

        # Initialize snake in center
        self.food = 0
        center_pos = round(self.full_size / 2)
        self.head = bini(center_pos) << self.size
        self.body_list = [(self.head >> self.size * 2), (self.head >> self.size)]

        # Game state
        self.food_points = 0
        self.move_points = 0
        self.energy = 100
        self.end = False

        self.update()
        self.place_food()

    def __str__(self) -> str:
        """Return string representation of the board."""
        con = "{:" + str(self.full_size) + "b}"
        walls = con.format(self.walls)
        head = con.format(self.head)
        food = con.format(self.food)
        body = con.format(self.body)

        rows = []
        r = ""
        for i in range(self.full_size):
            r += " "
            if walls[i] == "1":
                r = r + "X"
            elif head[i] == "1":
                r = r + "H"
            elif food[i] == "1":
                r = r + "F"
            elif body[i] == "1":
                r = r + "B"
            else:
                r = r + " "

            if (i + 1) % self.size == 0:
                rows.append(r[::-1])
                r = ""

        out = "\n".join(rows)
        return out

    def update(self) -> None:
        """Update board state and check for game over conditions."""
        self.body = sum(self.body_list)
        self.all = self.walls | self.food | self.body | self.head

        # Check collision with body
        if self.head & self.body != 0:
            self.end = True

        # Check collision with walls
        if self.head & self.walls != 0:
            self.end = True

        # Check energy
        if self.energy < self.move_points:
            self.end = True

    def place_food(self) -> None:
        """Place food randomly on an empty cell."""
        choices = []
        for i in range(self.full_size):
            if bini(i) & self.all == 0:
                choices.append(i)

        if choices:
            loc = random.choice(choices)
            self.food = bini(loc)
        else:
            # Board is full - game over
            self.end = True

    def push(self, move: int) -> None:
        """
        Execute a move.
        
        Args:
            move: Direction to move (0=Left, 1=Up, 2=Right, 3=Down)
        """
        old_head = self.head

        if move == 0:  # Left
            self.head = self.head >> 1
        elif move == 1:  # Up
            self.head = self.head << self.size
        elif move == 2:  # Right
            self.head = self.head << 1
        elif move == 3:  # Down
            self.head = self.head >> self.size
        else:
            self.end = True
            return

        # Check if food was eaten
        if self.head & self.food == 0:  # No food
            self.body_list.remove(self.body_list[0])
            self.body_list.append(old_head)
        else:  # Food eaten
            self.food_points += 1000
            self.energy += 100
            self.body_list.append(old_head)
            self.place_food()

        self.update()
        self.move_points += 1

    def get_state_array(self) -> np.ndarray:
        """
        Get state as a numpy array for visualization or neural network input.
        
        Returns:
            2D numpy array representing the board state:
            0 = empty, 1 = wall, 2 = food, 3 = body, 4 = head
        """
        state = np.zeros((self.size, self.size), dtype=np.int32)
        
        con = "{:" + str(self.full_size) + "b}"
        walls = con.format(self.walls)
        head = con.format(self.head)
        food = con.format(self.food)
        body = con.format(self.body)
        
        for i in range(self.full_size):
            row = i // self.size
            col = i % self.size
            
            if walls[self.full_size - 1 - i] == "1":
                state[row, col] = 1  # Wall
            elif food[self.full_size - 1 - i] == "1":
                state[row, col] = 2  # Food
            elif body[self.full_size - 1 - i] == "1":
                state[row, col] = 3  # Body
            elif head[self.full_size - 1 - i] == "1":
                state[row, col] = 4  # Head
        
        return state
