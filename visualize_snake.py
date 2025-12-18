"""
Snake Game Visualization Script

This script visualizes a trained agent playing the Snake game with a clean,
animated display showing the game state in real-time.
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from easy_marl.examples.snake.environment import SnakeEnv
from easy_marl.core.agents import PPOAgent


class SnakeVisualizer:
    """Clean and fast visualization for Snake game."""

    def __init__(self, agent, board_size=15, delay=0.1):
        """
        Initialize the visualizer.
        
        Args:
            agent: Trained PPOAgent to visualize
            board_size: Size of the game board
            delay: Delay between frames in seconds
        """
        self.agent = agent
        self.board_size = board_size
        self.delay = delay
        
        # Create environment
        params = {"board_size": board_size}
        self.env = SnakeEnv(
            agents=[agent],
            params=params,
            seed=None,  # Random seed for variety
            agent_index=0,
            observer_name="simple",
        )
        
        # Color scheme
        self.colors = {
            'empty': '#F5F5F5',
            'wall': '#2C3E50',
            'snake_head': '#27AE60',
            'snake_body': '#52BE80',
            'food': '#E74C3C',
        }
        
        # Game state
        self.game_active = False
        self.obs = None
        self.score = 0
        self.steps = 0
        
    def get_state_grid(self):
        """Convert current board state to a 2D grid for visualization."""
        board = self.env.board
        state = np.zeros((board.size, board.size), dtype=np.int32)
        
        con = "{:" + str(board.full_size) + "b}"
        walls = con.format(board.walls)
        head = con.format(board.head)
        food = con.format(board.food)
        body = con.format(board.body)
        
        for i in range(board.full_size):
            row = i // board.size
            col = i % board.size
            
            if walls[board.full_size - 1 - i] == "1":
                state[row, col] = 1  # Wall
            elif food[board.full_size - 1 - i] == "1":
                state[row, col] = 2  # Food
            elif body[board.full_size - 1 - i] == "1":
                state[row, col] = 3  # Body
            elif head[board.full_size - 1 - i] == "1":
                state[row, col] = 4  # Head
        
        return state
    
    def state_to_color(self, value):
        """Map state value to color."""
        if value == 0:
            return self.colors['empty']
        elif value == 1:
            return self.colors['wall']
        elif value == 2:
            return self.colors['food']
        elif value == 3:
            return self.colors['snake_body']
        elif value == 4:
            return self.colors['snake_head']
        return self.colors['empty']
    
    def play_episode(self, max_steps=1000):
        """
        Play one episode and yield states for animation.
        
        Args:
            max_steps: Maximum steps per episode
            
        Yields:
            Tuple of (state_grid, score, steps, done, info)
        """
        self.obs, _ = self.env.reset()
        self.score = 0
        self.steps = 0
        done = False
        
        while not done and self.steps < max_steps:
            # Get state for visualization
            state_grid = self.get_state_grid()
            info = {
                'food_points': self.env.board.food_points,
                'snake_length': len(self.env.board.body_list) + 1,
            }
            
            yield state_grid, self.score, self.steps, done, info
            
            # Get action from agent
            action, _ = self.agent.model.predict(self.obs, deterministic=True)
            
            # Take step
            self.obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            self.score = step_info.get('food_points', 0)
            self.steps += 1
            
            time.sleep(self.delay)
        
        # Yield final state
        state_grid = self.get_state_grid()
        info = {
            'food_points': self.env.board.food_points,
            'snake_length': len(self.env.board.body_list) + 1,
        }
        yield state_grid, self.score, self.steps, True, info
    
    def visualize_static(self, num_episodes=1, save_path=None):
        """
        Visualize episodes with static matplotlib updates.
        
        Args:
            num_episodes: Number of episodes to play
            save_path: Optional path to save screenshots
        """
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Configure plot settings (these don't change between frames)
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        for episode in range(num_episodes):
            print(f"\n{'=' * 50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'=' * 50}")
            
            for state_grid, score, steps, done, info in self.play_episode():
                ax.clear()
                
                # Re-apply axis settings after clear
                ax.set_xlim(0, self.board_size)
                ax.set_ylim(0, self.board_size)
                ax.set_aspect('equal')
                ax.axis('off')
                
                # Draw grid
                for i in range(self.board_size):
                    for j in range(self.board_size):
                        color = self.state_to_color(state_grid[i, j])
                        rect = patches.Rectangle(
                            (j, self.board_size - 1 - i), 1, 1,
                            linewidth=0.5, edgecolor='#BDC3C7',
                            facecolor=color
                        )
                        ax.add_patch(rect)
                
                # Add title with game info
                snake_length = info['snake_length']
                title = f"Snake Game | Score: {score} | Steps: {steps} | Length: {snake_length}"
                if done:
                    title += " | GAME OVER"
                ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                
                plt.draw()
                plt.pause(0.001)
            
            print(f"Episode finished: Score={score}, Steps={steps}, Length={info['snake_length']}")
            
            if save_path and episode == 0:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Screenshot saved to: {save_path}")
            
            if episode < num_episodes - 1:
                time.sleep(1)  # Pause between episodes
        
        plt.ioff()
        print(f"\n{'=' * 50}")
        print("Visualization complete!")
        print(f"{'=' * 50}")


def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize trained Snake agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize with default trained model
  python visualize_snake.py
  
  # Visualize with custom model
  python visualize_snake.py --model outputs/snake_demo/snake_agent.zip
  
  # Faster visualization
  python visualize_snake.py --delay 0.05 --episodes 3
  
  # Save screenshot
  python visualize_snake.py --save screenshot.png
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='outputs/snake_demo/snake_agent.zip',
        help='Path to trained model (default: outputs/snake_demo/snake_agent.zip)'
    )
    parser.add_argument(
        '--board-size', '-b',
        type=int,
        default=15,
        help='Size of the game board (default: 15)'
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=1,
        help='Number of episodes to play (default: 1)'
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=0.1,
        help='Delay between frames in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--save', '-s',
        type=str,
        default=None,
        help='Path to save screenshot of first episode'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("\nPlease train a model first by running:")
        print("  python examples_snake_demo.py")
        print("\nOr specify a different model path with --model")
        sys.exit(1)
    
    # Load the trained agent
    print("=" * 60)
    print("Snake Game Visualization")
    print("=" * 60)
    print(f"\nLoading model from: {args.model}")
    
    try:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        
        # Create a PPOAgent wrapper with the loaded model
        # Note: We need a temporary environment to initialize the agent structure,
        # but the actual model is loaded from the saved file
        params = {"board_size": args.board_size}
        env = SnakeEnv(
            agents=[],
            params=params,
            seed=42,
            agent_index=0,
            observer_name="simple",
        )
        agent = PPOAgent(env=env, seed=42)
        agent.model = model  # Replace with loaded model
        
        print("✓ Model loaded successfully")
        print("\nConfiguration:")
        print(f"  Board size: {args.board_size}x{args.board_size}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Delay: {args.delay}s per frame")
        print("\nControls:")
        print("  Close the window to stop")
        print("\nStarting visualization...\n")
        
        # Create visualizer and run
        visualizer = SnakeVisualizer(agent, board_size=args.board_size, delay=args.delay)
        visualizer.visualize_static(num_episodes=args.episodes, save_path=args.save)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
