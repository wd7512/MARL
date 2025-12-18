"""
Demo script for the Snake game example.

This demonstrates how to use the Snake game environment
and train a simple agent.
"""

from easy_marl.examples.snake.training import train


def main():
    print("=" * 60)
    print("Snake Game - Single Agent Training Demo")
    print("=" * 60)
    print()
    print("This example trains a single agent to play the Snake game")
    print("using PPO (Proximal Policy Optimization).")
    print()
    print("The agent learns to:")
    print("  - Navigate the board to collect food")
    print("  - Avoid walls and its own body")
    print("  - Maximize survival time and score")
    print()
    print("=" * 60)
    print()

    # Train for a small number of timesteps as a demo
    # For better results, increase timesteps to 100000 or more
    agent, info = train(
        timesteps=10000,  # Short training for demo
        seed=42,
        save_dir="outputs/snake_demo",
        verbose=True,
        board_size=15,
        eval_episodes=5,
    )

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    print("Model and results saved to: outputs/snake_demo/")
    print()
    print("To train for better results, increase timesteps:")
    print("  train(timesteps=100000, ...)")


if __name__ == "__main__":
    main()
