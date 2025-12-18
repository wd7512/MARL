# Easy MARL

A generic Multi-Agent Reinforcement Learning (MARL) framework with complete examples including electricity market bidding and Snake game. Built on Gymnasium and Stable-Baselines3, easy_marl provides clean abstractions for multi-agent learning with support for both Iterated Best Response (IBR) and Simultaneous Best Response (SBR) training.

## Features

- **Generic MARL Framework**: Extensible base classes for agents, environments, and training loops
- **PPO-Based Learning**: Built on Stable-Baselines3 with proper policy freezing for multi-agent scenarios
- **Training Flexibility**: Both sequential (IBR/SBR) and parallel execution modes
- **Reproducibility**: Explicit seeding across Python, NumPy, and PyTorch
- **Complete Examples**: 
  - Electricity market bidding simulation with strategic generators
  - Snake game (single-agent for now) demonstrating game-based RL

## Install

Requires Python 3.11+. Install with `uv` (recommended) or `pip`:

```bash
# With uv
uv sync --extra dev

# With pip
pip install -e ".[dev]"
```

## Quick Start

### Electricity Market Example

Run the electricity market training example:

```bash
python main.py
```

Or use the API directly:

```python
from easy_marl.examples.electricity_market.training import auto_train

# Automatically selects sequential or parallel based on problem size
agents, info = auto_train(N=5, num_rounds=3, timesteps_per_agent=5000)
```

### Snake Game Example

Run the Snake game training example:

```bash
python examples_snake_demo.py
```

Visualize a trained agent:

```bash
python visualize_snake.py
```

Or use the API directly:

```python
from easy_marl.examples.snake.training import train

# Train a single agent to play Snake
agent, info = train(timesteps=100000, board_size=15, seed=42)
```

## Project Structure

```
easy_marl/
  core/                         # Generic MARL framework
    agents.py                   # BaseAgent, PPOAgent with proper policy freezing
    environment.py              # MARLEnvironment base class
    training.py                 # Generic sequential/parallel training loops
  examples/
    electricity_market/         # Multi-agent electricity market example
      environment.py            # ElectricityMarketEnv (extends MARLEnvironment)
      market.py                 # Market clearing mechanisms (Numba-optimized)
      observators.py            # Observation builders for market state
      params.py                 # Parameter generation utilities
      training.py               # Market-specific training wrappers
    snake/                      # Single-agent Snake game example
      engine.py                 # Snake game logic using bitboard representation
      environment.py            # SnakeEnv (extends MARLEnvironment)
      observators.py            # Observation builders for Snake state
      training.py               # Snake-specific training utilities
tests/                          # Unit tests
main.py                         # Entry point for electricity market
examples_snake_demo.py          # Entry point for Snake game
```

### Backward Compatibility

The old structure (`easy_marl.src` and `easy_marl.examples.bidding`) is deprecated but still works:

```python
# Old (deprecated, but works)
from easy_marl.src.agents import PPOAgent
from easy_marl.examples.bidding.training import auto_train

# New (recommended)
from easy_marl.core.agents import PPOAgent
from easy_marl.examples.electricity_market.training import auto_train
```

## Training Modes

### Sequential Training (IBR or SBR)

```python
from easy_marl.examples.electricity_market.training import sequential_train

# Simultaneous Best Response (default)
agents, info = sequential_train(
    N=5, 
    num_rounds=3, 
    timesteps_per_agent=5000,
    update_schedule="SBR"  # Agents train against frozen opponents
)

# Iterated Best Response
agents, info = sequential_train(
    N=5, 
    num_rounds=3, 
    timesteps_per_agent=5000,
    update_schedule="IBR"  # Agents see immediate updates
)
```

### Parallel Training (SBR)

```python
from easy_marl.examples.electricity_market.training import parallel_train

agents, info = parallel_train(
    N=8, 
    num_rounds=5, 
    timesteps_per_agent=10000,
    n_workers=4  # Number of parallel processes
)
```

## Snake Game Example

The Snake example demonstrates how to use the MARL framework for a classic single-agent game:

```python
from easy_marl.examples.snake.training import train, evaluate_agent

# Train an agent
agent, info = train(
    timesteps=100000,
    board_size=15,
    seed=42,
    save_dir="outputs/snake"
)

# Evaluate the trained agent
eval_results = evaluate_agent(
    agent,
    params={"board_size": 15},
    num_episodes=10,
    seed=42
)

print(f"Mean food collected: {eval_results['mean_food']:.1f}")
print(f"Max snake length: {eval_results['mean_max_snake_length']:.1f}")
```

The Snake environment features:
- **Bitboard representation** for efficient state management
- **24-dimensional observation space** including:
  - Distance to walls in cardinal and diagonal directions
  - Distance to food in all directions
  - Distance to body segments in all directions
- **Simple action space**: 4 discrete actions (Left, Up, Right, Down)
- **Reward structure**:
  - +10 for eating food
  - -0.01 per move (encourages efficiency)
  - -1 for dying (collision with wall or body)

### Visualization

Watch your trained agent play with the visualization tool:

```bash
# Basic visualization
python visualize_snake.py

# Custom options
python visualize_snake.py --delay 0.05 --episodes 3 --save screenshot.png
```

Features:
- Real-time animated display with color-coded grid
- Live statistics (score, steps, snake length)
- Configurable speed and board size
- Screenshot capture support

This example is based on the [Snake-Learning](https://github.com/wd7512/Snake-Learning) implementation.

## Extending to New Domains

1. **Create a domain environment** that extends `MARLEnvironment`:

```python
from easy_marl.core.environment import MARLEnvironment

class MyEnvironment(MARLEnvironment):
    def _build_observation_space(self):
        # Define observation space
        
    def _build_action_space(self):
        # Define action space
        
    def _get_obs(self, agent_index=None):
        # Build observation for agent
        
    def step(self, action):
        # Execute action and return (obs, reward, done, truncated, info)
```

2. **Use generic training loops** from `easy_marl.core.training`

3. **Initialize agents** with `PPOAgent` or create custom agent classes extending `BaseAgent`

See `easy_marl/examples/electricity_market/` for a complete example.

## Testing

```bash
# All tests
pytest -v

# Quick tests (excluding slow training tests)
pytest tests/unit_tests/ --ignore=tests/unit_tests/test_bidding/
```

## Code Review

A comprehensive code review is available in [CODE_REVIEW.md](CODE_REVIEW.md), covering:
- Code quality assessment
- Security analysis
- Architecture patterns
- Performance considerations
- Refactoring recommendations

## Key Concepts

### Multi-Agent Training Regimes

**Iterated Best Response (IBR)**: Gauss-Seidel style updates where each agent optimizes against the current state of all others. Updates propagate immediately within a round.

**Simultaneous Best Response (SBR)**: Jacobi style updates where all agents optimize against a frozen snapshot of opponent policies. Updates are synchronized at round boundaries.

### Policy Freezing

PPOAgent implements proper policy freezing using serialization, ensuring that opponent policies remain fixed during training:

```python
# Get a truly frozen policy function
frozen_policy = agent.fixed_act_function(deterministic=True)

# Train the agent further
agent.train(total_timesteps=10000)

# Frozen policy is unaffected
action = frozen_policy(observation)  # Still uses old policy
```

## License

MIT License. See [LICENSE](LICENSE) for details.
