# Agentic Uplift

Multi-agent reinforcement learning (MARL) framework with an electricity market example. Includes PPO-based agents, Gymnasium-compatible environments, and reproducible sequential/parallel training.

## Install

Requires Python 3.11+ and `uv`.

```powershell
uv sync --extra dev
```

Editable install is set up automatically from `pyproject.toml`.

## Project Layout

```
easy_marl/
  src/                    # Core MARL framework
    agents.py             # Base + PPO agents
    environment.py        # Gymnasium-compatible MARL env
    observators.py        # Observation builders
  examples/
    bidding/              # Electricity market example
      market.py           # Market clearing + utils
      training.py         # Sequential/parallel training
tests/                    # Unit tests
outputs/                  # Training/eval artifacts (ignored)
main.py                   # Simple entry point
pyproject.toml            # Build + deps
```

## Quickstart

Run the default training:

```powershell
uv run python main.py
```

Use the bidding example training directly:

```powershell
uv run python -c "from easy_marl.examples.bidding.training import parallel_train; parallel_train(N=3, num_rounds=1, timesteps_per_agent=48)"
```

## Tests

```powershell
uv run pytest -q
```

## Key Concepts

- MARL environment returns agent-specific observations built via observers.
- Two training regimes:
  - Sequential (Iterated Best Response / IBR)
  - Parallel (Simultaneous Best Response / SBR)
- Reproducibility via explicit seeding (NumPy/Torch/processes).

## Development Notes

- Package installs as `easy_marl` (editable).
- Example code lives under `easy_marl/examples/bidding` to keep core MARL generic.
- Outputs are written to `outputs/` and excluded from packaging.

## License

Choose and add a license (e.g., MIT or Apache-2.0) before public release.
