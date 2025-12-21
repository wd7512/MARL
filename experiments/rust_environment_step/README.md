# Rust Environment Step Experiment

This experiment explores using Rust for the complete environment step function and observer functions in the MARL framework.

## Motivation

Unlike the market clearing experiment which tested a single isolated function, this experiment combines multiple operations:
- Market clearing (sorting, cumulative sum)
- Reward calculation (array operations)
- Observer computation (feature extraction)

By keeping all operations in Rust, we:
1. **Eliminate boundary crossings**: No Python-Rust-Python roundtrips
2. **Enable cross-operation optimization**: Compiler can optimize across functions
3. **Reduce memory copies**: Data stays in Rust until final result
4. **Leverage zero-cost abstractions**: Rust's ownership model

## What's Implemented

### 1. Complete Environment Step (`environment_step_rust`)

Combines in a single Rust call:
- Bid processing
- Market clearing algorithm
- Reward calculation with penalties
- Reward scaling

**Input**: Bids, quantities, costs, demand, parameters  
**Output**: Price, cleared quantities, rewards (all agents)

### 2. Observer Functions

- `simple_observer_rust`: Basic 4-feature observer
  - Demand, capacity, cost, hour of day
  
- `simple_observer_v3_rust`: Advanced 7-feature observer with cyclic encoding
  - Demand, capacity, cost
  - Hour of day (sin/cos encoding)
  - Day of week (sin/cos encoding)

## How to Run

### Prerequisites

Same as the market clearing experiment:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install maturin numpy numba

# Install MARL package
cd /path/to/MARL
pip install -e .
```

### Build and Test

```bash
# Build the Rust extension
cd experiments/rust_environment_step
maturin build --release
pip install target/wheels/rust_env_step-*.whl

# Run performance test
python performance_test.py
```

## Expected Results

### Environment Step Performance

The Rust version should show more significant speedup here because:
- Combines multiple operations (market clearing + rewards)
- Avoids Python function call overhead
- Reduces memory allocation/copying
- Better cache locality

**Predicted speedup**: 1.5x - 3x for typical problem sizes

### Observer Performance

Observers are simpler functions but called frequently (every step).

**Simple Observer** (basic arithmetic):
- Python: ~0.5 μs
- Rust: ~0.3 μs
- Predicted speedup: ~1.5x

**Simple Observer V3** (with trigonometry):
- Python: ~2 μs
- Rust: ~0.5 μs  
- Predicted speedup: ~3-4x (Rust's libm is very fast)

## Why This Might Be More Promising

### Compared to Market Clearing Alone

1. **More operations per call**: The environment step does market clearing + rewards + state updates
2. **Better amortization**: Fixed overhead of crossing Python-Rust boundary is spread over more work
3. **Less memory pressure**: Intermediate results stay in Rust
4. **Compiler opportunities**: LLVM can optimize across the combined operations

### Real Training Impact

In actual RL training:
- Environment step is called ~1M times during training
- If we save 2-3 μs per step → 2-3 seconds per training run
- Over 100s of experiments → meaningful time savings
- Observers called even more frequently

## Integration Example

```python
try:
    from rust_env_step import environment_step_rust
    USE_RUST_ENV = True
except ImportError:
    USE_RUST_ENV = False

class MARLElectricityMarketEnv(gym.Env):
    def step(self, action):
        # ... action processing ...
        
        if USE_RUST_ENV:
            # Single Rust call - fast path
            price, q_cleared, rewards = environment_step_rust(
                self.b_all, self.q_all, self.c,
                self.D_profile[self.t],
                self.lambda_bid_penalty,
                UNIT_LOL_PENALTY,
                self.demand_scale,
                self.cost_scale,
                self.T
            )
            q_cleared = np.array(q_cleared)
            rewards = np.array(rewards)
        else:
            # Original Python path
            price, q_cleared = market_clearing(self.b_all, self.q_all, self.D_profile[self.t])
            # ... reward calculation ...
        
        # ... rest of step logic ...
```

## Files

```
experiments/rust_environment_step/
├── README.md                   # This file
├── Cargo.toml                  # Rust dependencies
├── pyproject.toml              # Python packaging
├── src/
│   └── lib.rs                  # Rust implementation
└── performance_test.py         # Benchmark script
```

## Troubleshooting

### Build Errors

If you see compilation errors:
```bash
# Clean and rebuild
cd experiments/rust_environment_step
cargo clean
maturin build --release
```

### Import Errors

If Python can't find the module:
```bash
# Check installation
pip list | grep rust_env_step

# Reinstall if needed
pip install --force-reinstall target/wheels/rust_env_step-*.whl
```

### Performance Not as Expected

Factors affecting performance:
- Debug vs Release build (always use `--release`)
- CPU frequency scaling / thermal throttling
- Background processes
- Cache state (run multiple times)

## Next Steps

If this experiment shows promising results:

1. **Integrate into main codebase**: Add optional Rust path to `environment.py`
2. **Parallel environments**: Explore Rayon for multi-environment parallelism
3. **Batch operations**: Process multiple timesteps at once
4. **Custom observations**: Implement domain-specific feature extraction

## Comparison with Market Clearing Experiment

| Aspect | Market Clearing | Environment Step |
|--------|-----------------|------------------|
| Operations | 1 (clearing only) | 3+ (clearing + rewards + obs) |
| Boundary crossings | 1 per call | 1 per combined call |
| Python overhead | Significant | Amortized |
| Expected speedup | 0.7x - 1.2x | 1.5x - 3x |
| Real-world impact | Low (not bottleneck) | Medium (called frequently) |
| Integration complexity | Low | Medium |

## References

- Market clearing experiment: `../rust_market_clearing/`
- MARL environment: `../../easy_marl/src/environment.py`
- Observer functions: `../../easy_marl/src/observators.py`
- PyO3 docs: https://pyo3.rs/
