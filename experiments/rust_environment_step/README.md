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

# Run comprehensive Python vs Rust performance comparison
python performance_comparison.py

# Or run standalone Rust-only test (doesn't need easy_marl)
python test_rust_only.py

# Or run original performance test (requires easy_marl installed)
python performance_test.py
```

## Performance Results (Python vs Rust)

### Environment Step Performance

Complete environment step function (market clearing + rewards calculation) benchmarked over 1000 iterations:

| Generators | Python/numba | Rust/PyO3 | Speedup | Winner |
|------------|--------------|-----------|---------|--------|
| 5          | 11.68 μs     | 0.91 μs   | **12.80x** | Rust |
| 10         | 11.58 μs     | 1.10 μs   | **10.56x** | Rust |
| 20         | 11.71 μs     | 1.55 μs   | **7.53x**  | Rust |
| 50         | 12.34 μs     | 3.02 μs   | **4.09x**  | Rust |
| 100        | 13.03 μs     | 5.57 μs   | **2.34x**  | Rust |

**Average Speedup: 7.46x** ✅

**Key Finding**: Rust dramatically outperforms Python for combined operations because:
- Eliminates multiple Python-Rust boundary crossings
- Keeps all intermediate data in Rust (no copying)
- Compiler can optimize across the entire operation
- No Python function call overhead

### Observer Performance

Benchmarked over 10,000 iterations:

**Simple Observer** (4 features: demand, capacity, cost, hour):
- Python: 0.73 μs
- Rust: 0.23 μs
- **Speedup: 3.17x** ✅

**Simple Observer V3** (7 features with sin/cos encoding):
- Python: 3.92 μs
- Rust: 0.29 μs
- **Speedup: 13.41x** ✅

**Key Finding**: Rust's trigonometric functions (libm) are extremely fast, providing massive speedup for observers with cyclic time encoding.

## Why This IS Much More Promising

### Actual Results vs Market Clearing Alone

**Market Clearing Experiment** (Experiment 1):
- Average speedup: **0.70x** (Python/numba was 1.45x faster)
- Rust wins only for <10 generators
- Single isolated function limited by boundary crossing overhead

**Environment Step Experiment** (Experiment 2):
- Average speedup: **7.46x** (Rust is dramatically faster)
- Rust wins across all problem sizes (5-100 generators)
- Combined operations eliminate boundary crossing penalty

**The Difference**:
1. **More operations per call**: Environment step does market clearing + rewards + state updates in one Rust call
2. **Eliminated overhead**: Only ONE Python-Rust crossing instead of multiple
3. **Data locality**: All intermediate results stay in Rust memory
4. **Cross-operation optimization**: LLVM can optimize across the entire pipeline

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
├── performance_comparison.py   # Full Python vs Rust benchmark (standalone)
├── performance_test.py         # Original benchmark (needs easy_marl)
└── test_rust_only.py           # Rust-only tests
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
| **Actual speedup** | **0.70x (slower)** | **7.46x (faster!)** |
| Real-world impact | Low (not bottleneck) | High (called frequently) |
| Integration complexity | Low | Medium |

**Conclusion**: Combining operations in Rust provides dramatic performance improvements that isolated functions cannot achieve.

## References

- Market clearing experiment: `../rust_market_clearing/`
- MARL environment: `../../easy_marl/src/environment.py`
- Observer functions: `../../easy_marl/src/observators.py`
- PyO3 docs: https://pyo3.rs/
