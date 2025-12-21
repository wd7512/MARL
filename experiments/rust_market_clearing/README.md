# Rust Market Clearing Experiment

This experiment explores using Rust (via PyO3) as an alternative implementation for the market clearing algorithm in the MARL electricity market environment.

## Overview

The market clearing algorithm is the core computational component that determines:
- Which generators are selected to meet demand
- At what price the market clears
- How much each generator produces

This experiment implements the algorithm in Rust and compares performance against the Python/numba baseline.

## What's in This Folder

```
experiments/rust_market_clearing/
├── README.md                           # This file
├── rust_market/                        # Rust implementation
│   ├── Cargo.toml                      # Rust dependencies
│   ├── pyproject.toml                  # Python packaging config
│   ├── src/lib.rs                      # Rust source code
│   └── README.md                       # Rust-specific docs
├── performance_test.py                 # Performance benchmark script
├── demo_rust_integration.py            # Interactive demo
└── RUST_PERFORMANCE_STUDY.md          # Detailed analysis
```

## How to Run

### Prerequisites

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Install Python dependencies**:
   ```bash
   pip install maturin numpy numba
   ```

3. **Install the MARL package** (from repository root):
   ```bash
   cd /path/to/MARL
   pip install -e .
   ```

### Step 1: Build the Rust Extension

```bash
cd experiments/rust_market_clearing/rust_market
maturin build --release
pip install target/wheels/rust_market-0.1.0-*.whl
```

**Expected output:**
```
🔗 Found pyo3 bindings
🐍 Found CPython 3.11+
   Compiling rust_market v0.1.0
    Finished `release` profile [optimized] target(s) in 15.90s
📦 Built wheel for CPython to target/wheels/rust_market-0.1.0-*.whl
```

### Step 2: Run the Performance Test

```bash
cd experiments/rust_market_clearing
python performance_test.py
```

**Expected output:**
```
Performance Test: Market Clearing Algorithm
Comparing Python (numba) vs Rust implementations

============================================================
Benchmarking: Python (numba JIT)
============================================================
N=  5 generators: mean=   1.53μs, median=   1.47μs, ...
N= 10 generators: mean=   1.58μs, median=   1.55μs, ...
...

============================================================
Benchmarking: Rust (PyO3)
============================================================
N=  5 generators: mean=   1.24μs, median=   1.22μs, ...
N= 10 generators: mean=   1.54μs, median=   1.50μs, ...
...

============================================================
Performance Comparison
============================================================
    N |  Python (μs) |    Rust (μs) |  Speedup
------+--------------+--------------+---------
    5 |        1.53  |        1.24  |     1.23x
   10 |        1.58  |        1.54  |     1.02x
   20 |        1.68  |        2.23  |     0.75x
   50 |        2.05  |        4.45  |     0.46x
  100 |        2.78  |        7.82  |     0.36x
  200 |        4.46  |       14.77  |     0.30x

Average Speedup: 0.69x
⚠️  Python/numba is 1.45x faster on average

============================================================
Correctness Verification
============================================================
N=  5: ✅ PASS
N= 10: ✅ PASS
N= 50: ✅ PASS
```

### Step 3: Run the Interactive Demo

```bash
cd experiments/rust_market_clearing
python demo_rust_integration.py
```

This will show:
- Basic market clearing example with both implementations
- Quick performance comparison
- Integration patterns for using Rust as a drop-in replacement

## Results Summary

### Performance Results

| Generators | Python (numba) | Rust (PyO3) | Winner | Speedup |
|------------|----------------|-------------|--------|---------|
| 5          | 1.53 μs        | 1.24 μs     | **Rust** | 1.23x |
| 10         | 1.58 μs        | 1.54 μs     | **Rust** | 1.02x |
| 20         | 1.68 μs        | 2.23 μs     | Python | 0.75x |
| 50         | 2.05 μs        | 4.45 μs     | Python | 0.46x |
| 100        | 2.78 μs        | 7.82 μs     | Python | 0.36x |
| 200        | 4.46 μs        | 14.77 μs    | Python | 0.30x |

**Key Findings:**

1. ✅ **Rust is faster for small problems** (5-10 generators)
   - Lower function call overhead
   - Good baseline performance

2. ⚠️ **Python/numba scales better** (20+ generators)
   - Superior LLVM optimization for this algorithm
   - Better vectorization and cache efficiency
   - 3.3x faster at 200 generators

3. ✅ **Both are very fast** (<15 μs even at 200 generators)
   - Not a performance bottleneck for training
   - Market clearing takes <0.1% of total training time

### Correctness

- ✅ 100% correctness validation across all test cases
- ✅ Identical results to Python/numba implementation
- ✅ Handles edge cases (zero demand, full demand, single generator)

## Conclusions

### For the Market Clearing Algorithm

**Recommendation: Keep Python/numba**

Reasons:
- Already optimized (1-5 μs per call)
- Scales better for realistic problem sizes (20+ generators)
- Simpler to maintain (no compilation step)
- Not a performance bottleneck

### Learning Outcomes

This experiment successfully demonstrated:

1. ✅ **PyO3 Integration**: How to create Python extensions with Rust
2. ✅ **NumPy Interop**: Working with NumPy arrays in Rust
3. ✅ **Performance Benchmarking**: Rigorous comparison methodology
4. ✅ **When to Use Rust**: Understanding trade-offs vs JIT compilation

### Future Rust Opportunities

Consider Rust for:

1. **Agent Policy Networks**: Custom neural network operations
   - Parallel inference across multiple agents
   - Custom gradient computation
   - Memory-efficient state management

2. **Environment Step Function**: Core simulation loop
   - Parallel environment execution
   - Complex state transitions
   - Memory-safe concurrency

3. **Observation Building**: Feature extraction pipeline
   - Parallel observation computation for multiple agents
   - SIMD-optimized feature calculations
   - Zero-copy data transformations

4. **Replay Buffer**: Experience storage and sampling
   - Concurrent read/write operations
   - Memory-efficient circular buffers
   - Fast sampling algorithms

See the next experiment (`experiments/rust_environment_step/`) for exploration of these opportunities.

## Troubleshooting

### Rust Extension Not Found

If you get `ModuleNotFoundError: No module named 'rust_market'`:

```bash
# Rebuild and reinstall
cd experiments/rust_market_clearing/rust_market
maturin build --release
pip install --force-reinstall target/wheels/rust_market-*.whl
```

### Performance Numbers Don't Match

Performance can vary based on:
- CPU architecture and cache size
- System load and background processes
- Python version and numba version
- Release vs debug build (always use `--release`)

Run the benchmark multiple times and look at median values.

### Build Errors

Common issues:
- **Rust not installed**: Install from https://rustup.rs/
- **Wrong Python version**: Requires Python 3.11+
- **Missing dependencies**: Run `pip install maturin numpy`

## References

- [Full Performance Study](RUST_PERFORMANCE_STUDY.md)
- [Rust Implementation Details](rust_market/README.md)
- [PyO3 Documentation](https://pyo3.rs/)
- [Rust-Python Interop Tutorial](https://rust-exercises.com/rust-python-interop/)

## Next Steps

1. Explore Rust for other framework components (see `experiments/rust_environment_step/`)
2. Investigate parallel agent training with Rust-based environments
3. Consider custom PPO operations in Rust for specific use cases
