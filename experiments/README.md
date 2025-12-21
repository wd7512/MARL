# Rust Integration Experiments for MARL

This directory contains experiments exploring Rust integration into the MARL framework for performance optimization.

## Overview

These experiments demonstrate how Rust (via PyO3) can be integrated with Python-based reinforcement learning frameworks to improve computational performance while maintaining ease of use.

## Experiments

### 1. [Market Clearing](./rust_market_clearing/) ⚠️ Mixed Results

**What**: Rust implementation of the market clearing algorithm  
**Goal**: Compare Rust vs Python/numba for a single computational kernel  
**Status**: ✅ Complete  
**Result**: Python/numba is ~1.45x faster on average

**Key Finding**: For simple numerical algorithms that numba can JIT-compile well, Python/numba is hard to beat. Rust shows advantage only for very small problem sizes (<20 generators).

**Recommendation**: Keep Python/numba for market clearing

---

### 2. [Environment Step](./rust_environment_step/) ✅ Promising

**What**: Rust implementation of complete environment step + observers  
**Goal**: Test Rust for combined operations (market clearing + rewards + observations)  
**Status**: ✅ Complete  
**Result**: More promising than market clearing alone

**Performance**:
- Environment step (combined operations): 0.9-5.7 μs depending on problem size
- Simple observer: 0.25 μs
- Observer V3 (with trig): 0.29 μs

**Key Advantages**:
1. Combines multiple operations → amortizes boundary crossing overhead
2. Eliminates intermediate memory allocations
3. Enables cross-operation compiler optimizations
4. Very fast trigonometric functions (observer V3)

**Recommendation**: Consider for production if training time is a concern

---

## Quick Start

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install maturin numpy numba

# Install MARL package
cd /path/to/MARL
pip install -e .
```

### Run Experiment 1: Market Clearing

```bash
cd experiments/rust_market_clearing
cd rust_market
maturin build --release
pip install target/wheels/rust_market-*.whl
cd ..
python performance_test.py
```

### Run Experiment 2: Environment Step

```bash
cd experiments/rust_environment_step
maturin build --release
pip install target/wheels/rust_env_step-*.whl
python test_rust_only.py
```

## Results Summary

### Experiment 1: Market Clearing

| Generators | Python (numba) | Rust (PyO3) | Winner |
|------------|----------------|-------------|--------|
| 5          | 1.53 μs        | 1.24 μs     | Rust 1.23x |
| 20         | 1.68 μs        | 2.23 μs     | Python 1.33x |
| 100        | 2.78 μs        | 7.82 μs     | Python 2.81x |

### Experiment 2: Environment Step

| Component | Time (μs) | Notes |
|-----------|-----------|-------|
| Full step (5 gen) | 0.93 | Market + rewards combined |
| Full step (100 gen) | 5.67 | Scales linearly |
| Observer (simple) | 0.25 | 4 features |
| Observer V3 (trig) | 0.29 | 7 features with sin/cos |

## Key Learnings

### When Rust Wins

1. **Combined operations**: Multiple steps kept in Rust (like env step)
2. **Frequent small calls**: Observers called every step
3. **Trigonometric functions**: Rust's libm is very fast
4. **Memory efficiency**: Avoid Python allocations
5. **Cross-operation optimization**: Compiler sees whole pipeline

### When Python/Numba Wins

1. **Simple numerical algorithms**: What numba was designed for
2. **Vector operations**: NumPy is already using optimized BLAS
3. **Rapid prototyping**: No compilation step
4. **Well-optimized code paths**: Market clearing is already fast

### General Guidelines

Use Rust when:
- ✅ Operations can be combined (reduce boundary crossings)
- ✅ Need custom control flow beyond numba's scope
- ✅ Memory safety and concurrency are concerns
- ✅ Integration with Rust ecosystem (Rayon, Polars, etc.)

Keep Python when:
- ✅ Algorithm is already fast enough
- ✅ Numba handles it well
- ✅ Rapid iteration is more important than raw speed
- ✅ Not a performance bottleneck

## Performance Impact on Training

### Current Bottlenecks (profiled)

From actual training runs with N=5 agents, 3 rounds, 100 timesteps each:

1. **PPO training**: ~95% of time (neural network forward/backward passes)
2. **Environment steps**: ~3% of time
3. **Market clearing**: ~0.5% of time
4. **Observations**: ~0.3% of time
5. **Other**: ~1.2%

### Potential Savings with Rust

If we optimize environment step + observations:
- Current: ~3.3% of total time
- With Rust 2x speedup: ~1.65% of total time
- **Total speedup**: ~1.6% faster training

**Conclusion**: Rust helps but won't transform training speed. The real bottleneck is the neural network training itself.

## Future Directions

### High-Impact Opportunities

1. **Parallel Environment Execution** 🔥
   - Use Rayon to run multiple environments in parallel
   - Could provide 2-4x speedup on multi-core machines
   - Most impactful for sample collection

2. **Custom Neural Network Layers**
   - Implement domain-specific operations in Rust
   - Integrate with PyTorch via custom extensions
   - Could speed up policy inference

3. **Replay Buffer**
   - Concurrent read/write operations
   - Memory-efficient circular buffer
   - Fast sampling algorithms

4. **Batch Environment Steps**
   - Process multiple timesteps at once
   - Vectorized operations across agents
   - Reduce overhead

### Lower-Priority Opportunities

- Custom observation preprocessing pipelines
- Parallel agent policy inference
- Specialized reward calculations
- State history compression

## Directory Structure

```
experiments/
├── README.md                          # This file
├── rust_market_clearing/              # Experiment 1
│   ├── README.md                      # Experiment 1 documentation
│   ├── rust_market/                   # Rust implementation
│   ├── performance_test.py            # Benchmarks
│   ├── demo_rust_integration.py       # Demo script
│   └── RUST_PERFORMANCE_STUDY.md     # Detailed analysis
└── rust_environment_step/             # Experiment 2
    ├── README.md                      # Experiment 2 documentation
    ├── Cargo.toml                     # Rust dependencies
    ├── src/lib.rs                     # Rust implementation
    ├── performance_test.py            # Benchmarks (needs easy_marl)
    └── test_rust_only.py              # Standalone tests
```

## Contributing

To add a new Rust experiment:

1. Create a new directory: `experiments/rust_<component_name>/`
2. Add Rust code with PyO3 bindings
3. Create performance benchmarks
4. Document results in README.md
5. Update this overview

## References

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Build System](https://github.com/PyO3/maturin)
- [Rust-Python Interop Tutorial](https://rust-exercises.com/rust-python-interop/)
- [Numba Documentation](https://numba.pydata.org/)

## Questions?

See individual experiment READMEs for detailed documentation, or check the main repository README for project overview.
