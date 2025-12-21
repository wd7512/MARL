# Rust-Python Interop Performance Study

## Overview

This document summarizes the exploration of Rust for the MARL electricity market environment, following the tutorial at https://rust-exercises.com/rust-python-interop/.

## Implementation

### What Was Implemented

1. **Rust Market Clearing Module** (`rust_market/`)
   - Direct Rust translation of the market clearing algorithm from `market.py`
   - Uses PyO3 for Python bindings
   - Built with maturin as a Python extension module
   - Fully compatible with NumPy arrays

2. **Performance Testing Script** (`performance_test.py`)
   - Compares Python (numba JIT) vs Rust implementations
   - Tests multiple problem sizes (5-200 generators)
   - 1000 iterations per size for statistical reliability
   - Includes warmup phase for JIT compilation
   - Validates correctness of both implementations

### Market Clearing Algorithm

The algorithm sorts generators by bid price and allocates generation capacity to meet demand:

```
1. Sort generators by bid price (ascending)
2. Calculate cumulative supply at each price level
3. Find marginal generator where supply meets demand
4. Set market price at marginal generator's bid
5. Allocate cleared quantities to each generator
```

## Performance Results

### Benchmark Configuration
- **Hardware**: GitHub Actions runner (Linux x86_64)
- **Python**: 3.12
- **Rust**: 1.92.0
- **Numba**: 0.63.1
- **PyO3**: 0.21
- **Iterations**: 1000 per problem size
- **Build**: Release mode with optimizations

### Results Summary

| Generators | Python (numba) | Rust (PyO3) | Speedup |
|------------|----------------|-------------|---------|
| 5          | 1.59 μs        | 1.28 μs     | 1.24x   |
| 10         | 1.62 μs        | 1.54 μs     | 1.05x   |
| 20         | 1.73 μs        | 2.24 μs     | 0.77x   |
| 50         | 2.09 μs        | 4.48 μs     | 0.47x   |
| 100        | 2.74 μs        | 7.57 μs     | 0.36x   |
| 200        | 4.53 μs        | 14.95 μs    | 0.30x   |

**Average Speedup**: 0.70x (Rust is ~1.43x slower)

### Key Findings

1. **Rust is faster for small problems** (5-10 generators)
   - Lower overhead for simple cases
   - 1.24x speedup for 5 generators

2. **Python/numba scales better** (20+ generators)
   - Better optimization for larger problem sizes
   - 3.3x faster at 200 generators

3. **Both implementations are very fast**
   - Even at 200 generators: <15 μs per call
   - Not a bottleneck for the MARL training process

## Analysis

### Why is Numba Faster?

1. **JIT Specialization**: Numba compiles specialized versions for specific data types
2. **LLVM Optimization**: Both use LLVM, but numba may have better type inference
3. **Memory Layout**: Numba can optimize for contiguous NumPy arrays
4. **Cache Efficiency**: Better vectorization for sorting and cumsum operations
5. **PyO3 Overhead**: Some overhead from Python-Rust boundary crossing

### When to Use Rust?

Rust would be beneficial when:
- More complex algorithms beyond numba's scope
- Need for memory safety guarantees
- Parallel processing with fearless concurrency
- Integration with existing Rust ecosystem
- Stateful computations that benefit from Rust's type system

### When to Use Numba?

Numba is better for:
- Simple numerical algorithms (like this one)
- Heavy NumPy array operations
- Quick prototyping without compilation overhead
- When Python integration is seamless

## Educational Value

This exercise successfully demonstrated:

1. ✅ **PyO3 Usage**: Building Python extensions with Rust
2. ✅ **Maturin Build System**: Creating Python packages from Rust
3. ✅ **NumPy Integration**: Working with NumPy arrays in Rust
4. ✅ **Performance Benchmarking**: Rigorous comparison methodology
5. ✅ **Correctness Validation**: Ensuring algorithm equivalence

## Recommendations

### For This Project

**Keep using Python/numba** for the market clearing algorithm:
- Already well-optimized (1-5 μs per call)
- Not a performance bottleneck
- Simpler to maintain
- No build/compilation requirements

### Potential Rust Use Cases

Consider Rust for:
1. **Custom Gymnasium Environments** with complex state logic
2. **Large-scale simulations** requiring parallel processing
3. **Memory-intensive computations** where safety matters
4. **Integration with high-performance libraries** (e.g., BLAS/LAPACK)

## Building and Testing

### Build Rust Extension
```bash
cd rust_market
maturin build --release
pip install target/wheels/rust_market-*.whl
```

### Run Performance Tests
```bash
python performance_test.py
```

### Test Correctness
```python
import numpy as np
from rust_market import market_clearing_rust
from easy_marl.examples.bidding.market import market_clearing

bids = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
quantities = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
demand = 70.0

# Both should produce identical results
py_price, py_cleared = market_clearing(bids, quantities, demand)
rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)

assert np.isclose(py_price, rust_price)
assert np.allclose(py_cleared, rust_cleared)
```

## Conclusion

This exploration successfully implemented and benchmarked a Rust version of the market clearing algorithm using PyO3. While Rust provides excellent performance for small problem sizes, Python with numba JIT compilation proved to be more effective for this specific algorithm, especially as the problem size grows.

The exercise provided valuable learning about:
- Rust-Python interoperability via PyO3
- Performance characteristics of JIT vs AOT compilation
- When to choose Rust vs Python for numerical computing
- Proper benchmarking methodology

The implementation remains available in the repository as a reference for future Rust integration opportunities where it might provide more significant benefits.

## References

- [Rust-Python Interoperability Tutorial](https://rust-exercises.com/rust-python-interop/)
- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://github.com/PyO3/maturin)
- [Numba Documentation](https://numba.pydata.org/)
