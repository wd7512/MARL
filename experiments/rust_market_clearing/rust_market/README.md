# Rust Market Clearing Extension

This directory contains a Rust implementation of the market clearing algorithm from the MARL bidding environment, exposed to Python via PyO3.

## Overview

The market clearing function is the core computational component of the electricity market simulation. This Rust implementation provides a high-performance alternative to the Python/numba version.

## Implementation Details

The Rust version is a direct translation of the algorithm in `easy_marl/examples/bidding/market.py`:

1. **Sorting**: Sort generators by bid price (ascending)
2. **Cumulative Supply**: Calculate cumulative supply at each price level
3. **Market Clearing**: Find the marginal generator where supply meets demand
4. **Price Setting**: Set market price at the marginal generator's bid
5. **Quantity Allocation**: Allocate cleared quantities to each generator

## Building and Installation

### Prerequisites

1. Install Rust (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Install maturin (Python build tool for Rust extensions):
   ```bash
   pip install maturin
   ```

### Build Options

#### Development Build (Debug)
Fast compilation, slower execution:
```bash
cd rust_market
maturin develop
```

#### Release Build (Optimized)
Slower compilation, fastest execution:
```bash
cd rust_market
maturin develop --release
```

## Usage

After building, the Rust extension can be imported in Python:

```python
import numpy as np
from rust_market import market_clearing_rust

# Example: 5 generators bidding
bids = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
quantities = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
demand = 70.0

price, cleared_quantities = market_clearing_rust(bids, quantities, demand)

print(f"Market Price: ${price:.2f}")
print(f"Cleared Quantities: {cleared_quantities}")
```

## Performance Testing

Run the performance comparison between Python/numba and Rust:

```bash
cd /home/runner/work/MARL/MARL
python performance_test.py
```

This script:
- Tests various problem sizes (5 to 200 generators)
- Runs 1000 iterations per size for statistical reliability
- Includes warmup iterations for JIT compilation
- Verifies correctness of both implementations
- Reports mean, median, std dev, and min times
- Calculates speedup factors

## Expected Performance

The Rust implementation is expected to provide:
- Lower latency for small-scale problems
- Better cache locality and memory efficiency
- Consistent performance without JIT warmup overhead
- Competitive or better performance than numba for this algorithm

Actual performance depends on:
- Problem size (number of generators)
- Hardware (CPU architecture, cache size)
- Whether release or debug build is used

## Integration with MARL Environment

To use the Rust implementation in the MARL environment, modify `easy_marl/examples/bidding/market.py` or `easy_marl/src/environment.py` to import and use `market_clearing_rust` instead of the Python/numba version.

Example:
```python
try:
    from rust_market import market_clearing_rust
    market_clearing = lambda b, q, d: market_clearing_rust(b, q, d)
except ImportError:
    # Fallback to Python/numba implementation
    from easy_marl.examples.bidding.market import market_clearing
```

## References

- [PyO3 Documentation](https://pyo3.rs/)
- [Rust-Python Interoperability Exercises](https://rust-exercises.com/rust-python-interop/)
- [Maturin Documentation](https://github.com/PyO3/maturin)
