#!/usr/bin/env python
"""
Simple demonstration of the Rust market clearing extension.
This shows how the Rust implementation can be used as a drop-in replacement
for the Python/numba implementation.
"""

import numpy as np
import time

# Import both implementations
from easy_marl.examples.bidding.market import market_clearing as python_market_clearing

try:
    from rust_market import market_clearing_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  Rust extension not available. Install with:")
    print("   cd rust_market && maturin build --release && pip install target/wheels/rust_market-*.whl")
    print("\nShowing Python implementation only.\n")


def demo_basic_usage():
    """Demonstrate basic market clearing functionality."""
    print("=" * 60)
    print("Basic Market Clearing Example")
    print("=" * 60)
    
    # Example scenario: 5 generators bidding
    bids = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
    quantities = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
    demand = 70.0
    
    print(f"\nGenerator Bids:       {bids}")
    print(f"Generator Quantities: {quantities}")
    print(f"Market Demand:        {demand}")
    
    # Python implementation
    py_price, py_cleared = python_market_clearing(bids, quantities, demand)
    print(f"\n[Python/numba]")
    print(f"  Market Price:        ${py_price:.2f}")
    print(f"  Cleared Quantities:  {py_cleared}")
    print(f"  Total Supply:        {np.sum(py_cleared):.2f}")
    
    # Rust implementation (if available)
    if RUST_AVAILABLE:
        rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)
        rust_cleared = np.array(rust_cleared)
        
        print(f"\n[Rust/PyO3]")
        print(f"  Market Price:        ${rust_price:.2f}")
        print(f"  Cleared Quantities:  {rust_cleared}")
        print(f"  Total Supply:        {np.sum(rust_cleared):.2f}")
        
        # Verify they match
        match = np.isclose(py_price, rust_price) and np.allclose(py_cleared, rust_cleared)
        print(f"\n✅ Results match: {match}")


def demo_performance():
    """Quick performance comparison."""
    if not RUST_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("Quick Performance Comparison")
    print("=" * 60)
    
    N = 50
    bids = np.random.uniform(20.0, 100.0, N)
    quantities = np.random.uniform(10.0, 50.0, N)
    demand = np.sum(quantities) * 0.7
    
    # Warmup
    for _ in range(10):
        python_market_clearing(bids, quantities, demand)
        market_clearing_rust(bids, quantities, demand)
    
    # Python benchmark
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        python_market_clearing(bids, quantities, demand)
    py_time = (time.perf_counter() - start) / iterations * 1e6  # microseconds
    
    # Rust benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        market_clearing_rust(bids, quantities, demand)
    rust_time = (time.perf_counter() - start) / iterations * 1e6  # microseconds
    
    print(f"\nProblem size: {N} generators")
    print(f"Iterations:   {iterations}")
    print(f"\nPython/numba: {py_time:.3f} μs per call")
    print(f"Rust/PyO3:    {rust_time:.3f} μs per call")
    print(f"\nSpeedup:      {py_time/rust_time:.2f}x")
    
    if rust_time < py_time:
        print(f"✅ Rust is {py_time/rust_time:.2f}x faster")
    else:
        print(f"ℹ️  Python/numba is {rust_time/py_time:.2f}x faster")


def demo_integration():
    """Show how to use Rust as a drop-in replacement."""
    if not RUST_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("Drop-in Replacement Pattern")
    print("=" * 60)
    
    code = '''
# Option 1: Conditional import with fallback
try:
    from rust_market import market_clearing_rust
    market_clearing = lambda b, q, d: market_clearing_rust(b, q, d)
    print("Using Rust implementation")
except ImportError:
    from easy_marl.examples.bidding.market import market_clearing
    print("Using Python implementation")

# Option 2: Environment variable control
import os
if os.getenv("USE_RUST_MARKET", "0") == "1":
    from rust_market import market_clearing_rust as market_clearing
else:
    from easy_marl.examples.bidding.market import market_clearing
'''
    
    print("\nExample integration patterns:")
    print(code)


def main():
    """Run all demonstrations."""
    print("\n🦀 Rust-Python Interop Demonstration 🐍\n")
    
    demo_basic_usage()
    demo_performance()
    demo_integration()
    
    print("\n" + "=" * 60)
    print("For more details, see:")
    print("  - RUST_PERFORMANCE_STUDY.md")
    print("  - rust_market/README.md")
    print("  - performance_test.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
