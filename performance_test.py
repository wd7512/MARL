"""
Performance testing script comparing Python (numba) vs Rust implementations
of the market clearing algorithm used in the bidding environment.

As per requirements, training is ignored - we only benchmark the market clearing
function which is the core computational component.
"""

import time
import numpy as np
from typing import Callable, List, Tuple
import statistics

# Import Python/numba implementation
from easy_marl.examples.bidding.market import market_clearing


def benchmark_market_clearing(
    market_clearing_fn: Callable,
    n_generators_list: List[int],
    num_iterations: int = 1000,
    name: str = "Unknown"
) -> dict:
    """
    Benchmark market clearing function with different problem sizes.
    
    Args:
        market_clearing_fn: Function to benchmark
        n_generators_list: List of generator counts to test
        num_iterations: Number of iterations per size
        name: Name of the implementation
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    results = {}
    
    for N in n_generators_list:
        # Generate random test data
        bids = np.random.uniform(20.0, 100.0, N).astype(np.float64)
        quantities = np.random.uniform(10.0, 50.0, N).astype(np.float64)
        demand = np.sum(quantities) * 0.7  # 70% of total capacity
        
        # Warmup (important for JIT compilation)
        for _ in range(10):
            _ = market_clearing_fn(bids, quantities, demand)
        
        # Actual benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = market_clearing_fn(bids, quantities, demand)
            end = time.perf_counter()
            times.append((end - start) * 1e6)  # Convert to microseconds
        
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_time = statistics.stdev(times)
        min_time = min(times)
        
        results[N] = {
            'mean_us': mean_time,
            'median_us': median_time,
            'std_us': std_time,
            'min_us': min_time,
        }
        
        print(f"N={N:3d} generators: "
              f"mean={mean_time:8.3f}μs, "
              f"median={median_time:8.3f}μs, "
              f"std={std_time:7.3f}μs, "
              f"min={min_time:7.3f}μs")
    
    return results


def compare_implementations(
    python_results: dict,
    rust_results: dict,
) -> None:
    """
    Compare and print speedup between Python and Rust implementations.
    
    Args:
        python_results: Benchmark results from Python implementation
        rust_results: Benchmark results from Rust implementation
    """
    print(f"\n{'='*60}")
    print("Performance Comparison (Python/numba vs Rust)")
    print(f"{'='*60}")
    print(f"{'N':>5} | {'Python (μs)':>12} | {'Rust (μs)':>12} | {'Speedup':>8}")
    print(f"{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    
    for N in sorted(python_results.keys()):
        py_time = python_results[N]['mean_us']
        rust_time = rust_results[N]['mean_us']
        speedup = py_time / rust_time
        
        print(f"{N:5d} | {py_time:12.3f} | {rust_time:12.3f} | {speedup:8.2f}x")
    
    # Calculate average speedup
    speedups = [python_results[N]['mean_us'] / rust_results[N]['mean_us'] 
                for N in python_results.keys()]
    avg_speedup = statistics.mean(speedups)
    
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.0:
        print(f"✅ Rust is {avg_speedup:.2f}x faster on average")
    else:
        print(f"⚠️  Python/numba is {1/avg_speedup:.2f}x faster on average")


def main():
    """Main performance testing routine"""
    print("Performance Test: Market Clearing Algorithm")
    print("Comparing Python (numba) vs Rust implementations")
    print("\nTest Configuration:")
    print("- Testing different problem sizes (number of generators)")
    print("- 1000 iterations per size for statistical reliability")
    print("- Warmup iterations to account for JIT compilation")
    
    # Test with various problem sizes
    n_generators_list = [5, 10, 20, 50, 100, 200]
    num_iterations = 1000
    
    # Benchmark Python/numba implementation
    python_results = benchmark_market_clearing(
        market_clearing,
        n_generators_list,
        num_iterations,
        name="Python (numba JIT)"
    )
    
    # Try to import and benchmark Rust implementation
    try:
        from rust_market import market_clearing_rust
        
        # Wrapper to match the interface
        def rust_wrapper(bids, quantities, demand):
            price, cleared = market_clearing_rust(bids, quantities, demand)
            return price, np.array(cleared)
        
        rust_results = benchmark_market_clearing(
            rust_wrapper,
            n_generators_list,
            num_iterations,
            name="Rust (PyO3)"
        )
        
        # Compare results
        compare_implementations(python_results, rust_results)
        
        # Verify correctness
        print(f"\n{'='*60}")
        print("Correctness Verification")
        print(f"{'='*60}")
        
        for N in [5, 10, 50]:
            bids = np.random.uniform(20.0, 100.0, N).astype(np.float64)
            quantities = np.random.uniform(10.0, 50.0, N).astype(np.float64)
            demand = np.sum(quantities) * 0.7
            
            py_price, py_cleared = market_clearing(bids, quantities, demand)
            rust_price, rust_cleared = rust_wrapper(bids, quantities, demand)
            
            price_match = np.isclose(py_price, rust_price)
            cleared_match = np.allclose(py_cleared, rust_cleared)
            
            status = "✅ PASS" if price_match and cleared_match else "❌ FAIL"
            print(f"N={N:3d}: {status} (price_match={price_match}, cleared_match={cleared_match})")
        
    except ImportError:
        print("\n" + "="*60)
        print("⚠️  Rust implementation not available")
        print("="*60)
        print("To build and install the Rust extension:")
        print("1. Install Rust: https://rustup.rs/")
        print("2. Install maturin: pip install maturin")
        print("3. Build the extension: cd rust_market && maturin develop --release")
        print("4. Re-run this script")
        print("\nShowing Python/numba results only.")


if __name__ == "__main__":
    main()
