"""
Performance test for Rust environment step implementation.

This test compares:
1. Python environment step (multiple function calls)
2. Rust environment step (single combined function)

The Rust version should be faster because it:
- Avoids multiple Python-Rust boundary crossings
- Keeps data in Rust without copying
- Can optimize across operations
"""

import time
import numpy as np
from typing import Callable, List
import statistics

# Import Python implementations
from easy_marl.examples.bidding.market import market_clearing


def python_env_step(bids, quantities, costs, demand, lambda_bid_penalty, unit_lol_penalty,
                   demand_scale, cost_scale, time_horizon):
    """Python version of environment step."""
    # Market clearing
    price, q_cleared = market_clearing(bids, quantities, demand)
    
    # Rewards
    base_rewards = (price - costs) * q_cleared
    bid_penalties = lambda_bid_penalty * (bids - costs) ** 2
    
    total_cleared = np.sum(q_cleared)
    loss_of_load_penalty = unit_lol_penalty * max(0, demand - total_cleared)
    
    rewards = base_rewards - bid_penalties - loss_of_load_penalty
    
    # Scale reward
    rewards /= demand_scale * cost_scale * max(1, time_horizon)
    rewards *= 20
    
    return price, q_cleared, rewards


def benchmark_env_step(
    n_generators_list: List[int],
    num_iterations: int = 1000,
) -> dict:
    """Benchmark environment step functions."""
    print(f"\n{'='*60}")
    print("Environment Step Performance Test")
    print(f"{'='*60}")
    
    # Check if Rust implementation is available
    try:
        from rust_env_step import environment_step_rust
        rust_available = True
    except ImportError:
        rust_available = False
        print("\n⚠️  Rust implementation not available")
        print("Run: cd experiments/rust_environment_step && maturin build --release")
        print("     pip install target/wheels/rust_env_step-*.whl\n")
        return {}
    
    results = {'python': {}, 'rust': {}}
    
    for N in n_generators_list:
        print(f"\n{'='*60}")
        print(f"Testing with N={N} generators")
        print(f"{'='*60}")
        
        # Generate test data
        bids = np.random.uniform(20.0, 100.0, N).astype(np.float64)
        quantities = np.random.uniform(10.0, 50.0, N).astype(np.float64)
        costs = np.random.uniform(15.0, 80.0, N).astype(np.float64)
        demand = np.sum(quantities) * 0.7
        
        lambda_bid_penalty = 0.01
        unit_lol_penalty = 1.0
        demand_scale = 100.0
        cost_scale = 50.0
        time_horizon = 24
        
        # Warmup
        for _ in range(10):
            python_env_step(bids, quantities, costs, demand, lambda_bid_penalty,
                          unit_lol_penalty, demand_scale, cost_scale, time_horizon)
            environment_step_rust(bids, quantities, costs, demand, lambda_bid_penalty,
                                unit_lol_penalty, demand_scale, cost_scale, time_horizon)
        
        # Benchmark Python
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            python_env_step(bids, quantities, costs, demand, lambda_bid_penalty,
                          unit_lol_penalty, demand_scale, cost_scale, time_horizon)
            end = time.perf_counter()
            times.append((end - start) * 1e6)
        
        py_mean = statistics.mean(times)
        py_median = statistics.median(times)
        results['python'][N] = {'mean_us': py_mean, 'median_us': py_median}
        
        print(f"Python:   mean={py_mean:8.3f}μs, median={py_median:8.3f}μs")
        
        # Benchmark Rust
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            environment_step_rust(bids, quantities, costs, demand, lambda_bid_penalty,
                                unit_lol_penalty, demand_scale, cost_scale, time_horizon)
            end = time.perf_counter()
            times.append((end - start) * 1e6)
        
        rust_mean = statistics.mean(times)
        rust_median = statistics.median(times)
        results['rust'][N] = {'mean_us': rust_mean, 'median_us': rust_median}
        
        print(f"Rust:     mean={rust_mean:8.3f}μs, median={rust_median:8.3f}μs")
        print(f"Speedup:  {py_mean/rust_mean:.2f}x")
        
        # Verify correctness
        py_price, py_cleared, py_rewards = python_env_step(
            bids, quantities, costs, demand, lambda_bid_penalty,
            unit_lol_penalty, demand_scale, cost_scale, time_horizon
        )
        rust_price, rust_cleared, rust_rewards = environment_step_rust(
            bids, quantities, costs, demand, lambda_bid_penalty,
            unit_lol_penalty, demand_scale, cost_scale, time_horizon
        )
        rust_cleared = np.array(rust_cleared)
        rust_rewards = np.array(rust_rewards)
        
        price_match = np.isclose(py_price, rust_price)
        cleared_match = np.allclose(py_cleared, rust_cleared)
        rewards_match = np.allclose(py_rewards, rust_rewards)
        
        if price_match and cleared_match and rewards_match:
            print("Correctness: ✅ PASS")
        else:
            print(f"Correctness: ❌ FAIL (price={price_match}, cleared={cleared_match}, rewards={rewards_match})")
    
    return results


def benchmark_observers(num_iterations: int = 10000):
    """Benchmark observer functions."""
    print(f"\n{'='*60}")
    print("Observer Function Performance Test")
    print(f"{'='*60}")
    
    try:
        from rust_env_step import simple_observer_rust, simple_observer_v3_rust
        rust_available = True
    except ImportError:
        rust_available = False
        print("\n⚠️  Rust implementation not available\n")
        return
    
    # Test data
    demand_t = 75.0
    capacity_i = 30.0
    cost_i = 35.0
    t = 15
    demand_scale = 100.0
    capacity_scale = 50.0
    cost_scale = 50.0
    
    # Python observer (inline for comparison)
    def python_simple_observer():
        hour = t % 24
        return np.array([
            demand_t / demand_scale,
            capacity_i / capacity_scale,
            cost_i / cost_scale,
            hour / 24.0
        ])
    
    def python_simple_observer_v3():
        hour_of_day = t % 24
        hour_x = np.sin(2 * np.pi * hour_of_day / 24)
        hour_y = np.cos(2 * np.pi * hour_of_day / 24)
        
        day_of_week = (t // 24) % 7
        day_x = np.sin(2 * np.pi * day_of_week / 7)
        day_y = np.cos(2 * np.pi * day_of_week / 7)
        
        return np.array([
            demand_t / demand_scale,
            capacity_i / capacity_scale,
            cost_i / cost_scale,
            hour_x, hour_y, day_x, day_y
        ])
    
    # Warmup
    for _ in range(100):
        python_simple_observer()
        simple_observer_rust(demand_t, capacity_i, cost_i, t,
                            demand_scale, capacity_scale, cost_scale)
    
    print("\n--- Simple Observer ---")
    
    # Python
    start = time.perf_counter()
    for _ in range(num_iterations):
        python_simple_observer()
    py_time = (time.perf_counter() - start) / num_iterations * 1e6
    
    # Rust
    start = time.perf_counter()
    for _ in range(num_iterations):
        simple_observer_rust(demand_t, capacity_i, cost_i, t,
                            demand_scale, capacity_scale, cost_scale)
    rust_time = (time.perf_counter() - start) / num_iterations * 1e6
    
    print(f"Python: {py_time:.3f}μs")
    print(f"Rust:   {rust_time:.3f}μs")
    print(f"Speedup: {py_time/rust_time:.2f}x")
    
    # Warmup v3
    for _ in range(100):
        python_simple_observer_v3()
        simple_observer_v3_rust(demand_t, capacity_i, cost_i, t,
                               demand_scale, capacity_scale, cost_scale)
    
    print("\n--- Simple Observer V3 (with trig) ---")
    
    # Python
    start = time.perf_counter()
    for _ in range(num_iterations):
        python_simple_observer_v3()
    py_time = (time.perf_counter() - start) / num_iterations * 1e6
    
    # Rust
    start = time.perf_counter()
    for _ in range(num_iterations):
        simple_observer_v3_rust(demand_t, capacity_i, cost_i, t,
                               demand_scale, capacity_scale, cost_scale)
    rust_time = (time.perf_counter() - start) / num_iterations * 1e6
    
    print(f"Python: {py_time:.3f}μs")
    print(f"Rust:   {rust_time:.3f}μs")
    print(f"Speedup: {py_time/rust_time:.2f}x")


def print_summary(results):
    """Print summary of results."""
    if not results:
        return
    
    print(f"\n{'='*60}")
    print("Summary: Environment Step Performance")
    print(f"{'='*60}")
    print(f"{'N':>5} | {'Python (μs)':>12} | {'Rust (μs)':>12} | {'Speedup':>8}")
    print(f"{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    
    speedups = []
    for N in sorted(results['python'].keys()):
        py_time = results['python'][N]['mean_us']
        rust_time = results['rust'][N]['mean_us']
        speedup = py_time / rust_time
        speedups.append(speedup)
        
        print(f"{N:5d} | {py_time:12.3f} | {rust_time:12.3f} | {speedup:8.2f}x")
    
    avg_speedup = statistics.mean(speedups)
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.0:
        print(f"✅ Rust is {avg_speedup:.2f}x faster on average")
    else:
        print(f"⚠️  Python is {1/avg_speedup:.2f}x faster on average")


def main():
    """Run all benchmarks."""
    print("Environment Step + Observer Performance Tests")
    print("=" * 60)
    
    # Test environment step
    n_generators_list = [5, 10, 20, 50, 100]
    results = benchmark_env_step(n_generators_list, num_iterations=1000)
    print_summary(results)
    
    # Test observers
    benchmark_observers(num_iterations=10000)
    
    print(f"\n{'='*60}")
    print("Test Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
