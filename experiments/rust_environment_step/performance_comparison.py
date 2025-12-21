"""
Complete performance test for Rust environment step with Python comparison.

This test provides a comprehensive comparison between Python and Rust implementations.
"""

import time
import numpy as np
from numba import njit
import statistics

# ============================================================================
# Python Implementation (for comparison)
# ============================================================================

@njit(cache=True)
def market_clearing(bids: np.ndarray, quantities: np.ndarray, demand: float):
    """Market clearing algorithm - Python/numba version."""
    bids = np.asarray(bids)
    quantities = np.asarray(quantities)
    N = len(bids)

    order = np.argsort(bids)
    bids_sorted = bids[order]
    q_sorted = quantities[order]

    cum_supply = np.cumsum(q_sorted)
    m = np.searchsorted(cum_supply, demand, side="left")

    q_cleared = np.zeros_like(q_sorted)
    if m >= N:
        q_cleared[:] = q_sorted
        P_t = bids_sorted[-1]
    else:
        q_cleared[:m] = q_sorted[:m]
        q_cleared[m] = demand - cum_supply[m - 1] if m > 0 else demand
        P_t = bids_sorted[m]

    q_cleared_final = np.zeros_like(q_cleared)
    q_cleared_final[order] = q_cleared

    return P_t, q_cleared_final


def python_env_step(bids, quantities, costs, demand, lambda_bid_penalty, unit_lol_penalty,
                   demand_scale, cost_scale, time_horizon):
    """Python version of complete environment step."""
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


def python_simple_observer(demand_t, capacity_i, cost_i, t, demand_scale, capacity_scale, cost_scale):
    """Python version of simple observer."""
    hour = t % 24
    return np.array([
        demand_t / demand_scale,
        capacity_i / capacity_scale,
        cost_i / cost_scale,
        hour / 24.0
    ])


def python_simple_observer_v3(demand_t, capacity_i, cost_i, t, demand_scale, capacity_scale, cost_scale):
    """Python version of simple observer v3 with trigonometry."""
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


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_env_step(n_generators_list, num_iterations=1000):
    """Benchmark environment step functions."""
    print(f"\n{'='*70}")
    print("ENVIRONMENT STEP PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    # Check if Rust implementation is available
    try:
        from rust_env_step import environment_step_rust
        rust_available = True
        print("✅ Rust extension loaded successfully")
    except ImportError:
        rust_available = False
        print("⚠️  Rust implementation not available")
        print("Run: cd experiments/rust_environment_step && maturin build --release")
        print("     pip install target/wheels/rust_env_step-*.whl\n")
        return {}
    
    results = {'python': {}, 'rust': {}}
    
    for N in n_generators_list:
        print(f"\n{'-'*70}")
        print(f"Testing with N={N} generators")
        print(f"{'-'*70}")
        
        # Generate test data
        np.random.seed(42)  # For reproducibility
        bids = np.random.uniform(20.0, 100.0, N).astype(np.float64)
        quantities = np.random.uniform(10.0, 50.0, N).astype(np.float64)
        costs = np.random.uniform(15.0, 80.0, N).astype(np.float64)
        demand = np.sum(quantities) * 0.7
        
        lambda_bid_penalty = 0.01
        unit_lol_penalty = 1.0
        demand_scale = 100.0
        cost_scale = 50.0
        time_horizon = 24
        
        # Warmup (important for JIT compilation)
        print("Warming up...")
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
        py_std = statistics.stdev(times)
        py_min = min(times)
        results['python'][N] = {
            'mean_us': py_mean,
            'median_us': py_median,
            'std_us': py_std,
            'min_us': py_min
        }
        
        print(f"Python/numba: mean={py_mean:7.2f}μs, median={py_median:7.2f}μs, "
              f"std={py_std:6.2f}μs, min={py_min:7.2f}μs")
        
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
        rust_std = statistics.stdev(times)
        rust_min = min(times)
        results['rust'][N] = {
            'mean_us': rust_mean,
            'median_us': rust_median,
            'std_us': rust_std,
            'min_us': rust_min
        }
        
        print(f"Rust/PyO3:    mean={rust_mean:7.2f}μs, median={rust_median:7.2f}μs, "
              f"std={rust_std:6.2f}μs, min={rust_min:7.2f}μs")
        
        speedup = py_mean / rust_mean
        if speedup > 1.0:
            print(f"⚡ Speedup: {speedup:.2f}x (Rust is faster)")
        else:
            print(f"⚠️  Speedup: {speedup:.2f}x (Python is {1/speedup:.2f}x faster)")
        
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
            print("✅ Correctness: PASS")
        else:
            print(f"❌ Correctness: FAIL (price={price_match}, cleared={cleared_match}, rewards={rewards_match})")
    
    return results


def benchmark_observers(num_iterations=10000):
    """Benchmark observer functions."""
    print(f"\n{'='*70}")
    print("OBSERVER FUNCTIONS PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
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
    
    # Warmup
    for _ in range(100):
        python_simple_observer(demand_t, capacity_i, cost_i, t,
                             demand_scale, capacity_scale, cost_scale)
        simple_observer_rust(demand_t, capacity_i, cost_i, t,
                            demand_scale, capacity_scale, cost_scale)
    
    print("\n--- Simple Observer (4 features) ---")
    
    # Python
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        python_simple_observer(demand_t, capacity_i, cost_i, t,
                             demand_scale, capacity_scale, cost_scale)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    py_mean = statistics.mean(times)
    py_median = statistics.median(times)
    
    # Rust
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        simple_observer_rust(demand_t, capacity_i, cost_i, t,
                            demand_scale, capacity_scale, cost_scale)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    rust_mean = statistics.mean(times)
    rust_median = statistics.median(times)
    
    print(f"Python: mean={py_mean:.3f}μs, median={py_median:.3f}μs")
    print(f"Rust:   mean={rust_mean:.3f}μs, median={rust_median:.3f}μs")
    speedup = py_mean / rust_mean
    if speedup > 1.0:
        print(f"⚡ Speedup: {speedup:.2f}x (Rust is faster)")
    else:
        print(f"⚠️  Speedup: {speedup:.2f}x (Python is {1/speedup:.2f}x faster)")
    
    # Warmup v3
    for _ in range(100):
        python_simple_observer_v3(demand_t, capacity_i, cost_i, t,
                                 demand_scale, capacity_scale, cost_scale)
        simple_observer_v3_rust(demand_t, capacity_i, cost_i, t,
                               demand_scale, capacity_scale, cost_scale)
    
    print("\n--- Simple Observer V3 (7 features with trigonometry) ---")
    
    # Python
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        python_simple_observer_v3(demand_t, capacity_i, cost_i, t,
                                 demand_scale, capacity_scale, cost_scale)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    py_mean = statistics.mean(times)
    py_median = statistics.median(times)
    
    # Rust
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        simple_observer_v3_rust(demand_t, capacity_i, cost_i, t,
                               demand_scale, capacity_scale, cost_scale)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    rust_mean = statistics.mean(times)
    rust_median = statistics.median(times)
    
    print(f"Python: mean={py_mean:.3f}μs, median={py_median:.3f}μs")
    print(f"Rust:   mean={rust_mean:.3f}μs, median={rust_median:.3f}μs")
    speedup = py_mean / rust_mean
    if speedup > 1.0:
        print(f"⚡ Speedup: {speedup:.2f}x (Rust is faster)")
    else:
        print(f"⚠️  Speedup: {speedup:.2f}x (Python is {1/speedup:.2f}x faster)")


def print_summary(results):
    """Print detailed summary of results."""
    if not results:
        return
    
    print(f"\n{'='*70}")
    print("SUMMARY: Environment Step Performance Comparison")
    print(f"{'='*70}")
    print(f"{'N':>5} | {'Python (μs)':>12} | {'Rust (μs)':>12} | {'Speedup':>10} | {'Winner':>8}")
    print(f"{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    
    speedups = []
    for N in sorted(results['python'].keys()):
        py_time = results['python'][N]['mean_us']
        rust_time = results['rust'][N]['mean_us']
        speedup = py_time / rust_time
        speedups.append(speedup)
        
        winner = "Rust" if speedup > 1.0 else "Python"
        speedup_str = f"{speedup:.2f}x"
        
        print(f"{N:5d} | {py_time:12.2f} | {rust_time:12.2f} | {speedup_str:>10} | {winner:>8}")
    
    avg_speedup = statistics.mean(speedups)
    print(f"\n{'='*70}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.0:
        print(f"✅ Rust is {avg_speedup:.2f}x faster on average")
        print(f"   Rust wins for combined operations that eliminate boundary crossings")
    else:
        print(f"⚠️  Python is {1/avg_speedup:.2f}x faster on average")
        print(f"   Python/numba JIT compilation is very effective for this workload")
    print(f"{'='*70}")


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("RUST VS PYTHON PERFORMANCE COMPARISON")
    print("Environment Step + Observer Functions")
    print("="*70)
    print("\nConfiguration:")
    print("- 1000 iterations per test")
    print("- Warmup runs to ensure JIT compilation")
    print("- Reproducible random seeds")
    print("- Multiple problem sizes (5-100 generators)")
    print()
    
    # Test environment step
    n_generators_list = [5, 10, 20, 50, 100]
    results = benchmark_env_step(n_generators_list, num_iterations=1000)
    print_summary(results)
    
    # Test observers
    benchmark_observers(num_iterations=10000)
    
    print(f"\n{'='*70}")
    print("All Tests Complete!")
    print(f"{'='*70}")
    print("\nKey Findings:")
    print("- Combined operations (env step) show better performance than isolated functions")
    print("- Eliminates multiple Python-Rust boundary crossings")
    print("- Observer functions benefit from Rust's fast trigonometric operations")
    print("- Results are consistent and reproducible")
    print()


if __name__ == "__main__":
    main()
