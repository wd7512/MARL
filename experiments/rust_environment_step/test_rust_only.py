"""
Standalone performance test for Rust environment step.
Tests just the Rust implementation without needing easy_marl installed.
"""

import time
import numpy as np
import statistics

try:
    from rust_env_step import environment_step_rust, simple_observer_rust, simple_observer_v3_rust
    print("✅ Rust extension loaded successfully\n")
except ImportError as e:
    print(f"❌ Failed to import Rust extension: {e}")
    print("\nTo install:")
    print("  cd experiments/rust_environment_step")
    print("  maturin build --release")
    print("  pip install target/wheels/rust_env_step-*.whl")
    exit(1)


def test_env_step():
    """Test environment step function."""
    print("="*60)
    print("Environment Step Performance Test")
    print("="*60)
    
    n_list = [5, 10, 20, 50, 100]
    
    for N in n_list:
        # Generate test data
        bids = np.random.uniform(20.0, 100.0, N).astype(np.float64)
        quantities = np.random.uniform(10.0, 50.0, N).astype(np.float64)
        costs = np.random.uniform(15.0, 80.0, N).astype(np.float64)
        demand = np.sum(quantities) * 0.7
        
        # Warmup
        for _ in range(10):
            environment_step_rust(bids, quantities, costs, demand, 
                                0.01, 1.0, 100.0, 50.0, 24)
        
        # Benchmark
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            price, cleared, rewards = environment_step_rust(
                bids, quantities, costs, demand,
                0.01, 1.0, 100.0, 50.0, 24
            )
            end = time.perf_counter()
            times.append((end - start) * 1e6)
        
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        
        print(f"N={N:3d}: mean={mean_time:7.2f}μs, median={median_time:7.2f}μs")
        
        # Sanity check
        assert isinstance(price, float)
        assert len(cleared) == N
        assert len(rewards) == N
    
    print("\n✅ All tests passed!")


def test_observers():
    """Test observer functions."""
    print("\n" + "="*60)
    print("Observer Functions Performance Test")
    print("="*60)
    
    # Test data
    demand_t = 75.0
    capacity_i = 30.0
    cost_i = 35.0
    t = 15
    demand_scale = 100.0
    capacity_scale = 50.0
    cost_scale = 50.0
    
    print("\n--- Simple Observer (4 features) ---")
    
    # Warmup
    for _ in range(100):
        simple_observer_rust(demand_t, capacity_i, cost_i, t,
                           demand_scale, capacity_scale, cost_scale)
    
    # Benchmark
    times = []
    for _ in range(10000):
        start = time.perf_counter()
        obs = simple_observer_rust(demand_t, capacity_i, cost_i, t,
                                   demand_scale, capacity_scale, cost_scale)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    
    mean_time = statistics.mean(times)
    print(f"Mean: {mean_time:.3f}μs")
    print(f"Result: {obs}")
    
    print("\n--- Simple Observer V3 (7 features with trig) ---")
    
    # Warmup
    for _ in range(100):
        simple_observer_v3_rust(demand_t, capacity_i, cost_i, t,
                               demand_scale, capacity_scale, cost_scale)
    
    # Benchmark
    times = []
    for _ in range(10000):
        start = time.perf_counter()
        obs = simple_observer_v3_rust(demand_t, capacity_i, cost_i, t,
                                      demand_scale, capacity_scale, cost_scale)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    
    mean_time = statistics.mean(times)
    print(f"Mean: {mean_time:.3f}μs")
    print(f"Result: {obs}")
    
    print("\n✅ All observer tests passed!")


if __name__ == "__main__":
    test_env_step()
    test_observers()
    
    print("\n" + "="*60)
    print("All Tests Complete!")
    print("="*60)
    print("\nFor comparison with Python implementation,")
    print("run: python performance_test.py")
