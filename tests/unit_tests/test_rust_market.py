"""
Integration test for Rust market clearing implementation.
This test verifies that the Rust implementation produces identical results
to the Python/numba implementation across various scenarios.
"""

import pytest
import numpy as np
from easy_marl.examples.bidding.market import market_clearing


def rust_available():
    """Check if Rust implementation is available."""
    try:
        from rust_market import market_clearing_rust
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not rust_available(), reason="Rust extension not installed")
class TestRustMarketClearing:
    """Test suite for Rust market clearing implementation."""

    def test_that_rust_matches_python_basic(self):
        """Test basic market clearing with simple inputs."""
        from rust_market import market_clearing_rust

        bids = np.array([25.0, 30.0, 35.0, 40.0, 45.0])
        quantities = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
        demand = 70.0

        py_price, py_cleared = market_clearing(bids, quantities, demand)
        rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)
        rust_cleared = np.array(rust_cleared)

        assert np.isclose(py_price, rust_price), "Prices should match"
        assert np.allclose(py_cleared, rust_cleared), "Cleared quantities should match"

    def test_that_rust_matches_python_full_demand(self):
        """Test when demand exceeds total supply."""
        from rust_market import market_clearing_rust

        bids = np.array([20.0, 30.0, 40.0])
        quantities = np.array([10.0, 10.0, 10.0])
        demand = 100.0  # More than total supply of 30

        py_price, py_cleared = market_clearing(bids, quantities, demand)
        rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)
        rust_cleared = np.array(rust_cleared)

        assert np.isclose(py_price, rust_price), "Prices should match"
        assert np.allclose(py_cleared, rust_cleared), "Cleared quantities should match"

    def test_that_rust_matches_python_partial_demand(self):
        """Test with partial demand (only some generators selected)."""
        from rust_market import market_clearing_rust

        bids = np.array([20.0, 30.0, 40.0, 50.0])
        quantities = np.array([25.0, 25.0, 25.0, 25.0])
        demand = 30.0  # Less than one generator's capacity

        py_price, py_cleared = market_clearing(bids, quantities, demand)
        rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)
        rust_cleared = np.array(rust_cleared)

        assert np.isclose(py_price, rust_price), "Prices should match"
        assert np.allclose(py_cleared, rust_cleared), "Cleared quantities should match"

    def test_that_rust_matches_python_random_scenarios(self):
        """Test with multiple random scenarios."""
        from rust_market import market_clearing_rust

        np.random.seed(42)

        for _ in range(10):
            N = np.random.randint(5, 50)
            bids = np.random.uniform(20.0, 100.0, N)
            quantities = np.random.uniform(10.0, 50.0, N)
            demand = np.sum(quantities) * np.random.uniform(0.3, 0.9)

            py_price, py_cleared = market_clearing(bids, quantities, demand)
            rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)
            rust_cleared = np.array(rust_cleared)

            assert np.isclose(py_price, rust_price), f"Prices should match for N={N}"
            assert np.allclose(
                py_cleared, rust_cleared, rtol=1e-5
            ), f"Cleared quantities should match for N={N}"

    def test_that_rust_handles_edge_case_single_generator(self):
        """Test with a single generator."""
        from rust_market import market_clearing_rust

        bids = np.array([30.0])
        quantities = np.array([50.0])
        demand = 25.0

        py_price, py_cleared = market_clearing(bids, quantities, demand)
        rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)
        rust_cleared = np.array(rust_cleared)

        assert np.isclose(py_price, rust_price), "Prices should match"
        assert np.allclose(py_cleared, rust_cleared), "Cleared quantities should match"

    def test_that_rust_handles_zero_demand(self):
        """Test with zero demand."""
        from rust_market import market_clearing_rust

        bids = np.array([20.0, 30.0, 40.0])
        quantities = np.array([10.0, 10.0, 10.0])
        demand = 0.0

        py_price, py_cleared = market_clearing(bids, quantities, demand)
        rust_price, rust_cleared = market_clearing_rust(bids, quantities, demand)
        rust_cleared = np.array(rust_cleared)

        assert np.isclose(py_price, rust_price), "Prices should match"
        assert np.allclose(py_cleared, rust_cleared), "Cleared quantities should match"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
