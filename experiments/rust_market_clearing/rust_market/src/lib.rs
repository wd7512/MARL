use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

/// Market clearing algorithm implemented in Rust for performance
/// This is a direct translation of the Python/numba version in market.py
#[pyfunction]
fn market_clearing_rust(
    bids: PyReadonlyArray1<f64>,
    quantities: PyReadonlyArray1<f64>,
    demand: f64,
) -> PyResult<(f64, Vec<f64>)> {
    let bids = bids.as_slice()?;
    let quantities = quantities.as_slice()?;
    let n = bids.len();
    
    // Create sorted indices based on bids
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        bids[a].partial_cmp(&bids[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Sort bids and quantities
    let bids_sorted: Vec<f64> = order.iter().map(|&i| bids[i]).collect();
    let q_sorted: Vec<f64> = order.iter().map(|&i| quantities[i]).collect();
    
    // Compute cumulative supply
    let mut cum_supply = vec![0.0; n];
    cum_supply[0] = q_sorted[0];
    for i in 1..n {
        cum_supply[i] = cum_supply[i - 1] + q_sorted[i];
    }
    
    // Find the first index where cumulative supply meets/exceeds demand
    let m = cum_supply.iter().position(|&cs| cs >= demand).unwrap_or(n);
    
    // Initialize cleared quantities
    let mut q_cleared = vec![0.0; n];
    let p_t: f64;
    
    if m >= n {
        // Demand exceeds total supply
        q_cleared.copy_from_slice(&q_sorted);
        p_t = bids_sorted[n - 1];
    } else {
        // Fill cleared quantities
        for i in 0..m {
            q_cleared[i] = q_sorted[i];
        }
        if m > 0 {
            q_cleared[m] = demand - cum_supply[m - 1];
        } else {
            q_cleared[m] = demand;
        }
        p_t = bids_sorted[m];
    }
    
    // Reorder q_cleared to original order
    let mut q_cleared_final = vec![0.0; n];
    for (i, &idx) in order.iter().enumerate() {
        q_cleared_final[idx] = q_cleared[i];
    }
    
    Ok((p_t, q_cleared_final))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_market(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(market_clearing_rust, m)?)?;
    Ok(())
}
