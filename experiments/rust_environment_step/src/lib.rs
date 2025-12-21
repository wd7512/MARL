use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

/// Complete environment step function in Rust
/// 
/// This combines multiple operations:
/// - Bid updates (action processing)
/// - Market clearing
/// - Reward calculation
/// - State updates
/// 
/// By doing everything in Rust, we eliminate multiple Python-Rust boundary crossings
#[pyfunction]
fn environment_step_rust(
    // Current state
    bids: PyReadonlyArray1<f64>,
    quantities: PyReadonlyArray1<f64>,
    costs: PyReadonlyArray1<f64>,
    demand: f64,
    // Parameters
    lambda_bid_penalty: f64,
    unit_lol_penalty: f64,
    demand_scale: f64,
    cost_scale: f64,
    time_horizon: usize,
) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
    let bids = bids.as_slice()?;
    let quantities = quantities.as_slice()?;
    let costs = costs.as_slice()?;
    let n = bids.len();
    
    // Market clearing (inline to avoid function call overhead)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        bids[a].partial_cmp(&bids[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let bids_sorted: Vec<f64> = order.iter().map(|&i| bids[i]).collect();
    let q_sorted: Vec<f64> = order.iter().map(|&i| quantities[i]).collect();
    
    let mut cum_supply = vec![0.0; n];
    cum_supply[0] = q_sorted[0];
    for i in 1..n {
        cum_supply[i] = cum_supply[i - 1] + q_sorted[i];
    }
    
    let m = cum_supply.iter().position(|&cs| cs >= demand).unwrap_or(n);
    
    let mut q_cleared = vec![0.0; n];
    let price: f64;
    
    if m >= n {
        q_cleared.copy_from_slice(&q_sorted);
        price = bids_sorted[n - 1];
    } else {
        for i in 0..m {
            q_cleared[i] = q_sorted[i];
        }
        if m > 0 {
            q_cleared[m] = demand - cum_supply[m - 1];
        } else {
            q_cleared[m] = demand;
        }
        price = bids_sorted[m];
    }
    
    let mut q_cleared_final = vec![0.0; n];
    for (i, &idx) in order.iter().enumerate() {
        q_cleared_final[idx] = q_cleared[i];
    }
    
    // Reward calculation (inline)
    let mut rewards = vec![0.0; n];
    let total_cleared: f64 = q_cleared_final.iter().sum();
    let loss_of_load_penalty = unit_lol_penalty * (demand - total_cleared).max(0.0);
    
    for i in 0..n {
        let base_reward = (price - costs[i]) * q_cleared_final[i];
        let bid_penalty = lambda_bid_penalty * (bids[i] - costs[i]).powi(2);
        rewards[i] = base_reward - bid_penalty - loss_of_load_penalty;
        
        // Scale reward
        rewards[i] /= demand_scale * cost_scale * (time_horizon as f64).max(1.0);
        rewards[i] *= 20.0;
    }
    
    Ok((price, q_cleared_final, rewards))
}

/// Observer function - simple version with basic features
#[pyfunction]
fn simple_observer_rust(
    demand_t: f64,
    capacity_i: f64,
    cost_i: f64,
    t: usize,
    demand_scale: f64,
    capacity_scale: f64,
    cost_scale: f64,
) -> PyResult<Vec<f64>> {
    let hour_of_day = t % 24;
    let obs = vec![
        demand_t / demand_scale,
        capacity_i / capacity_scale,
        cost_i / cost_scale,
        (hour_of_day as f64) / 24.0,
    ];
    Ok(obs)
}

/// Observer function - version 3 with cyclic time encoding
#[pyfunction]
fn simple_observer_v3_rust(
    demand_t: f64,
    capacity_i: f64,
    cost_i: f64,
    t: usize,
    demand_scale: f64,
    capacity_scale: f64,
    cost_scale: f64,
) -> PyResult<Vec<f64>> {
    let hour_of_day = (t % 24) as f64;
    let hour_x = (2.0 * std::f64::consts::PI * hour_of_day / 24.0).sin();
    let hour_y = (2.0 * std::f64::consts::PI * hour_of_day / 24.0).cos();
    
    let day_of_week = ((t / 24) % 7) as f64;
    let day_x = (2.0 * std::f64::consts::PI * day_of_week / 7.0).sin();
    let day_y = (2.0 * std::f64::consts::PI * day_of_week / 7.0).cos();
    
    let obs = vec![
        demand_t / demand_scale,
        capacity_i / capacity_scale,
        cost_i / cost_scale,
        hour_x,
        hour_y,
        day_x,
        day_y,
    ];
    Ok(obs)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_env_step(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(environment_step_rust, m)?)?;
    m.add_function(wrap_pyfunction!(simple_observer_rust, m)?)?;
    m.add_function(wrap_pyfunction!(simple_observer_v3_rust, m)?)?;
    Ok(())
}
