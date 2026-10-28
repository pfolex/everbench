//! High-performance Rust kernels for `everbench`:
//! - Stockfish Fishtest-style Sequential Probability Ratio Test (SPRT) log-likelihood bounds
//! - Elo difference & confidence interval estimation
//! - TIES (Trim, Elect Sign, Disjoint Merge) orthogonal parameter tensor merging

/// Compute SPRT lower and upper Log-Likelihood Ratio (LLR) thresholds `(lower_b, upper_a)`
/// for Type-I error rate `alpha` and Type-II error rate `beta`:
/// `B = ln(beta / (1 - alpha))`, `A = ln((1 - beta) / alpha)`
pub fn sprt_bounds(alpha: f64, beta: f64) -> (f64, f64) {
    let lower_b = (beta / (1.0 - alpha)).ln();
    let upper_a = ((1.0 - beta) / alpha).ln();
    (lower_b, upper_a)
}

/// Convert an Elo difference into expected score `E in (0, 1)` under the logistic model.
pub fn elo_to_score(elo_diff: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf(-elo_diff / 400.0))
}

/// Convert an empirical win/draw/loss score `s in (0, 1)` into an Elo difference.
pub fn score_to_elo(score: f64) -> f64 {
    let clamped = score.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / clamped - 1.0).log10()
}

/// Perform TIES (Trim, Elect Sign, Disjoint Merge) across multiple flattened parameter delta vectors.
pub fn ties_merge_deltas(deltas: &[Vec<f64>], density: f64) -> Vec<f64> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let dim = deltas[0].len();
    let keep_k = ((dim as f64) * density.clamp(0.0, 1.0)).ceil() as usize;

    // Step 1: Trim each contributor delta to its top-k magnitude parameters
    let trimmed: Vec<Vec<f64>> = deltas
        .iter()
        .map(|vec| {
            let mut mags: Vec<(usize, f64)> =
                vec.iter().copied().enumerate().map(|(i, v)| (i, v.abs())).collect();
            mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut out = vec![0.0; dim];
            for (idx, _) in mags.into_iter().take(keep_k) {
                out[idx] = vec[idx];
            }
            out
        })
        .collect();

    // Step 2 & 3: Elect aggregate sign by total magnitude mass and disjoint-mean aligned entries
    let mut merged = vec![0.0; dim];
    for d in 0..dim {
        let mass_sum: f64 = trimmed.iter().map(|t| t[d]).sum();
        if mass_sum == 0.0 {
            continue;
        }
        let elected_sign = mass_sum.signum();
        let aligned: Vec<f64> = trimmed
            .iter()
            .map(|t| t[d])
            .filter(|&v| v != 0.0 && v.signum() == elected_sign)
            .collect();
        if !aligned.is_empty() {
            merged[d] = aligned.iter().sum::<f64>() / (aligned.len() as f64);
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprt_and_ties() {
        let (b, a) = sprt_bounds(0.05, 0.05);
        assert!(b < -2.9 && a > 2.9);
        let d1 = vec![0.8, -0.01, 0.5, 0.0];
        let d2 = vec![0.6, 0.9, -0.02, 0.0];
        let merged = ties_merge_deltas(&[d1, d2], 0.5);
        assert!((merged[0] - 0.7).abs() < 1e-9);
    }
}
