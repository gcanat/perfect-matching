//! Goldberg-Kennedy Cost Scaling Auction Algorithm
//! ================================================
//! Solves the Linear Sum Assignment Problem (LSAP):
//!   Minimize  `sum_ij cost[i][j] * x[i][j]`.
//!
//!   Constraints:
//!   - each row `i` assigned to exactly one column `j`
//!   - each column `j` assigned to exactly one row `i`
//!   - `x[i][j]` in `{0, 1}`
//!
//! Core idea
//! ---------
//! Prices `p[j]` are maintained for each column (object/item).
//! Each row (bidder) computes its "best value" and bids up the price of its
//! favourite column. Epsilon-complementary slackness (Epsilon-CS) is
//! maintained throughout: for every assignment (`i -> j`),
//!  `cost[i][j] - p[j]  >=  max_k (cost[i][k] - p[k]) - epsilon`
//!
//! A full auction round (phase) works at a fixed epsilon.
//! epsilon is divided by `ALPHA` each phase until `epsilon < min(1/n^2, 1e-6)`

use bytemuck::cast_slice;
use pulp::{Arch, Simd, WithSimd};

// Scaling factor for epsilon
const ALPHA: f32 = 7.0;

// Minimum epsilon threshold
const MIN_EPS_THRESH: f32 = 1e-7;

/// Helper function to find the min and max value in the profit vec in a single
/// pass.
fn find_max(profit: &[f32]) -> f32 {
    let mut max = f32::MIN;
    for value in profit.iter() {
        if value > &max {
            max = *value;
        }
    }
    max
}

/// Struct holding the variables needed for the Cost Scaling Auction algorithm.
struct Auctioner {
    profit: Vec<f32>,
    prices: Vec<f32>,
    row4col: Vec<usize>,
    col4row: Vec<usize>,
    unassigned: Vec<usize>,
    n: usize,
    epsilon: f32,
}

impl Auctioner {
    /// Create a new `Auctioner` given a `cost` vector, the number of rows `n` and
    /// an optional `epsilon` start value.
    fn new(cost: &[f32], n: usize, epsilon: Option<f32>) -> Self {
        let profit: Vec<f32> = cost.iter().map(|v| -v).collect();
        let prices: Vec<f32> = vec![0_f32; n];
        let row4col: Vec<usize> = vec![usize::MAX; n];
        let col4row: Vec<usize> = vec![usize::MAX; n];
        let unassigned: Vec<usize> = Vec::from_iter(0..n);
        let epsilon = match epsilon {
            Some(eps) => eps,
            None => {
                let max_val = find_max(&profit);
                (max_val.abs()).max(1.0)
            }
        };
        Auctioner {
            profit,
            prices,
            row4col,
            col4row,
            unassigned,
            n,
            epsilon,
        }
    }

    /// For a given `row`, return the first and second highest net profit.
    fn best_and_second(self: &Auctioner, row: usize) -> (usize, f32, f32) {
        let mut best_col = usize::MAX;
        let mut best_val = f32::MIN;
        let mut second_val = f32::MIN;
        for j in 0..self.n {
            let net = self.profit[row * self.n + j] - self.prices[j];
            if net > best_val {
                second_val = best_val;
                best_val = net;
                best_col = j;
            } else if net > second_val {
                second_val = net;
            }
        }
        (best_col, best_val, second_val)
    }

    /// SIMD version of `best_and_second`.
    fn best_and_second_simd(&self, row: usize, arch: Arch) -> (usize, f32, f32) {
        let result = arch.dispatch(BestAndSecond {
            profit_row: &self.profit[row * self.n..(row + 1) * self.n],
            prices: &self.prices,
        });
        (result.best_col, result.best_val, result.second_val)
    }

    /// One bidder iteration for `row`:
    /// 1. Find its best column j* and the runner-up value.
    /// 2. Raise price[j*] so that row is just indifferent between
    ///    j* and the second-best.
    /// 3. Assign row -> j*; evict the previously assigned row if any.
    fn bid_and_assign(self: &mut Auctioner, row: usize) -> usize {
        let (best_col, best_val, second_val) = self.best_and_second(row);
        let gamma: f32 = if second_val == f32::MIN {
            best_val + self.epsilon
        } else {
            (best_val - second_val) + self.epsilon
        };
        self.prices[best_col] += gamma;

        let prev_row = self.col4row[best_col];
        if prev_row != usize::MAX {
            self.row4col[prev_row] = usize::MAX;
        }
        self.row4col[row] = best_col;
        self.col4row[best_col] = row;
        prev_row
    }

    /// SIMD version of `bid_and_assign`.
    fn bid_and_assign_simd(&mut self, row: usize, arch: Arch) -> usize {
        let (best_col, best_val, second_val) = self.best_and_second_simd(row, arch);
        let gamma: f32 = if second_val == f32::NEG_INFINITY {
            best_val + self.epsilon
        } else {
            (best_val - second_val) + self.epsilon
        };
        self.prices[best_col] += gamma;

        let prev_row = self.col4row[best_col];
        if prev_row != usize::MAX {
            self.row4col[prev_row] = usize::MAX;
        }
        self.row4col[row] = best_col;
        self.col4row[best_col] = row;
        prev_row
    }

    /// Run auction at a fixed `epsilon` value.
    fn run_phase(self: &mut Auctioner) {
        // reset assignments
        self.row4col.fill(usize::MAX);
        self.col4row.fill(usize::MAX);
        self.unassigned.clear();
        self.unassigned.extend(0..self.n);
        let max_iter = self.n * 100;
        let mut iter = 0;
        while !self.unassigned.is_empty() && (iter < max_iter) {
            if let Some(row) = self.unassigned.pop() {
                let evicted = self.bid_and_assign(row);
                if evicted != usize::MAX {
                    self.unassigned.push(evicted);
                }
            }
            iter += 1;
        }
    }

    /// SIMD version of `run_phase`.
    fn run_phase_simd(&mut self, arch: Arch) {
        self.row4col.fill(usize::MAX);
        self.col4row.fill(usize::MAX);
        self.unassigned.clear();
        self.unassigned.extend(0..self.n);
        let max_iter = self.n * 100;
        let mut iter = 0;
        while !self.unassigned.is_empty() && (iter < max_iter) {
            if let Some(row) = self.unassigned.pop() {
                let evicted = self.bid_and_assign_simd(row, arch);
                if evicted != usize::MAX {
                    self.unassigned.push(evicted);
                }
            }
            iter += 1;
        }
    }

    /// Solve the Linear Sum Assignment problem by iteratively running auction phases
    /// until `epsilon` is small enough.
    fn solve(self: &mut Auctioner) {
        let nn = (self.n as f32).powi(2);
        let eps_thresh = (MIN_EPS_THRESH).max(1.0 / nn);
        while self.epsilon >= eps_thresh {
            self.run_phase();
            self.epsilon /= ALPHA;
        }
    }

    /// SIMD version of `solve`.
    fn solve_simd(&mut self, arch: Arch) {
        let nn = (self.n as f32).powi(2);
        let eps_thresh = (MIN_EPS_THRESH).max(1.0 / nn);
        while self.epsilon >= eps_thresh {
            self.run_phase_simd(arch);
            self.epsilon /= ALPHA;
        }
    }
}

/// SIMD kernel: finds the best and second-best `profit_row[j] - prices[j]`
/// across all columns, returning the argmax and both values.
struct BestAndSecond<'a> {
    profit_row: &'a [f32],
    prices: &'a [f32],
}

struct TopTwo {
    best_col: usize,
    best_val: f32,
    second_val: f32,
}

impl WithSimd for BestAndSecond<'_> {
    type Output = TopTwo;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> TopTwo {
        let Self { profit_row, prices } = self;
        let n = profit_row.len();

        // Per-lane running top-2 accumulators.
        let mut best_v = simd.splat_f32s(f32::NEG_INFINITY);
        let mut second_v = simd.splat_f32s(f32::NEG_INFINITY);
        let mut best_col_v = simd.splat_u32s(u32::MAX);

        let (profit_chunks, profit_tail) = S::as_simd_f32s(profit_row);
        let (prices_chunks, _) = S::as_simd_f32s(prices);

        let lanes = std::mem::size_of::<S::f32s>() / 4;

        // Precompute iota = [0, 1, ..., lanes-1] for per-chunk column indices.
        let iota: S::u32s = {
            let mut buf = simd.splat_u32s(0u32);
            let buf_slice: &mut [u32] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut buf));
            for (k, dst) in buf_slice.iter_mut().enumerate() {
                *dst = k as u32;
            }
            buf
        };

        for (chunk_idx, (p_chunk, v_chunk)) in
            profit_chunks.iter().zip(prices_chunks.iter()).enumerate()
        {
            let base = chunk_idx * lanes;
            let col_indices = simd.add_u32s(iota, simd.splat_u32s(base as u32));

            // net[j] = profit[j] - prices[j]
            let net = simd.sub_f32s(*p_chunk, *v_chunk);

            // update_best: lanes where net strictly exceeds current best
            let update_best = simd.less_than_f32s(best_v, net);

            // second = if update_best { old best } else { max(second, net) }
            let new_second = simd.max_f32s(second_v, net);
            second_v = simd.select_f32s_m32s(update_best, best_v, new_second);

            best_v = simd.max_f32s(best_v, net);
            best_col_v = simd.select_u32s_m32s(update_best, col_indices, best_col_v);
        }

        // Horizontal reduction: collapse per-lane vectors to scalar top-2.
        let best_vals: &[f32] = cast_slice(std::slice::from_ref(&best_v));
        let best_cols: &[u32] = cast_slice(std::slice::from_ref(&best_col_v));
        let second_vals: &[f32] = cast_slice(std::slice::from_ref(&second_v));

        let mut best_val = f32::NEG_INFINITY;
        let mut best_col = usize::MAX;
        let mut winner_lane = 0usize;
        for (k, (&bv, &bc)) in best_vals.iter().zip(best_cols.iter()).enumerate() {
            if bv > best_val {
                best_val = bv;
                best_col = bc as usize;
                winner_lane = k;
            }
        }

        // Global second = max over all per-lane second_vals, plus all per-lane
        // best_vals except the winner lane (those are genuine runners-up).
        let mut second_val = f32::NEG_INFINITY;
        for (k, (&bv, &sv)) in best_vals.iter().zip(second_vals.iter()).enumerate() {
            if sv > second_val {
                second_val = sv;
            }
            if k != winner_lane && bv > second_val {
                second_val = bv;
            }
        }

        // Scalar tail: columns that didn't fill a full SIMD vector.
        let tail_start = n - profit_tail.len();
        for j in tail_start..n {
            let net = profit_row[j] - prices[j];
            if net > best_val {
                second_val = best_val;
                best_val = net;
                best_col = j;
            } else if net > second_val {
                second_val = net;
            }
        }

        TopTwo {
            best_col,
            best_val,
            second_val,
        }
    }
}

/// Run the Cost Scaling Auction algorithm.
///
/// # Arguments
/// * `c` - a slice representing a cost matrix.
/// * `nrow` - the number of rows in the matrix.
///
/// # Returns
/// A Vec where the entry `i` is the worker's index assigned to this job.
///
/// # Examples
///
/// ```
/// use perfect_matching::csa::csa_scalar;
/// let cost = vec![9_f32, 2., 7., 3., 6., 1., 5., 8., 4.];
/// assert_eq!(csa_scalar(&cost, 3, None), vec![1, 2, 0]);
/// ```
pub fn csa_scalar(c: &[f32], nrow: usize, epsilon: Option<f32>) -> Vec<usize> {
    let mut auctioner = Auctioner::new(c, nrow, epsilon);
    auctioner.solve();
    auctioner.row4col
}

/// Run the Cost Scaling Auction algorithm using SIMD instructions.
///
/// Functionally identical to [`csa_scalar`]; uses vectorised f32 arithmetic
/// in the inner `best_and_second` scan to exploit available SIMD width
/// (AVX-512, AVX2, SSE4, NEON, …).
///
/// # Arguments
/// * `c` - a slice representing a cost matrix.
/// * `nrow` - the number of rows in the matrix.
///
/// # Returns
/// A Vec where the entry `i` is the worker's index assigned to this job.
///
/// # Examples
///
/// ```
/// use perfect_matching::csa::csa_simd;
/// let cost = vec![9_f32, 2., 7., 3., 6., 1., 5., 8., 4.];
/// assert_eq!(csa_simd(&cost, 3, None), vec![1, 2, 0]);
/// ```
pub fn csa_simd(c: &[f32], nrow: usize, epsilon: Option<f32>) -> Vec<usize> {
    let arch = Arch::new();
    let mut auctioner = Auctioner::new(c, nrow, epsilon);
    auctioner.solve_simd(arch);
    auctioner.row4col
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csa_3x3() {
        let cost = vec![9_f32, 2., 7., 3., 6., 1., 5., 8., 4.];
        assert_eq!(csa_scalar(&cost, 3, None), vec![1, 2, 0]);
        assert_eq!(csa_simd(&cost, 3, None), vec![1, 2, 0]);
    }

    #[test]
    fn test_csa_4x4() {
        let cost = vec![
            10_f32, 5., 13., 8., 4., 12., 7., 3., 9., 2., 11., 6., 6., 8., 4., 10.,
        ];
        assert_eq!(csa_scalar(&cost, 4, None), vec![3, 0, 1, 2]);
        assert_eq!(csa_simd(&cost, 4, None), vec![3, 0, 1, 2]);
    }
}
