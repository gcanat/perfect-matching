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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csa_3x3() {
        let cost = vec![9_f32, 2., 7., 3., 6., 1., 5., 8., 4.];
        assert_eq!(csa_scalar(&cost, 3, None), vec![1, 2, 0]);
    }

    #[test]
    fn test_csa_4x4() {
        let cost = vec![
            10_f32, 5., 13., 8., 4., 12., 7., 3., 9., 2., 11., 6., 6., 8., 4., 10.,
        ];
        assert_eq!(csa_scalar(&cost, 4, None), vec![3, 0, 1, 2]);
    }
}
