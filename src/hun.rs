//! Kuhn-Munkres algorithm a.k.a Hungarian Matching.
//!
//! Adapted from the C++ implementation at https://en.wikipedia.org/wiki/Hungarian_algorithm

/// Checks if b < a. Sets a = min(a, b). Returns true if b < a.
#[inline]
fn ckmin<T: PartialOrd>(a: &mut T, b: T) -> bool {
    if b < *a {
        *a = b;
        true
    } else {
        false
    }
}

/// Performs the Hungarian matching algorithm.
///
/// Given `J` jobs and `W` workers (`J <= W`), computes the minimum cost to assign each jobs
/// to distinct workers.
///
/// # Arguments
/// * `weights` - A slice representing a `J x W` cost matrix where `c[j][w]` is the cost to
///   assign job `j` to worker `w`. The slice is a row-major representation of the cost matrix.
///
/// # Returns
/// A `Vec<T>` of length `J`, where entry `j` is the worker's index assigned to this job.
///
/// # Panics
/// Panics if `weights` is empty, rows have inconsistent lengths, or `J > W`.
///
/// # Examples
///
/// ```
/// use perfect_matching::hun::hungarian_matching;
/// let costs = vec![8_i64, 5, 9, 4, 2, 4, 7, 3, 8];
/// let assignments = hungarian_matching(&costs, 3, 3);
/// assert_eq!(assignments, vec![0, 2, 1]);
/// ```
pub fn hungarian_matching<T>(weights: &[T], n_rows: usize, n_cols: usize) -> Vec<usize>
where
    T: Copy + Ord + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T>,
    T: num_traits::Bounded + num_traits::Signed,
{
    assert!(
        n_rows <= n_cols,
        "Number of jobs must not exceed number of workers"
    );

    // job[w] = job assigned to w-th worker, or None if unassigned.
    // A virtual (W+1)-th worker slot is appended for convenience.
    let mut job: Vec<Option<usize>> = vec![None; n_cols + 1];
    let mut result = vec![0usize; n_rows];
    // job potentials
    let mut ys: Vec<T> = vec![T::default(); n_rows];
    // worker potentials: -yt[n_cols] will accumulate the sum of all deltas.
    let mut yt: Vec<T> = vec![T::default(); n_cols + 1];

    let inf = T::max_value();

    for j_cur in 0..n_rows {
        // Assign j_cur-th job by routing through the virtual worker slot W.
        let mut w_cur = n_cols;
        job[w_cur] = Some(j_cur);

        // min reduced cost over edges from the augmenting-path set Z to each worker
        let mut min_to: Vec<T> = vec![inf; n_cols + 1];
        // previous worker on the alternating path
        let mut prev: Vec<Option<usize>> = vec![None; n_cols + 1];
        // whether each worker is currently in the set Z
        let mut in_z: Vec<bool> = vec![false; n_cols + 1];

        // Augment: runs at most j_cur + 1 times
        while job[w_cur].is_some() {
            in_z[w_cur] = true;
            let j = job[w_cur].unwrap();
            let mut delta = inf;
            let mut w_next = 0usize;

            for w in 0..n_cols {
                if !in_z[w] {
                    let reduced = weights[j * n_cols + w] - ys[j] - yt[w];
                    if ckmin(&mut min_to[w], reduced) {
                        prev[w] = Some(w_cur);
                    }
                    if ckmin(&mut delta, min_to[w]) {
                        w_next = w;
                    }
                }
            }

            // Update potentials for all workers
            for w in 0..=n_cols {
                if in_z[w] {
                    if let Some(jw) = job[w] {
                        ys[jw] = ys[jw] + delta;
                    }
                    yt[w] = yt[w] - delta;
                } else {
                    min_to[w] = min_to[w] - delta;
                }
            }

            w_cur = w_next;
        }

        // Update job assignments along the alternating path back to W
        while w_cur != n_cols {
            let w_prev = prev[w_cur].unwrap();
            job[w_cur] = job[w_prev];
            result[job[w_cur].unwrap()] = w_cur;
            w_cur = w_prev;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hun_3x3_i64() {
        let cost = vec![9_i64, 2, 7, 3, 6, 1, 5, 8, 4];
        assert_eq!(hungarian_matching(&cost, 3, 3), vec![1, 2, 0]);
    }

    #[test]
    fn test_hun_3x3_i32() {
        let cost = vec![9_i32, 2, 7, 3, 6, 1, 5, 8, 4];
        assert_eq!(hungarian_matching(&cost, 3, 3), vec![1, 2, 0]);
    }

    #[test]
    fn test_hun_4x4_i16() {
        let cost = vec![10_i16, 5, 13, 8, 4, 12, 7, 3, 9, 2, 11, 6, 6, 8, 4, 10];
        assert_eq!(hungarian_matching(&cost, 4, 4), vec![3, 0, 1, 2]);
    }

    #[test]
    fn test_hun_4x4_i8() {
        let cost = vec![10_i8, 5, 13, 8, 4, 12, 7, 3, 9, 2, 11, 6, 6, 8, 4, 10];
        assert_eq!(hungarian_matching(&cost, 4, 4), vec![3, 0, 1, 2]);
    }
}
