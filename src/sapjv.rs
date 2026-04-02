use bytemuck::cast_slice;
use pulp::{Arch, Simd, WithSimd};

/// Shortest Augmenting Path algorithm for the Linear Sum Assignment Problem.
///
/// Based on: Crouse, "On implementing 2D rectangular assignment algorithms",
/// https://ui.adsabs.harvard.edu/abs/2016ITAES..52.1679C/abstract
/// Given `J` jobs and `W` workers (`J <= W`), computes the minimum cost to assign each jobs
/// to distinct workers.
///
/// # Arguments
/// * `c` - A slice representing a `J x W` cost matrix where `c[j][w]` is the cost to
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
/// use perfect_matching::sapjv::lsap_scalar;
/// let costs = vec![8_f32, 5., 9., 4., 2., 4., 7., 3., 8.];
/// let assignments = lsap_scalar(&costs, 3, 3);
/// assert_eq!(assignments, vec![0, 2, 1]);
/// ```
pub fn lsap_scalar<T>(c: &[T], nrow: usize, ncol: usize) -> Vec<usize>
where
    T: Copy + PartialOrd + std::ops::Sub<Output = T> + std::ops::Add<Output = T>,
    T: num_traits::Bounded + num_traits::Zero,
{
    assert!(nrow <= ncol);

    let inf = T::max_value();

    let mut u = vec![T::zero(); nrow]; // row potentials
    let mut v = vec![T::zero(); ncol]; // col potentials
    let mut col4row = vec![usize::MAX; nrow]; // col assigned to each row
    let mut row4col = vec![usize::MAX; ncol]; // row assigned to each col

    for cur_row in 0..nrow {
        // Dijkstra-like shortest path from cur_row to any unassigned col
        let mut shortest_path_costs = vec![inf; ncol];
        let mut path = vec![usize::MAX; ncol];
        let mut visited = vec![false; ncol];

        let mut i = cur_row;
        let mut sink = usize::MAX;
        let mut min_val = T::zero();

        while sink == usize::MAX {
            let mut idx = usize::MAX;
            let mut lowest = inf;

            for j in 0..ncol {
                if !visited[j] {
                    let r = c[i * ncol + j] - u[i] - v[j] + min_val;
                    if r < shortest_path_costs[j] {
                        shortest_path_costs[j] = r;
                        path[j] = i;
                    }
                    if shortest_path_costs[j] < lowest
                        || (shortest_path_costs[j] == lowest && row4col[j] == usize::MAX)
                    {
                        lowest = shortest_path_costs[j];
                        idx = j;
                    }
                }
            }

            min_val = lowest;
            visited[idx] = true;

            if row4col[idx] == usize::MAX {
                sink = idx;
            } else {
                i = row4col[idx];
            }
        }

        // Update potentials along the path
        u[cur_row] = u[cur_row] + min_val;
        for j in 0..ncol {
            if visited[j] {
                let r = row4col[j];
                if r != usize::MAX {
                    u[r] = u[r] - shortest_path_costs[j] + min_val;
                }
                v[j] = v[j] + shortest_path_costs[j] - min_val;
            }
        }

        // Augment along the path back to cur_row
        let mut j = sink;
        loop {
            let i = path[j];
            row4col[j] = i;
            let prev_j = col4row[i];
            col4row[i] = j;
            if i == cur_row {
                break;
            }
            j = prev_j;
        }
    }

    col4row
}

/// SIMD implementation for LSAP
struct InnerScan<'a> {
    c_row: &'a [f32],
    v: &'a [f32],
    visited: &'a [u32],
    spc: &'a mut [f32],
    path: &'a mut [u32],
    row4col: &'a [u32],
    u_i: f32,
    min_val: f32,
    row_i: u32,
}

struct ScanResult {
    pub best_cost: f32,
    pub best_col: usize,
}

impl WithSimd for InnerScan<'_> {
    type Output = ScanResult;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> ScanResult {
        let Self {
            c_row,
            v,
            visited,
            spc,
            path,
            row4col,
            u_i,
            min_val,
            row_i,
        } = self;
        let ncol = c_row.len();

        let offset = simd.splat_f32s(min_val - u_i);
        let inf_v = simd.splat_f32s(f32::INFINITY);
        let row_v = simd.splat_u32s(row_i);
        let zero_u = simd.splat_u32s(0u32);

        let mut best_cost_v = simd.splat_f32s(f32::INFINITY);
        let mut best_col_v = simd.splat_u32s(u32::MAX);

        let (c_chunks, c_tail) = S::as_simd_f32s(c_row);
        let (v_chunks, _) = S::as_simd_f32s(v);
        let (spc_chunks, _) = S::as_mut_simd_f32s(spc);
        let (path_chunks, _) = S::as_mut_simd_u32s(path);
        let (vis_chunks, _) = S::as_simd_u32s(visited);

        let lanes = std::mem::size_of::<S::f32s>() / 4;

        // Precompute iota = [0, 1, 2, ..., lanes-1] once; per-chunk indices
        // are derived as iota + splat(base), replacing the scalar fill loop.
        let iota: S::u32s = {
            let mut buf = zero_u;
            let buf_slice: &mut [u32] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut buf));
            for (k, dst) in buf_slice.iter_mut().enumerate() {
                *dst = k as u32;
            }
            buf
        };

        for (chunk_idx, ((((c_v, v_v), spc_v), path_v), vis_v)) in c_chunks
            .iter()
            .zip(v_chunks.iter())
            .zip(spc_chunks.iter_mut())
            .zip(path_chunks.iter_mut())
            .zip(vis_chunks.iter())
            .enumerate()
        {
            let base = chunk_idx * lanes;

            // visited is already u32; cast directly to a SIMD vector and mask
            let is_visited = simd.greater_than_u32s(*vis_v, zero_u);

            // r = c[j] - v[j] + (min_val - u_i)
            let r = simd.add_f32s(simd.sub_f32s(*c_v, *v_v), offset);

            // spc[j] = min(spc[j], r)  for unvisited lanes only
            let old_spc_v = *spc_v;
            let new_spc = simd.min_f32s(*spc_v, r);
            *spc_v = simd.select_f32s_m32s(is_visited, *spc_v, new_spc);

            // path[j] = row_i  where r improved spc AND lane is unvisited
            let improved = simd.less_than_f32s(r, old_spc_v);
            let update_path = simd.and_m32s(simd.not_m32s(is_visited), improved);
            *path_v = simd.select_u32s_m32s(update_path, row_v, *path_v);

            // For argmin: mask visited lanes out with infinity
            let cost_for_min = simd.select_f32s_m32s(is_visited, inf_v, *spc_v);

            // Column indices for this chunk: iota + base
            let col_indices = simd.add_u32s(iota, simd.splat_u32s(base as u32));

            // Update running per-lane argmin
            let new_is_better = simd.less_than_f32s(cost_for_min, best_cost_v);
            best_cost_v = simd.select_f32s_m32s(new_is_better, cost_for_min, best_cost_v);
            best_col_v = simd.select_u32s_m32s(new_is_better, col_indices, best_col_v);
        }

        // Horizontal reduction: fold SIMD lanes down to a scalar argmin
        let costs: &[f32] = cast_slice(std::slice::from_ref(&best_cost_v));
        let cols: &[u32] = cast_slice(std::slice::from_ref(&best_col_v));
        let mut best_cost = f32::INFINITY;
        let mut best_col = usize::MAX;
        for (&cost, &col) in costs.iter().zip(cols.iter()) {
            if cost < best_cost
                || (cost == best_cost && col != u32::MAX && row4col[col as usize] == u32::MAX)
            {
                best_cost = cost;
                best_col = col as usize;
            }
        }

        // Scalar tail: remainder columns that don't fill a full SIMD vector
        let tail_start = ncol - c_tail.len();
        for j in tail_start..ncol {
            if visited[j] == 0u32 {
                let r = c_row[j] - u_i - v[j] + min_val;
                if r < spc[j] {
                    spc[j] = r;
                    path[j] = row_i;
                }
                if spc[j] < best_cost || (spc[j] == best_cost && row4col[j] == u32::MAX) {
                    best_cost = spc[j];
                    best_col = j;
                }
            }
        }

        ScanResult {
            best_cost,
            best_col,
        }
    }
}

/// Shortest Augmenting Path algorithm with SIMD instructions for the Linear Sum Assignment Problem.
///
/// Based on: Crouse, "On implementing 2D rectangular assignment algorithms",
/// https://ui.adsabs.harvard.edu/abs/2016ITAES..52.1679C/abstract
/// Given `J` jobs and `W` workers (`J <= W`), computes the minimum cost to assign each jobs
/// to distinct workers.
///
/// # Arguments
/// * `c` - A slice representing a `J x W` cost matrix where `c[j][w]` is the cost to
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
/// use perfect_matching::sapjv::lsap_simd;
/// let costs = vec![8_f32, 5., 9., 4., 2., 4., 7., 3., 8.];
/// let assignments = lsap_simd(&costs, 3, 3);
/// assert_eq!(assignments, vec![0, 2, 1]);
/// ```
pub fn lsap_simd(c: &[f32], nrow: usize, ncol: usize) -> Vec<usize> {
    assert!(nrow <= ncol);

    let arch = Arch::new();

    let mut u = vec![0f32; nrow];
    let mut v = vec![0f32; ncol];
    let mut col4row = vec![usize::MAX; nrow];
    let mut row4col = vec![u32::MAX; ncol];
    let mut visited = vec![0u32; ncol];
    let mut spc = vec![f32::INFINITY; ncol];
    let mut path = vec![u32::MAX; ncol];

    for cur_row in 0..nrow {
        spc.fill(f32::INFINITY);
        path.fill(u32::MAX);
        visited.fill(0);

        let mut i = cur_row;
        let mut sink = usize::MAX;
        let mut min_val = 0f32;

        while sink == usize::MAX {
            let res = arch.dispatch(InnerScan {
                c_row: &c[i * ncol..(i + 1) * ncol],
                v: &v,
                visited: &visited,
                spc: &mut spc,
                path: &mut path,
                row4col: &row4col,
                u_i: u[i],
                min_val,
                row_i: i as u32,
            });

            min_val = res.best_cost;
            let j = res.best_col;

            // Tie-breaking: if a free column exists at the same cost, prefer it as immediate sink
            // This keeps complexity O(N²) on near-uniform matrices, but it slows down on dense
            // matrices.
            // if row4col[j] != u32::MAX {
            //     for jj in 0..ncol {
            //         if visited[jj] == 0 && row4col[jj] == u32::MAX && spc[jj] == min_val {
            //             j = jj;
            //             break;
            //         }
            //     }
            // }

            visited[j] = 1;

            if row4col[j] == u32::MAX {
                sink = j;
            } else {
                i = row4col[j] as usize;
            }
        }

        u[cur_row] += min_val;
        for j in 0..ncol {
            if visited[j] != 0 {
                let r = row4col[j];
                if r != u32::MAX {
                    u[r as usize] += min_val - spc[j];
                }
                v[j] += spc[j] - min_val;
            }
        }

        let mut j = sink;
        loop {
            let pi = path[j] as usize;
            row4col[j] = pi as u32;
            let prev_j = col4row[pi];
            col4row[pi] = j;
            if pi == cur_row {
                break;
            }
            j = prev_j;
        }
    }

    col4row
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_costs() {
        let cost_matrix = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let asgmt = lsap_scalar(&cost_matrix, 3, 3);
        let asgmt_simd = lsap_simd(&cost_matrix, 3, 3);
        // all costs are equal: matches are made starting from the end
        assert_eq!(asgmt, vec![2, 1, 0]);
        assert_eq!(asgmt_simd, vec![2, 1, 0]);
    }

    #[test]
    fn test_known_assignment() {
        let costs_f32 = vec![8_f32, 5., 9., 4., 2., 4., 7., 3., 8.];
        assert_eq!(lsap_scalar(&costs_f32, 3, 3), vec![0, 2, 1]);
        assert_eq!(lsap_simd(&costs_f32, 3, 3), vec![0, 2, 1]);

        let costs_i64 = vec![8_i64, 5, 9, 4, 2, 4, 7, 3, 8];
        assert_eq!(lsap_scalar(&costs_i64, 3, 3), vec![0, 2, 1]);
    }

    #[test]
    fn test_single_element() {
        let costs = vec![42.0_f32];
        assert_eq!(lsap_scalar(&costs, 1, 1), vec![0]);
        assert_eq!(lsap_simd(&costs, 1, 1), vec![0]);
    }

    #[test]
    fn test_rectangular() {
        let costs = vec![3.0_f32, 1., 2., 1., 3., 2.];
        assert_eq!(lsap_scalar(&costs, 2, 3), vec![1, 0]);
        assert_eq!(lsap_simd(&costs, 2, 3), vec![1, 0]);
    }

    #[test]
    fn test_optimal_in_diagonal() {
        let costs = vec![0.0_f32, 1., 1., 1., 0., 1., 1., 1., 0.];
        assert_eq!(lsap_scalar(&costs, 3, 3), vec![0, 1, 2]);
        assert_eq!(lsap_simd(&costs, 3, 3), vec![0, 1, 2]);
    }
}
