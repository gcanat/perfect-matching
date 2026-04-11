# perfect-matching

`perfect-matching` is a library for solving the Linear Sum Assignment Problem.

It provides a fast implementation of the Jonker-Volgenant algorithm.
Two versions are available: 
- `lsap_scalar`: straight forward implementation
- `lsap_simd`: SIMD accelerated implementation

Other algorithms available:
- Cost Scaling Auction algorithm: similar to lsap, scalar and simd versions are available. Very fast for dense square cost matrices.
- Kuhn-Munkres aglorithm (aka Hungarian Matching): classical algorithm, works with integer cost matrices. Slower than other algorithms.

More algorithms may be implemented in the future. Maybe some specialized algorithms for sparse cost matrices.

## Example usage
```rust
use perfect_matching::sapjv::{lsap_scalar, lsap_simd};
use perfect_matching::csa::{csa_scalar, csa_simd};
use perfect_matching::hun::hungarian_matching;

// a row-major representation of the cost matrix
let costs = vec![8_f32, 5., 9., 4., 2., 4., 7., 3., 8.];
let expected_result = vec![0, 2, 1];

// args: cost_matrix, nrows, ncols
let assignments1 = lsap_scalar(&costs, 3, 3);
let assignments2 = lsap_simd(&costs, 3, 3);
assert_eq!(assignments1, expected_result);
assert_eq!(assignments2, expected_result);

// CSA algorithm with both scalar and SIMD implementations
// args: cost_matrix, nrows, ncols, optional start epsilon
let assignments3 = csa_scalar(&costs, 3, 3, None);
let assignments4 = csa_simd(&costs, 3, 3, None);
assert_eq!(assignments3, expected_result);
assert_eq!(assignments4, expected_result);

// Kuhn-Munkres algorithm
let costs = vec![8_i32, 5, 9, 4, 2, 4, 7, 3, 8];
let assignments5 = hungarian_matching(&costs, 3, 3);
assert_eq!(assignments5, expected_result);
```
