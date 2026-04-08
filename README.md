# perfect-matching

`perfect-matching` is a library for solving the Linear Sum Assignment Problem.

It provides a fast implementation of the Jonker-Volgenant algorithm.
Two versions are available: 
- `lsap_scalar`: straight forward implementation
- `lsap_simd`: SIMD accelerated implementation

It also provides an implementation of the Cost Scaling Auction algorithm.
Similar to lsap, scalar and simd versions are available.

More algorithms may be implemented in the future. Maybe some specialized algorithms for sparse cost matrices.

## Example usage
```rust
use perfect_matching::sapjv::{lsap_scalar, lsap_simd};
use perfect_matching::csa::{csa_scalar, csa_simd};

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
```
