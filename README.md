# perfect-matching

`perfect-matching` is a library for solving the Linear Sum Assignment Problem.

It provides a fast implementation of the Jonker-Volgenant algorithm.
Two versions are available: 
- `lsap_scalar`: straight forward implementation
- `lsap_simd`: SIMD accelerated implementation


More algorithms may be implemented in the future. Maybe some specialized algorithms for sparse cost matrices.

## Example usage
```rust
use perfect_matching::sapjv::{lsap_scalar, lsap_simd};

// a row-major representation of the cost matrix
let costs = vec![8_f32, 5., 9., 4., 2., 4., 7., 3., 8.];
// args: cost_matrix, nrows, ncols
let assignments1 = lsap_scalar(&costs, 3, 3);
let assignments2 = lsap_simd(&costs, 3, 3);
assert_eq!(assignments1, vec![0, 2, 1]);
assert_eq!(assignments2, vec![0, 2, 1]);
```
