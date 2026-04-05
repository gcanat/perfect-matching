#![warn(missing_docs)]
//! Perfect-matching is a library implementing several algorithm to solve the
//! Linear Sum Assignments problem.
//!
//! It currently implements the following algorithms:
//! - Jonker-Volgenant
//! - Cost Scaling Auction
//!
//! # Example
//! ```
//! use perfect_matching::sapjv::lsap_scalar;
//! let costs = vec![3.0_f32, 1., 2., 1., 3., 2.];
//! assert_eq!(lsap_scalar(&costs, 2, 3), vec![1, 0]);
//! ```
pub mod csa;
pub mod sapjv;
