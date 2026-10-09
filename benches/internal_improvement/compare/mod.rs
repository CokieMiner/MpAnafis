//! Equal-width and rectangular production multiplication against GMP and FLINT.
//! Numeric buffers are prepared before timing. Dispatch and external-library
//! allocations remain timed. Case labels distinguish serial and parallel budgets.

mod flint;
mod production;
mod unbalanced;

pub use flint::{
    FlintLimb, FlintSize, FlintThreadBudget, assert_one_limb_width, flint_mpn_mul, flint_mpn_mul_n,
};
