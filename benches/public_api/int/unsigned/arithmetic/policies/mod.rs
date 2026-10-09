//! Bounded-precision arithmetic-policy benchmarks.
//!
//! Rug computes the exact result and applies the paired bounded policy.
//!
//! Every operation has a bounded successful-path ladder. A representative
//! 1024-bit edge cell separately forces addition or multiplication overflow,
//! subtraction underflow, or a zero divisor. This isolates the policy branches
//! without multiplying the full benchmark runtime by another ladder.
//! Arguments name the precision: successful addition uses `(bits - 4)`-bit
//! values, successful multiplication uses half-width factors, and division or
//! remainder uses a full-width dividend with a half-width divisor. Those shapes
//! prove the result fits while retaining non-trivial limb work.

mod cases;
mod checked;
mod overflowing;
mod saturating;
mod strict;
mod try_ops;
mod wrapping;

pub use cases::{EDGE_WIDTH, Operation, Scenario, mp_pairs};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use cases::{
    rug_max, rug_pairs, rug_width, verify_option, verify_overflowing, verify_result, verify_value,
};
