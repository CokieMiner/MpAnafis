//! Bounded-precision arithmetic-policy benchmarks.
//!
//! GMP has no checked, wrapping, overflowing, saturating, or strict integer
//! type. Each `rug` cell therefore performs the GMP arithmetic and the same
//! policy transformation as its paired `mp` cell. The ordinary unlimited
//! GMP kernel costs remain available in [`super::operators`].
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
