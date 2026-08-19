//! Core arithmetic on `MpUint`.
//!
//! - [`operators`]: the `core::ops` implementations and their in-place forms.
//! - [`policies`]: the `checked` / `wrapping` / `saturating` / `overflowing` /
//!   `strict` / `try` families, each against the plain GMP operation, so the
//!   cell reads as the cost of the policy wrapper.
//! - [`helpers`]: `mul_add`, `square`, `pow`, `midpoint`, `abs_diff`.

mod helpers;
mod operators;
mod policies;
