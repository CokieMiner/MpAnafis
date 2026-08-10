//! Division on `ArbiUint`.
//!
//! - [`quotient`]: `div_rem`, `div_trunc`, `rem_trunc` on equal-width operands.
//! - [`shapes`]: the same three methods on the operand shapes that actually
//!   reach the recursive divider.
//! - [`rounding`]: the Euclidean, floor and ceiling families.
//! - [`predicates`]: divisibility tests, which can answer without a full
//!   division.

mod predicates;
mod quotient;
mod rounding;
mod shapes;
