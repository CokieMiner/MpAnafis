//! Unbalanced operand geometry.
//!
//! Everything here varies the *ratio* between the two operands rather than
//! their size. The tower is tuned for balanced products and reaches unbalanced
//! ones by blocking or by a fractional-ratio Toom variant, and which of those
//! wins is a question of shape, not of width.

pub mod deep_lopsided;
pub mod fractional_ratio;
pub mod lopsided;
pub mod toom_windows;
