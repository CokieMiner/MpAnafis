//! Bitwise operations on `MpUint`.
//!
//! - [`operators`]: `BitAnd`, `BitOr`, `BitXor`, `not_with_width`, `try_not`.
//! - [`shifts`]: `Shl`, `Shr` and the shift policy family.
//! - [`manipulation`]: whole-word rearrangement — `reverse_bits`,
//!   `rotate_left`, `rotate_right`, `swap_bytes`.
//! - [`inspection`]: population counts, run lengths and bit scans.
//! - [`single_bit`]: addressed reads and writes of one bit, plus `bit_range`.
//!
//! Several of these have no single GMP entry point. Where the composed GMP
//! expression is the one a caller would otherwise write by hand, it is used as
//! the counterpart and the module documentation says so; where composing it
//! would just restate our own implementation, the module has no `rug` half.

mod inspection;
mod manipulation;
mod operators;
mod shifts;
mod single_bit;
