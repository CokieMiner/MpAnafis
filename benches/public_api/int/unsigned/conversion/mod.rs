//! Conversion and serialisation.
//!
//! - [`strings`]: radix formatting and parsing, including the power-of-two
//!   radices where digits are a pure bit slice.
//! - [`bytes`]: endian byte vector round trips.
//! - [`primitives`]: casts to the fixed-width integer and float types.

mod bytes;
mod primitives;
mod strings;
