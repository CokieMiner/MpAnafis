//! `ArbiUint` benchmarks, grouped by the sections of
//! `docs/int/api-inventory.md`.
//!
//! # What is not benchmarked
//!
//! Constant-time accessors that read a field and return: `capacity`,
//! `is_zero`, `is_one`, `swap`, `as_debug_verbose`, and the `Precision`
//! constructors. They have no measurable cost against a twenty nanosecond
//! timer and no GMP counterpart to compare against, so a cell for them would
//! report timer noise. `reserve` and `shrink_to_fit` are allocator calls, not
//! arithmetic; they are exercised indirectly by the assignment benchmarks that
//! reserve once and reuse the destination.

mod arithmetic;
mod bitwise;
mod comparison;
mod conversion;
mod division;
mod modular;
mod theory;
