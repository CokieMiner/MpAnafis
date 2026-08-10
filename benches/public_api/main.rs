//! Public API benchmarks comparing `arbi-anafis` against Rug/GMP across the
//! public method surface documented in `docs/int/api-inventory.md`.
//!
//! # Layout
//!
//! One directory per numeric domain, so the rational and floating point types
//! can be added beside the integers without disturbing them:
//!
//! ```text
//! public_api/
//!   int/        ArbiUint and ArbiInt
//!     unsigned/ ArbiUint
//!     signed/   ArbiInt
//! ```
//!
//! # Pairing rule
//!
//! Every benchmarked method is its own module holding an `arbi` function and,
//! wherever GMP expresses the same operation, a `rug` counterpart measured on
//! identical operands. Divan then prints the pair adjacently:
//!
//! ```text
//! int::unsigned::arithmetic::mul
//! ├─ arbi   ...
//! ╰─ rug    ...
//! ```
//!
//! so every cell is a ratio without any cross-referencing. A module with no
//! `rug` function states in its documentation why GMP has no counterpart.
//!
//! Rug is a `gmp-mpfr-sys` dependency available on x86-64 Linux only; on every
//! other target the `rug` halves compile out and the `arbi` halves still run.

mod int;

fn main() {
    divan::main();
}
