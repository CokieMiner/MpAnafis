//! Exact odd-prime bitmap for small native cofactors.
//!
//! One bit represents each odd integer below 2^20. A 16-bit target covers
//! its entire limb domain instead, keeping the bitmap within its object-size
//! limit. The build script generates the Eratosthenes sieve as target-independent
//! bytes, avoiding both constant-evaluation loops and runtime initialization.

/// Bit `(n >> 1) & 7` of byte `n >> 4` is set exactly for odd nonprimes.
/// Its target-dependent extent is owned by the build-time bitmap generator.
pub static ODD_COMPOSITE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/prime_bitmap.bin"));
