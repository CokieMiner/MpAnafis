//! Reusable forced-tier formatting tuner.

use core::hint::black_box;

use super::{FormatCache, InternalArbiUint, Limb};

/// Root formatting tier measured by [`Tuner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Algorithm {
    /// Linear per-digit schoolbook extraction.
    Schoolbook,
    /// Barrett divide-and-conquer recursive formatting.
    Recursive,
}

#[cfg(target_pointer_width = "64")]
const FORMAT_HASH: usize = 0x9E37_79B9_7F4A_7C15;
#[cfg(target_pointer_width = "32")]
const FORMAT_HASH: usize = 0x9E37_79B9;
#[cfg(target_pointer_width = "16")]
const FORMAT_HASH: usize = 0x9E37;

/// Reusable fixed-input state for comparing formatting algorithms at a fixed limb width.
#[derive(Debug)]
pub struct Tuner {
    algorithm: Algorithm,
    value: InternalArbiUint,
    radix: u32,
    format_cache: FormatCache,
}

impl Tuner {
    /// Constructs a tuner for the given algorithm, operand width, and radix.
    ///
    /// The operand is filled with deterministic non-zero limbs so that the
    /// formatting work is representative of a dense value. For the recursive
    /// algorithm the shared [`FormatCache`] is warmed with one full formatting
    /// pass, so every timed run measures the same warmed domains and scratch
    /// buffers as the production recursive path.
    ///
    /// # Panics
    ///
    /// Panics if `len` is zero or `radix` is outside `3..=36` or is a power
    /// of two. Power-of-two radices use a separate bit-extraction path that
    /// does not participate in this crossover.
    #[must_use]
    pub fn new(algorithm: Algorithm, len: usize, radix: u32) -> Self {
        assert!(len != 0, "formatting tuner operand width must be nonzero");
        assert!(
            (3..=36).contains(&radix) && !radix.is_power_of_two(),
            "formatting tuner radix must be a non-power-of-two in 3..=36"
        );
        let limbs: Vec<Limb> = (0..len)
            .map(|index| index.wrapping_mul(FORMAT_HASH) | 1)
            .collect();
        let value = InternalArbiUint::from_limbs(limbs);
        let mut format_cache = FormatCache::new();
        if algorithm == Algorithm::Recursive {
            drop(black_box(value.to_string_radix_recursive_with_cache(
                radix,
                &mut format_cache,
            )));
        }
        Self {
            algorithm,
            value,
            radix,
            format_cache,
        }
    }

    /// Runs the configured formatting algorithm, consuming the output.
    pub fn run(&mut self) {
        let result = match self.algorithm {
            Algorithm::Schoolbook => self.value.to_string_radix_schoolbook(self.radix),
            Algorithm::Recursive => self
                .value
                .to_string_radix_recursive_with_cache(self.radix, &mut self.format_cache),
        };
        let _ = black_box(&result);
    }

    /// Returns the formatted output for verification.
    #[must_use]
    pub fn output(&mut self) -> String {
        match self.algorithm {
            Algorithm::Schoolbook => self.value.to_string_radix_schoolbook(self.radix),
            Algorithm::Recursive => self
                .value
                .to_string_radix_recursive_with_cache(self.radix, &mut self.format_cache),
        }
    }
}
