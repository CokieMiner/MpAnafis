//! Radix formatting runners with retained operands and reusable recursive caches.

use core::hint::black_box;

use alloc::{string::String, vec::Vec};

use super::{FormatCache, InternalMpUint, Limb};

#[cfg(target_pointer_width = "64")]
const FORMAT_HASH: usize = 0x9E37_79B9_7F4A_7C15;
#[cfg(target_pointer_width = "32")]
const FORMAT_HASH: usize = 0x9E37_79B9;
#[cfg(target_pointer_width = "16")]
const FORMAT_HASH: usize = 0x9E37;

/// Root formatting tier measured by [`FormattingRunner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FormattingAlgorithm {
    /// Repeated division by native-limb radix powers.
    Schoolbook,
    /// Barrett divide-and-conquer recursive formatting.
    Recursive,
}

/// Retained operand and cache for formatting at a fixed limb width.
#[derive(Debug)]
pub struct FormattingRunner {
    algorithm: FormattingAlgorithm,
    value: InternalMpUint,
    radix: u32,
    format_cache: FormatCache,
}

impl FormattingRunner {
    /// Constructs a deterministic operand with `len` nonzero limbs.
    ///
    /// Recursive formatting initializes its cache with one complete pass.
    /// Later calls reuse the cache and allocate their output strings.
    ///
    /// # Panics
    ///
    /// Panics if `len` is zero or `radix` is outside `3..=36` or is a power
    /// of two. Power-of-two radices use a separate bit-extraction path that
    /// does not participate in this crossover.
    #[must_use]
    pub fn new(algorithm: FormattingAlgorithm, len: usize, radix: u32) -> Self {
        assert!(len != 0, "formatting tuner operand width must be nonzero");
        assert!(
            (3..=36).contains(&radix) && !radix.is_power_of_two(),
            "formatting tuner radix must be a non-power-of-two in 3..=36"
        );
        let limbs: Vec<Limb> = (0..len)
            .map(|index| index.wrapping_mul(FORMAT_HASH) | 1)
            .collect();
        let value = InternalMpUint::from_limbs(limbs);
        let mut format_cache = FormatCache::new();
        if algorithm == FormattingAlgorithm::Recursive {
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

    /// Formats the operand and drops the output string before returning.
    #[inline]
    pub fn run(&mut self) {
        let result = match self.algorithm {
            FormattingAlgorithm::Schoolbook => self.value.to_string_radix_schoolbook(self.radix),
            FormattingAlgorithm::Recursive => self
                .value
                .to_string_radix_recursive_with_cache(self.radix, &mut self.format_cache),
        };
        let _ = black_box(&result);
    }

    /// Returns the formatted output for verification.
    #[must_use]
    pub fn output(&mut self) -> String {
        match self.algorithm {
            FormattingAlgorithm::Schoolbook => self.value.to_string_radix_schoolbook(self.radix),
            FormattingAlgorithm::Recursive => self
                .value
                .to_string_radix_recursive_with_cache(self.radix, &mut self.format_cache),
        }
    }
}
