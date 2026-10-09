//! Radix parsing runners with deterministic input and a schoolbook reference.

use core::hint::black_box;

use alloc::{string::String, vec::Vec};

use super::{Convert, InternalMpUint, RadixParameters, TuningResult};

/// Parsing root selected independently of the multiplication dispatcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParsingAlgorithm {
    /// Accumulate the complete digit string using native-limb multiplications.
    Schoolbook,
    /// Reconstruct chunk blocks with the compiled schoolbook-leaf cutoff.
    Recursive,
    /// Use the compiled public parsing entry and leaf cutoffs.
    Production,
}

/// Prepared valid input and a schoolbook reference, both outside timing.
#[derive(Debug)]
pub struct ParsingRunner {
    algorithm: ParsingAlgorithm,
    input: String,
    radix: u32,
    expected: InternalMpUint,
}

impl ParsingRunner {
    /// Constructs a deterministic input containing exactly `chunks` full radix chunks.
    ///
    /// # Panics
    ///
    /// Panics if `chunks` is zero, `radix` is outside `3..=36` or is a power
    /// of two, or the input byte count exceeds the addressable string span.
    #[must_use]
    pub fn new(algorithm: ParsingAlgorithm, chunks: usize, radix: u32) -> Self {
        assert!(chunks != 0, "parsing tuner needs at least one chunk");
        assert!(
            (3..=36).contains(&radix) && !radix.is_power_of_two(),
            "parsing tuner needs a supported non-power-of-two radix"
        );
        let parameters = RadixParameters::for_limb(radix);
        let digits = chunks
            .checked_mul(parameters.max_digits)
            .expect("parsing input length fits usize");
        assert!(
            isize::try_from(digits).is_ok(),
            "parsing input is addressable"
        );
        let mut state = 42_u32;
        let bytes: Vec<u8> = (0..digits)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let digit = if index == 0 {
                    state
                        .rem_euclid(radix.checked_sub(1).expect("radix is at least three"))
                        .checked_add(1)
                        .expect("radix is at most 36")
                } else {
                    state.rem_euclid(radix)
                };
                Convert::byte_from_digit(u8::try_from(digit).expect("radix digits below 36 fit u8"))
            })
            .collect();
        let input = String::from_utf8(bytes).expect("the generator emits ASCII digits");
        let expected =
            InternalMpUint::parse_non_power_of_two(input.as_bytes(), radix, parameters, false, 1)
                .expect("generated digits are valid");
        Self {
            algorithm,
            input,
            radix,
            expected,
        }
    }

    /// Parses the prepared input and returns an owned result.
    ///
    /// Each call includes digit validation, output allocation, and any power
    /// precomputation and combination. The caller determines when to drop the result.
    #[expect(
        unsafe_code,
        reason = "Immutable validated digits make the parse result infallible; unchecked extraction omits a redundant timed error branch"
    )]
    #[must_use]
    pub fn run(&self) -> TuningResult {
        let input = black_box(&self.input);
        let radix = black_box(self.radix);
        let result = match self.algorithm {
            ParsingAlgorithm::Production => InternalMpUint::from_str_radix(input, radix),
            ParsingAlgorithm::Schoolbook | ParsingAlgorithm::Recursive => {
                let parameters = RadixParameters::for_limb(radix);
                let (_, leaf) = Convert::parsing_thresholds(radix);
                InternalMpUint::parse_non_power_of_two(
                    input.as_bytes(),
                    radix,
                    parameters,
                    self.algorithm == ParsingAlgorithm::Recursive,
                    leaf,
                )
            }
        };
        // SAFETY: new validates radix in 3..=36, generates nonempty ASCII
        // digits below radix, and parses the same input successfully with the
        // schoolbook path. Input and radix never mutate, so each selected
        // parser returns Ok for this representable integer.
        let value = unsafe { result.unwrap_unchecked() };
        TuningResult::new(value)
    }

    /// Checks the selected parser against the schoolbook reference.
    #[must_use]
    pub fn verify(&self) -> bool {
        self.run().as_ref() == self.expected.limbs()
    }

    /// Returns the prepared input string.
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }
}
