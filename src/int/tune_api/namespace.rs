//! Internal tuning namespace and runner constructors.

use super::{
    DivisionRunner, FormattingAlgorithm, FormattingRunner, Limb, MultiplicationAlgorithm,
    MultiplicationRunner, SquaringAlgorithm, SquaringRunner,
};

/// Namespace for the feature-gated tuning and benchmark facade.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct Tuner;

impl Tuner {
    /// Creates reusable state for one multiplication tier sample.
    #[must_use]
    pub fn multiplication(
        algorithm: MultiplicationAlgorithm,
        len_a: usize,
        len_b: usize,
    ) -> MultiplicationRunner {
        MultiplicationRunner::new(algorithm, len_a, len_b)
    }

    /// Creates reusable state for one squaring tier sample.
    #[must_use]
    pub fn squaring(algorithm: SquaringAlgorithm, len: usize) -> SquaringRunner {
        SquaringRunner::new(algorithm, len)
    }

    /// Creates reusable state for one division algorithm sample.
    #[must_use]
    pub fn division(num: &[Limb], den: &[Limb]) -> DivisionRunner {
        DivisionRunner::new(num, den)
    }

    /// Creates reusable state for one formatting algorithm sample.
    #[must_use]
    pub fn formatting(algorithm: FormattingAlgorithm, len: usize, radix: u32) -> FormattingRunner {
        FormattingRunner::new(algorithm, len, radix)
    }
}
