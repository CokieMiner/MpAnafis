//! Exact order-statistic bounds for paired timing ratios.
//!
//! A confirmation uses fresh slots after its pilot and scheduled looks at
//! 31, 63 and 127 observations. Under independent, stationary slots conditional
//! on the preceding search, the sign distribution bounds the population median
//! of the rounded timing ratios without a normality assumption. The pilot
//! selects a work budget; it contributes no confirmation observations.

use super::MIN_MARGIN_PPM;

/// Measured median and one-sided bounds, in parts per million.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RatioEstimate {
    pub median: u128,
    pub lower: u128,
    pub upper: u128,
}

/// Paired-ratio inference and confirmation work budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairedStatistics;

impl PairedStatistics {
    /// Allocate an error budget to one fresh comparison and its cells.
    ///
    /// For comparison j>=1 and c cells, each one-sided error is at most
    /// `2^-15 / (ceil_pow2(j)^2 * ceil_pow2(c))` per tail and look. The three
    /// scheduled looks and two tails consume at most six times that allocation.
    /// Since `c/ceil_pow2(c) <= 1` and
    /// `sum(j>=1, 1/ceil_pow2(j)^2) = 3/2`, their union bound for one session is
    /// at most `9/32768 < 0.001`. Cells need not be mutually independent.
    /// Fresh slots for each cell must be independent and stationary conditional
    /// on earlier search decisions; this calculation does not test that premise.
    /// Screening observations never establish an accepted comparison.
    pub fn confidence_bits(comparison: u64, cells: usize) -> u32 {
        assert!(
            comparison > 0 && cells > 0,
            "positive comparison and cell counts"
        );
        let comparison_bits = u64::BITS
            .checked_sub(
                comparison
                    .checked_sub(1)
                    .expect("positive ordinal")
                    .leading_zeros(),
            )
            .expect("leading zeros do not exceed the integer width");
        let cell_bits = usize::BITS
            .checked_sub(
                cells
                    .checked_sub(1)
                    .expect("positive cells")
                    .leading_zeros(),
            )
            .expect("leading zeros do not exceed the pointer width");
        15_u32
            .checked_add(
                comparison_bits
                    .checked_mul(2)
                    .expect("u64 exponent fits u32"),
            )
            .and_then(|bits| bits.checked_add(cell_bits))
            .expect("pointer-width confidence exponents fit u32")
    }

    /// Choose the first scheduled look from pilot dispersion and error budget.
    ///
    /// These are bounded measurement budgets, not hardware or mathematical
    /// crossovers. A broad pilot requests more work; an unresolved bound at the
    /// largest budget remains inconclusive. Subsequent looks only add fresh
    /// observations up to the predeclared 63 and 127 observation totals.
    pub fn confirmation_samples(pilot: &[u128], confidence_bits: u32) -> usize {
        assert!(!pilot.is_empty(), "confirmation requires a pilot");
        let minimum = pilot.iter().copied().min().expect("nonempty pilot");
        let maximum = pilot.iter().copied().max().expect("nonempty pilot");
        let spread = maximum.checked_sub(minimum).expect("ordered pilot extrema");
        let units = spread.div_ceil(u128::from(MIN_MARGIN_PPM));
        let requested = units
            .checked_mul(units)
            .and_then(|count| count.checked_mul(31))
            .unwrap_or(u128::MAX);
        for count in [31_usize, 63, 127] {
            if u128::try_from(count).expect("small slot count") >= requested
                && u32::try_from(count).expect("small slot count") >= confidence_bits
            {
                return count;
            }
        }
        127
    }

    /// Bound a median from a fixed, fresh sample of at most 127 paired ratios.
    ///
    /// If `X~Binomial(n,1/2)` and `P(X<=k)<=2^-bits`, order statistics k and
    /// n-1-k are one-sided bounds. Ties make the bounds conservative. Exact
    /// integer Pascal coefficients avoid floating-point tail approximations.
    /// Missing coverage returns `[0, u128::MAX]`, never an accepted point value.
    pub fn median_bounds(samples: &mut [u128], confidence_bits: u32) -> RatioEstimate {
        assert!(
            !samples.is_empty() && samples.len() <= 127 && !samples.len().is_multiple_of(2),
            "confirmation sample count must be odd and at most 127"
        );
        samples.sort_unstable();
        let median = *samples
            .get(samples.len().div_euclid(2))
            .expect("nonempty median");
        let mut estimate = RatioEstimate {
            median,
            lower: 0,
            upper: u128::MAX,
        };
        let count = u32::try_from(samples.len()).expect("at most 127 slots");
        if confidence_bits > count || samples.contains(&0) {
            return estimate;
        }
        let tail_limit = 1_u128
            << count
                .checked_sub(confidence_bits)
                .expect("bounded exponent");
        let mut coefficients =
            vec![0_u128; samples.len().checked_add(1).expect("bounded row width")];
        *coefficients.first_mut().expect("nonempty Pascal row") = 1;
        for row in 1..=samples.len() {
            for index in (1..=row).rev() {
                let previous = *coefficients
                    .get(index.checked_sub(1).expect("positive index"))
                    .expect("Pascal predecessor");
                let entry = coefficients.get_mut(index).expect("bounded Pascal index");
                *entry = entry.checked_add(previous).expect("row n<=127 fits u128");
            }
        }
        let mut tail = 0_u128;
        for (rank, &coefficient) in coefficients
            .iter()
            .take(samples.len().div_euclid(2))
            .enumerate()
        {
            tail = tail
                .checked_add(coefficient)
                .expect("binomial mass fits u128");
            if tail > tail_limit {
                break;
            }
            estimate.lower = *samples.get(rank).expect("lower order statistic");
            estimate.upper = *samples
                .get(
                    samples
                        .len()
                        .checked_sub(rank)
                        .and_then(|index| index.checked_sub(1))
                        .expect("upper rank"),
                )
                .expect("upper order statistic");
        }
        estimate
    }
}
