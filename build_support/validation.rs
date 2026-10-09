//! Finite-value and threshold-chain predicates for tuning validation.

/// Namespace for tuning input and dispatch-order contracts.
#[derive(Clone, Copy, Debug)]
pub struct Validation;

impl Validation {
    /// Returns whether an input is nonzero and below both reserved sentinels.
    #[must_use]
    pub const fn valid_finite(value: usize) -> bool {
        value != 0 && value < usize::MAX - 1
    }

    /// Returns whether crossover thresholds are nonzero, nondecreasing, and below `usize::MAX`.
    ///
    /// Adjacent equal thresholds shadow lower tiers. The sentinel `usize::MAX - 1`
    /// is permitted in a terminating tail of disabled tiers.
    #[must_use]
    pub fn valid_threshold_chain(values: &[usize]) -> bool {
        if values
            .iter()
            .any(|&value| value == 0 || value == usize::MAX)
        {
            return false;
        }
        values.windows(2).all(|pair| {
            let [first, second] = pair else {
                return false;
            };
            first <= second
        })
    }

    /// Returns whether an optional crossover is disabled or follows its predecessor.
    ///
    /// Enabled crossovers must remain below `usize::MAX - 1`.
    #[must_use]
    pub const fn valid_optional_crossover(value: usize, predecessor: usize) -> bool {
        value == 0 || (value < usize::MAX - 1 && value > predecessor)
    }

    /// Returns whether a transform crossover is disabled or follows the active ladder.
    ///
    /// If the top tier is disabled with `usize::MAX - 1`, its last active
    /// predecessor supplies the lower bound. The caller validates ladder order.
    #[must_use]
    pub fn valid_transform_crossover(
        value: usize,
        toom_cook_85: usize,
        conventional_chain: &[usize],
    ) -> bool {
        if value == 0 {
            return true;
        }
        if value >= usize::MAX - 1 {
            return false;
        }
        let predecessor = if toom_cook_85 < usize::MAX - 1 {
            toom_cook_85
        } else {
            conventional_chain
                .iter()
                .rev()
                .copied()
                .find(|&val| val != 0 && val < usize::MAX - 1)
                .unwrap_or(0)
        };
        value > predecessor
    }
}
