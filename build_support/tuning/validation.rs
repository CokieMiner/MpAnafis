//! Shared finite-value and threshold-chain predicates.

/// A finite geometry value cannot be zero or either reserved sentinel.
pub const fn valid_finite(value: usize) -> bool {
    value != 0 && value < usize::MAX - 1
}

pub fn valid_threshold_chain(values: &[usize]) -> bool {
    if values
        .iter()
        .any(|&value| value == 0 || value == usize::MAX)
    {
        return false;
    }
    // Equal adjacent thresholds intentionally shadow the lower tier. The
    // MAX-1 sentinel is valid only as a nondecreasing tail.
    values.windows(2).all(|pair| {
        let [first, second] = pair else {
            return false;
        };
        first <= second
    })
}

pub const fn valid_optional_crossover(value: usize, predecessor: usize) -> bool {
    value == 0 || (value < usize::MAX - 1 && value > predecessor)
}
