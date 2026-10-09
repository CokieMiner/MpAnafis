//! Dense, alternating, and endpoint limb patterns for carry-chain tests.

use alloc::{vec, vec::Vec};

use crate::int::types::Limb;

pub fn row_patterns(len: usize) -> Vec<Vec<Limb>> {
    let mut patterns = vec![vec![0; len], vec![Limb::MAX; len]];
    if !cfg!(miri) {
        patterns.push(
            (0..len)
                .map(|index| {
                    if index.is_multiple_of(2) {
                        Limb::MAX
                    } else {
                        0
                    }
                })
                .collect(),
        );
        patterns.push(
            (0..len)
                .map(|index| {
                    if index.is_multiple_of(2) {
                        0
                    } else {
                        Limb::MAX
                    }
                })
                .collect(),
        );
        patterns.push(vec![Limb::MAX - 7; len]);
    }
    if len != 0 {
        let mut low = vec![0; len];
        *low.first_mut().expect("nonempty pattern") = Limb::MAX;
        let mut high = vec![0; len];
        *high.last_mut().expect("nonempty pattern") = Limb::MAX;
        patterns.extend([low, high]);
    }
    patterns
}
