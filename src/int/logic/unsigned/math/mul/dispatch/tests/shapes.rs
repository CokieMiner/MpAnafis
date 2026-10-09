//! Operand grids spanning crossovers and split ratios.

use alloc::{vec, vec::Vec};

use crate::int::logic::unsigned::math::{
    BALANCED_TOOM8_THRESHOLD, KARATSUBA_THRESHOLD, SQR_KARATSUBA_THRESHOLD,
    SQR_TOOM_COOK_4_THRESHOLD, SQR_TOOM_COOK_6_THRESHOLD, SQR_TOOM_COOK_85_THRESHOLD,
    SQR_TOOM_COOK_THRESHOLD, TOOM_COOK_4_THRESHOLD, TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD,
    TOOM_COOK_THRESHOLD,
};
#[cfg(not(target_pointer_width = "16"))]
use crate::int::logic::unsigned::math::{SQR_SSA_THRESHOLD, SSA_THRESHOLD};

pub fn widths() -> Vec<usize> {
    let mut result = vec![0, 1, 2, 3, 5, 7, 8, 17, 64, 100, 255, 700, 3_000, 6_000];
    let thresholds = [
        KARATSUBA_THRESHOLD,
        TOOM_COOK_THRESHOLD,
        TOOM_COOK_4_THRESHOLD,
        TOOM_COOK_6_THRESHOLD,
        TOOM_COOK_85_THRESHOLD,
        BALANCED_TOOM8_THRESHOLD,
        SQR_KARATSUBA_THRESHOLD,
        SQR_TOOM_COOK_THRESHOLD,
        SQR_TOOM_COOK_4_THRESHOLD,
        SQR_TOOM_COOK_6_THRESHOLD,
        SQR_TOOM_COOK_85_THRESHOLD,
        #[cfg(not(target_pointer_width = "16"))]
        SSA_THRESHOLD,
        #[cfg(not(target_pointer_width = "16"))]
        SQR_SSA_THRESHOLD,
    ];
    for threshold in thresholds {
        for delta in 0..3 {
            result.push(threshold.saturating_sub(delta));
            result.push(threshold.saturating_add(delta));
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

pub fn products() -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for larger in widths() {
        for (numerator, denominator) in [
            (1, 1),
            (17, 18),
            (4, 5),
            (3, 4),
            (2, 3),
            (1, 2),
            (1, 4),
            (1, 16),
        ] {
            let smaller = larger
                .checked_mul(numerator)
                .expect("test ratio fits")
                .div_euclid(denominator);
            result.push((larger, smaller));
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}
