//! Exact double-limb product decomposition.

#![expect(
    clippy::as_conversions,
    reason = "Widening usize to DoubleLimb is exact on every pointer width"
)]

use proptest::prelude::*;

use crate::int::types::{DoubleLimb, Limb};

use super::super::ArchKernels;

proptest! {
    #[test]
    fn limb_product_reconstructs_the_exact_widened_product(left in any::<Limb>(), right in any::<Limb>()) {
        for (a, b) in [(left, right), (0, right), (1, right), (Limb::MAX, right), (Limb::MAX, Limb::MAX)] {
            let (low, high) = ArchKernels::mul_limb_lo_hi(a, b);
            let reconstructed = ((high as DoubleLimb) << Limb::BITS) | (low as DoubleLimb);
            prop_assert_eq!(reconstructed, (a as DoubleLimb).checked_mul(b as DoubleLimb).expect("full limb product fits"));
        }
    }
}
