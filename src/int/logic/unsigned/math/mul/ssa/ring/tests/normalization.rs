//! Canonical normalization, the negative-one flag, and excluded sentinels.

#![expect(
    unsafe_code,
    clippy::indexing_slicing,
    reason = "Fixed complete coefficient windows retain two sentinel limbs around arbitrary native guard digits"
)]

use alloc::vec;

use super::{LIMB_BITS, Limb, SsaRing};

#[test]
fn normalization_retains_its_negative_one_flag_for_arbitrary_guards() {
    for width in [1, 2, 3, 8] {
        let bits = width * LIMB_BITS;
        for guard in [0, 1, 2, Limb::MAX] {
            for negative_one in [false, true] {
                if guard == 0 && negative_one {
                    continue;
                }
                let mut coefficient = vec![37; width + 3];
                coefficient[1..=width].fill(0);
                coefficient[1] = if negative_one { guard - 1 } else { guard };
                coefficient[width + 1] = guard;
                // SAFETY: the complete initialized coefficient uses a positive
                // limb-aligned ring. Normalization admits any native guard;
                // both sentinels lie outside the borrowed window.
                let actual = unsafe { SsaRing::normalize(&mut coefficient[1..=width + 1], bits) };
                assert_eq!(actual, negative_one, "width={width}, guard={guard}");
                assert_eq!(&coefficient[1..=width], vec![0; width]);
                assert_eq!(coefficient[width + 1], Limb::from(negative_one));
                assert_eq!((coefficient[0], coefficient[width + 2]), (37, 37));
            }
        }
    }
}
