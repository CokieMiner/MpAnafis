//! Exact forced-profile fixtures, independent of automatic prime selection.

use crate::{LIMB_BITS, NttPlan, PlannedProfile, plan::Planning, profile::NttProfile};

pub fn forced(left: usize, right: usize, profile: NttProfile) -> Option<NttPlan> {
    let width = usize::try_from(profile.chunk_bits).ok()?;
    if left == 0 || right == 0 || width == 0 {
        return None;
    }
    let digits_a = left.checked_mul(LIMB_BITS)?.div_ceil(width);
    let digits_b = right.checked_mul(LIMB_BITS)?.div_ceil(width);
    let conv_len = digits_a.checked_add(digits_b)?.checked_sub(1)?;
    let transform_len = conv_len.checked_next_power_of_two()?.max(4);
    if !profile.admits(transform_len, digits_a.min(digits_b)) {
        return None;
    }
    Planning::admit(
        left,
        right,
        PlannedProfile {
            profile,
            digits_a,
            digits_b,
            conv_len,
            transform_len,
            transform_log: transform_len.trailing_zeros(),
        },
    )
}
