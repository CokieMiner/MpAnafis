//! Portable fallback for two independent same-width additions.

use super::Limb;

/// Add two independent source spans into two destination spans.
///
/// # Safety
///
/// - Every pointer must cover `len` readable limbs.
/// - Both destination pointers must cover `len` writable limbs.
/// - No destination span may overlap any other span.
pub unsafe fn add_two_limbs_unchecked(
    dst_a: *mut Limb,
    src_a: *const Limb,
    dst_b: *mut Limb,
    src_b: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut carry_a = false;
    let mut carry_b = false;
    let mut index = 0;
    while index < len {
        // SAFETY: the caller provides four disjoint spans of `len` limbs and
        // the loop condition bounds index within every span.
        let (left_sum, left_overflow) =
            unsafe { (*dst_a.add(index)).overflowing_add(*src_a.add(index)) };
        let (left_result, left_carry_overflow) = left_sum.overflowing_add(Limb::from(carry_a));
        // SAFETY: the same span proof applies to the independent right pair.
        let (right_sum, right_overflow) =
            unsafe { (*dst_b.add(index)).overflowing_add(*src_b.add(index)) };
        let (right_result, right_carry_overflow) = right_sum.overflowing_add(Limb::from(carry_b));
        // SAFETY: both destinations are valid at index and all spans are disjoint.
        unsafe {
            *dst_a.add(index) = left_result;
            *dst_b.add(index) = right_result;
        }
        carry_a = left_overflow | left_carry_overflow;
        carry_b = right_overflow | right_carry_overflow;
        index = index.wrapping_add(1);
    }
    (Limb::from(carry_a), Limb::from(carry_b))
}
