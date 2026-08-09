//! Portable simultaneous addition and subtraction fallback.

use super::Limb;

/// Replace `sum` with `sum + difference` and `difference` with
/// `sum_original - difference_original`, returning carry and borrow.
///
/// # Safety
///
/// - Both pointers must be valid for reads and writes of `len` limbs.
/// - The two spans must not overlap.
pub unsafe fn add_sub_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut carry = false;
    let mut borrow = false;
    let mut index = 0_usize;
    while index < len {
        // SAFETY: the caller supplies two disjoint spans of `len` limbs, and
        // `index` is bounded by the loop condition.
        let (sum_limb, difference_limb) = unsafe { (*sum.add(index), *difference.add(index)) };
        let (partial_sum, overflow_a) = sum_limb.overflowing_add(difference_limb);
        let (final_sum, overflow_b) = partial_sum.overflowing_add(Limb::from(carry));
        let (partial_difference, underflow_a) = sum_limb.overflowing_sub(difference_limb);
        let (final_difference, underflow_b) =
            partial_difference.overflowing_sub(Limb::from(borrow));
        // SAFETY: both destinations are valid at `index` and do not overlap.
        unsafe {
            *sum.add(index) = final_sum;
            *difference.add(index) = final_difference;
        }
        carry = overflow_a | overflow_b;
        borrow = underflow_a | underflow_b;
        index = index.wrapping_add(1);
    }
    (Limb::from(carry), Limb::from(borrow))
}
