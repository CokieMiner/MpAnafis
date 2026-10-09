//! Portable simultaneous addition and subtraction fallback.

use super::Limb;

/// Computes `S' + c * B^len = S + D` and `D' - b * B^len = S - D`.
///
/// Simultaneously computes addition and subtraction on two disjoint limb spans,
/// returning `(carry, borrow)` where `carry, borrow in {0, 1}`.
///
/// # Safety
///
/// - For nonzero `len`, both pointers must be aligned and valid for reads and
///   writes of `len` initialized limbs, with byte spans at most `isize::MAX`.
/// - Spans `sum[0..len]` and `difference[0..len]` must not overlap.
pub unsafe fn add_sub_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut carry = false;
    let mut borrow = false;
    for index in 0..len {
        // SAFETY: the caller supplies two disjoint spans of `len` limbs, and
        // `index` is bounded by the loop condition.
        let (sum_limb, difference_limb) = unsafe { (*sum.add(index), *difference.add(index)) };
        let (final_sum, next_carry) = sum_limb.carrying_add(difference_limb, carry);
        let (final_difference, next_borrow) = sum_limb.borrowing_sub(difference_limb, borrow);
        // SAFETY: both destinations are valid at `index` and do not overlap.
        unsafe {
            *sum.add(index) = final_sum;
            *difference.add(index) = final_difference;
        }
        carry = next_carry;
        borrow = next_borrow;
    }
    (Limb::from(carry), Limb::from(borrow))
}
