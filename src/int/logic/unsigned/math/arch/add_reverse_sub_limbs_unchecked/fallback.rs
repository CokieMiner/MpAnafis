//! Portable simultaneous addition and reverse-subtraction fallback.

use super::Limb;

/// Adds the original spans into `sum` and subtracts original `sum` from `difference`.
///
/// Returns the binary carry and borrow. Empty spans access no pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized, writable limbs in
/// disjoint live allocations of at most `isize::MAX` bytes.
pub unsafe fn add_reverse_sub_limbs_unchecked(
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
        let (final_difference, next_borrow) = difference_limb.borrowing_sub(sum_limb, borrow);
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
