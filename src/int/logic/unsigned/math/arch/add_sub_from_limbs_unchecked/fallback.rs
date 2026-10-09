//! Portable shared-source simultaneous addition and subtraction fallback.

use super::Limb;

/// Adds `source` into `sum` and writes `sum_old - source` to `difference`.
///
/// Returns the binary carry and borrow. Empty spans access no pointers.
///
/// # Safety
///
/// Nonempty spans must be aligned and cover `len` limbs in live allocations
/// of at most `isize::MAX` bytes. `sum` and `source` must be initialized;
/// `sum` and `difference` must be writable. `sum` must be disjoint from both
/// other spans. `difference` and `source` must be disjoint or exactly identical.
pub unsafe fn add_sub_from_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    source: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut carry = false;
    let mut borrow = false;
    for index in 0..len {
        // SAFETY: the caller provides valid spans and index is loop-bounded.
        let (sum_limb, source_limb) = unsafe { (*sum.add(index), *source.add(index)) };
        let (final_sum, next_carry) = sum_limb.carrying_add(source_limb, carry);
        let (final_difference, next_borrow) = sum_limb.borrowing_sub(source_limb, borrow);
        // SAFETY: both destinations are valid and each source limb was loaded
        // before either output store, permitting exact difference/source alias.
        unsafe {
            *sum.add(index) = final_sum;
            *difference.add(index) = final_difference;
        }
        carry = next_carry;
        borrow = next_borrow;
    }
    (Limb::from(carry), Limb::from(borrow))
}
