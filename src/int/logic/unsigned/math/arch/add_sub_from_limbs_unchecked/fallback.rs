//! Portable shared-source simultaneous addition and subtraction fallback.

use super::Limb;

/// Replace `sum` with `sum_original + source` and write
/// `sum_original - source` to `difference`, returning carry and borrow.
///
/// # Safety
///
/// - All three pointers must cover `len` limbs.
/// - `difference` may equal `source` exactly, but otherwise must not overlap an input span.
/// - `sum` and `source` must not overlap; `sum` is also its own input span.
pub unsafe fn add_sub_from_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    source: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut carry = false;
    let mut borrow = false;
    let mut index = 0_usize;
    while index < len {
        // SAFETY: the caller provides valid spans and index is loop-bounded.
        let (sum_limb, source_limb) = unsafe { (*sum.add(index), *source.add(index)) };
        let (partial_sum, overflow_a) = sum_limb.overflowing_add(source_limb);
        let (final_sum, overflow_b) = partial_sum.overflowing_add(Limb::from(carry));
        let (partial_difference, underflow_a) = sum_limb.overflowing_sub(source_limb);
        let (final_difference, underflow_b) =
            partial_difference.overflowing_sub(Limb::from(borrow));
        // SAFETY: both destinations are valid and each source limb was loaded
        // before either output store, permitting exact difference/source alias.
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
