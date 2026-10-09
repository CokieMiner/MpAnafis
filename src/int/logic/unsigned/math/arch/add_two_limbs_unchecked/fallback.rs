//! Portable fallback for two independent same-width additions.

use super::Limb;

/// Computes `A' + c_a * B^len = A + source_a` and `B' + c_b * B^len = B + source_b`.
///
/// Concurrently executes two independent addition chains of length `len`, returning `(c_a, c_b)`.
///
/// # Safety
///
/// - For nonzero `len`, every pointer must cover `len` aligned, initialized
///   limbs, with byte spans at most `isize::MAX`. Destinations must be writable.
/// - Destination spans `dst_a[0..len]` and `dst_b[0..len]` must not overlap.
/// - `dst_a[0..len]` must not overlap `src_b[0..len]`, and `dst_b[0..len]` must not overlap `src_a[0..len]`.
/// - In-place self-aliasing is permitted: `dst_a == src_a` and `dst_b == src_b`.
/// - Input spans `src_a[0..len]` and `src_b[0..len]` may alias or be disjoint.
pub unsafe fn add_two_limbs_unchecked(
    dst_a: *mut Limb,
    src_a: *const Limb,
    dst_b: *mut Limb,
    src_b: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    let mut carry_a = false;
    let mut carry_b = false;
    for index in 0..len {
        // SAFETY: index < len lies in each aligned, initialized span. The
        // source is disjoint from its destination or aliases it exactly.
        let (left_result, next_carry_a) =
            unsafe { (*dst_a.add(index)).carrying_add(*src_a.add(index), carry_a) };
        // SAFETY: the same span proof applies to the independent right pair.
        let (right_result, next_carry_b) =
            unsafe { (*dst_b.add(index)).carrying_add(*src_b.add(index), carry_b) };
        // SAFETY: both destinations are writable at index and disjoint. All
        // source values at this index were loaded before either store.
        unsafe {
            *dst_a.add(index) = left_result;
            *dst_b.add(index) = right_result;
        }
        carry_a = next_carry_a;
        carry_b = next_carry_b;
    }
    (Limb::from(carry_a), Limb::from(carry_b))
}
