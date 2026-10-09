//! Portable implementation of `add_limbs_unchecked`.

use super::Limb;

/// Adds `src` to `dst` and returns the binary carry.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must be aligned, initialized, and cover `len` limbs in
/// live allocations of at most `isize::MAX` bytes. The destination must be
/// writable. The two spans must be disjoint or exactly identical.
#[expect(
    clippy::inline_always,
    reason = "Keep the carry recurrence in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 1 {
        // SAFETY: len == 1 bounds both aligned, initialized spans.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: len == 1 bounds the aligned, writable destination.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    let mut carry = false;
    for i in 0..len {
        // SAFETY: i < len bounds both initialized spans. The two inputs are
        // read before the write, including when the pointers are identical.
        let (sum, c) = unsafe { (*dst.add(i)).carrying_add(*src.add(i), carry) };
        // SAFETY: i < len bounds the aligned, writable destination.
        unsafe {
            *dst.add(i) = sum;
        }
        carry = c;
    }
    Limb::from(carry)
}
