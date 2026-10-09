//! Portable implementation of `add_limbs_3_unchecked`.

use super::Limb;

/// Writes the low `len` limbs of `src1 + src2` and returns the binary carry.
///
/// The incoming carry is zero. Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// For nonempty spans, all pointers must be aligned and cover `len` limbs.
/// Each span must lie within one live allocation and have at most `isize::MAX` bytes.
/// The sources must be initialized and readable; the destination must be
/// writable and disjoint from both sources. Its previous contents are unused.
/// The two source spans may overlap each other.
#[expect(
    clippy::inline_always,
    reason = "Keep the carry recurrence in the selected arithmetic caller"
)]
#[inline(always)]
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: len == 1 and the caller supplies aligned, initialized sources.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: len == 1 and dst covers one aligned, writable limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    let mut carry = false;
    for i in 0..len {
        // SAFETY: i < len bounds both aligned, initialized source spans.
        let (sum, c) = unsafe { (*src1.add(i)).carrying_add(*src2.add(i), carry) };
        // SAFETY: i < len bounds the aligned, writable destination span.
        unsafe {
            *dst.add(i) = sum;
        }
        carry = c;
    }
    Limb::from(carry)
}
