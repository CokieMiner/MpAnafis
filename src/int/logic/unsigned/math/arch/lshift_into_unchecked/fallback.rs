//! Pure Rust fallback for the out-of-place left-shift kernel.

use super::{LIMB_BITS, Limb};

/// Evaluates `dst <- (src << shift) mod B^len` out of place over `len` limbs.
///
/// Writes shifted limbs into `dst` and returns the bits shifted out of the
/// most significant limb: `src[len - 1] >> (LIMB_BITS - shift)`.
///
/// # Safety
///
/// - For nonzero `len`, both pointers must be aligned and cover `len` limbs,
///   with byte spans at most `isize::MAX`. Source limbs must be initialized;
///   destination limbs must be writable and may be uninitialized.
/// - Spans `dst[0..len]` and `src[0..len]` must not overlap.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Inlining keeps the limb loop at its caller; LIMB_BITS is 16, 32, or 64 and fits in u32"
)]
#[inline(always)]
pub unsafe fn lshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    // SAFETY: 0 < shift < LIMB_BITS makes the complementary count exact.
    let c_shift = unsafe { (LIMB_BITS as u32).unchecked_sub(shift) };
    let mut carry: Limb = 0;
    for i in 0..len {
        // SAFETY: Caller guarantees `dst` writable and `src` readable for
        // `len` elements, shift in 1..LIMB_BITS, and no aliasing.
        let val = unsafe { *src.add(i) };
        // SAFETY: i < len selects a writable destination limb. The source was
        // read before this store, and the spans are disjoint.
        unsafe {
            *dst.add(i) = (val << shift) | carry;
        }
        carry = val >> c_shift;
    }
    carry
}
