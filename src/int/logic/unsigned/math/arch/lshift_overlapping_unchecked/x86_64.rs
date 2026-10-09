//! SSE2 overlap-safe left shift for `x86_64`.

#![expect(
    clippy::cast_ptr_alignment,
    reason = "unaligned SSE2 intrinsic pointers do not assert source alignment"
)]

use core::arch::x86_64::{
    __m128i, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi64x, _mm_sll_epi64, _mm_srl_epi64,
    _mm_storeu_si128,
};

use super::Limb;

/// Shift `limbs[..len]` into `limbs[offset..offset + len]`, where `offset` may
/// be zero.
///
/// # Safety
///
/// For nonzero `len`, `offset + len` must fit in `usize` and the complete span
/// must be aligned, initialized, writable, and within `isize::MAX` bytes.
/// `shift` must be in `1..64`. SSE2 is enabled on `x86_64`.
pub unsafe fn lshift_overlapping_unchecked(
    limbs: *mut Limb,
    len: usize,
    offset: usize,
    shift: u32,
) -> Limb {
    // SAFETY: the caller establishes an aligned, initialized writable span
    // and 0 < shift < 64. Each pair runs only while scalar_end > 2, so its
    // index >= 1 and both source loads lie below len. offset+index fits the
    // complete span. Descending stores occur after both overlapping loads.
    unsafe {
        if len == 0 {
            return 0;
        }
        let drop = 64_u32.unchecked_sub(shift);
        let carry = *limbs.add(len).sub(1) >> drop;
        let left = _mm_set1_epi64x(i64::from(shift));
        let right = _mm_set1_epi64x(i64::from(drop));
        let mut scalar_end = len;
        while scalar_end > 2 {
            let index = scalar_end.unchecked_sub(2);
            let upper = _mm_loadu_si128(limbs.add(index).cast::<__m128i>());
            let lower = _mm_loadu_si128(limbs.add(index).sub(1).cast::<__m128i>());
            let merged = _mm_or_si128(_mm_sll_epi64(upper, left), _mm_srl_epi64(lower, right));
            _mm_storeu_si128(
                limbs.add(offset.unchecked_add(index)).cast::<__m128i>(),
                merged,
            );
            scalar_end = index;
        }
        while scalar_end > 1 {
            scalar_end = scalar_end.unchecked_sub(1);
            *limbs.add(offset.unchecked_add(scalar_end)) = (*limbs.add(scalar_end) << shift)
                | (*limbs.add(scalar_end).sub(1) >> drop);
        }
        *limbs.add(offset) = *limbs << shift;
        carry
    }
}
