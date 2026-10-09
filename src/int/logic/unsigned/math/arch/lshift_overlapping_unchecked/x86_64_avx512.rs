//! AVX-512 overlap-safe left shift for `x86_64`.

#![expect(
    clippy::cast_ptr_alignment,
    reason = "unaligned AVX-512 intrinsic pointers do not assert source alignment"
)]

use core::arch::x86_64::{
    __m512i, _mm512_alignr_epi64, _mm512_loadu_si512, _mm512_or_si512, _mm512_setzero_si512,
    _mm512_sll_epi64, _mm512_srl_epi64, _mm512_storeu_si512, _mm_set1_epi64x,
};

use super::Limb;

/// Shift `limbs[..len]` into `limbs[offset..offset + len]`, where `offset` may
/// be zero.
///
/// # Safety
///
/// For nonzero `len`, `offset + len` must fit in `usize` and the complete span
/// must be aligned, initialized, writable, and within `isize::MAX` bytes.
/// `shift` must be in `1..64`, and the CPU must support AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn lshift_overlapping_unchecked(
    limbs: *mut Limb,
    len: usize,
    offset: usize,
    shift: u32,
) -> Limb {
    // SAFETY: the caller establishes an aligned, initialized writable span,
    // AVX-512F support, and 0 < shift < 64. Each block runs while scalar_end > 8,
    // so index >= 1 and both loads lie below len. offset+index fits the span.
    // Descending stores occur after both overlapping loads. The final full
    // block, when present, loads limbs 0..8 before storing the displaced result.
    unsafe {
        if len == 0 {
            return 0;
        }
        let drop = 64_u32.unchecked_sub(shift);
        let carry = *limbs.add(len).sub(1) >> drop;
        let left = _mm_set1_epi64x(i64::from(shift));
        let right = _mm_set1_epi64x(i64::from(drop));
        let mut scalar_end = len;
        while scalar_end > 8 {
            let index = scalar_end.unchecked_sub(8);
            let upper = _mm512_loadu_si512(limbs.add(index).cast::<__m512i>());
            let lower = _mm512_loadu_si512(limbs.add(index).sub(1).cast::<__m512i>());
            let merged = _mm512_or_si512(
                _mm512_sll_epi64(upper, left),
                _mm512_srl_epi64(lower, right),
            );
            _mm512_storeu_si512(
                limbs.add(offset.unchecked_add(index)).cast::<__m512i>(),
                merged,
            );
            scalar_end = index;
        }
        if scalar_end == 8 {
            let upper = _mm512_loadu_si512(limbs.cast::<__m512i>());
            let dropped = _mm512_srl_epi64(upper, right);
            let incoming = _mm512_alignr_epi64(dropped, _mm512_setzero_si512(), 7);
            let merged = _mm512_or_si512(_mm512_sll_epi64(upper, left), incoming);
            _mm512_storeu_si512(limbs.add(offset).cast::<__m512i>(), merged);
            return carry;
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
