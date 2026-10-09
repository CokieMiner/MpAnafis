//! `x86_64` AVX2 in-place left-shift kernel.
//!
//! 256-bit loop (`vpsllq`/`vpsrlq`/`vpor`) processing four limbs per
//! iteration, selected at runtime only when the host (including any VM the
//! process runs in) reports AVX2 via CPUID. The loop descends so each store
//! lands above the next iteration's read window, keeping the in-place
//! contract safe; a scalar tail covers the low limbs and a fully scalar path
//! covers spans shorter than one vector.

#![expect(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics require typed `__m128i`/`__m256i` pointers while guaranteeing unaligned access; the cast from `Limb` pointers is the intrinsic API itself and asserts no alignment."
)]

use core::arch::x86_64::{
    __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_sll_epi64, _mm256_srl_epi64,
    _mm256_storeu_si256, _mm_set1_epi64x,
};

use super::Limb;

/// Left-shift `len` limbs in-place by `shift` bits (0 < shift < `LIMB_BITS`).
/// Returns the bits shifted out of the top limb.
///
/// # Safety
///
/// The CPU must support AVX2. For nonzero `len`, `limbs` must cover `len`
/// aligned, initialized, writable limbs within `isize::MAX` bytes.
/// `shift` must satisfy `0 < shift < 64`.
#[target_feature(enable = "avx2")]
pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    // SAFETY: the caller provides the initialized writable span, AVX2, and
    // 0 < shift < 64. Each quad runs while end > 4, so index=end-4 >= 1 and
    // both loads stay below len. Descending stores follow both loads and cannot
    // replace unread lower limbs. The scalar tail ends at limb zero.
    unsafe {
        if len == 0 {
            return 0;
        }
        // Kernel contract: 0 < shift < LIMB_BITS, so 64 - shift cannot
        // underflow; `add(len)` lands one past the end, `sub(1)` on the top
        // limb, so no out-of-bounds access is possible.
        let drop = 64_u32.unchecked_sub(shift);
        let carry_out = *limbs.add(len).sub(1) >> drop;

        // Quad at `index` writes positions index..index + 4 from sources
        // index..index + 4 and index - 1..index + 3. Descending by 4 keeps
        // each store above the next quad's reads, so in-place shifting never
        // consumes a written limb. The loop covers positions
        // `last_quad..len`; the scalar tail below covers the remaining
        // 0..`last_quad`.
        let tail_end = if len >= 5 {
            let left = _mm_set1_epi64x(i64::from(shift));
            let right = _mm_set1_epi64x(i64::from(drop));
            let mut last_quad = len;
            while last_quad > 4 {
                let index = last_quad.unchecked_sub(4);
                let upper = _mm256_loadu_si256(limbs.add(index).cast::<__m256i>());
                let lower = _mm256_loadu_si256(limbs.add(index).sub(1).cast::<__m256i>());
                let merged = _mm256_or_si256(
                    _mm256_sll_epi64(upper, left),
                    _mm256_srl_epi64(lower, right),
                );
                _mm256_storeu_si256(limbs.add(index).cast::<__m256i>(), merged);
                last_quad = index;
            }
            last_quad
        } else {
            len
        };

        // Scalar tail, descending so writes never feed the merge of a lower
        // limb: out[i] = (src[i] << shift) | (src[i - 1] >> drop). The guard
        // i >= 2 keeps i - 1 and i - 2 in bounds.
        let mut i = tail_end;
        while i >= 2 {
            *limbs.add(i).sub(1) = (*limbs.add(i).sub(1) << shift) | (*limbs.add(i).sub(2) >> drop);
            i = i.unchecked_sub(1);
        }
        *limbs <<= shift;
        carry_out
    }
}
