//! Flat coefficient matrix addressing for Fermat-ring FFT.
//!
//! All coefficients are stored contiguously in one `&mut [Limb]` buffer. Each
//! coefficient slot has a fixed width of `coeff_limbs` limbs, eliminating
//! per-element heap allocation entirely.

use super::{Limb, SsaTransform};

/// Coefficient addressing for flat FFT matrices.
///
/// Contributed to the [`SsaTransform`] namespace declared in
/// [`drive`](super::drive).
impl SsaTransform {
    /// Returns a shared slice of the coefficient at `index`.
    ///
    /// # Safety
    /// `index < transform_len` and `buf.len() >= transform_len * coeff_limbs`.
    #[allow(
        clippy::inline_always,
        reason = "zero-cost pointer arithmetic on the hot FFT path"
    )]
    #[inline(always)]
    pub unsafe fn coeff(buf: &[Limb], index: usize, coeff_limbs: usize) -> &[Limb] {
        let offset = index.wrapping_mul(coeff_limbs);
        // SAFETY: caller guarantees index < transform_len and buf is large enough.
        unsafe { buf.get_unchecked(offset..offset.wrapping_add(coeff_limbs)) }
    }

    /// Returns a mutable slice of the coefficient at `index`.
    ///
    /// # Safety
    /// `index < transform_len` and `buf.len() >= transform_len * coeff_limbs`.
    #[allow(
        clippy::inline_always,
        reason = "zero-cost pointer arithmetic on the hot FFT path"
    )]
    #[inline(always)]
    pub unsafe fn coeff_mut(buf: &mut [Limb], index: usize, coeff_limbs: usize) -> &mut [Limb] {
        let offset = index.wrapping_mul(coeff_limbs);
        // SAFETY: caller guarantees index < transform_len and buf is large enough.
        unsafe { buf.get_unchecked_mut(offset..offset.wrapping_add(coeff_limbs)) }
    }
}
