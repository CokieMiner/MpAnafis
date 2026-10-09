//! Architecture-neutral shift contracts, with base `B = 2^LIMB_BITS`.

#![expect(
    unsafe_code,
    reason = "The facade forwards validated limb spans and shift counts to selected backends"
)]

#[cfg(not(target_pointer_width = "16"))]
use super::selected_lshift_overlapping_kernel;
use super::{
    ArchKernels, Limb, selected_lshift_into_kernel, selected_lshift_into_small_kernel,
    selected_lshift_kernel, selected_rshift_into_kernel, selected_rshift_into_small_kernel,
    selected_rshift_kernel,
};

impl ArchKernels {
    /// Computes `limbs << shift mod B^len` and returns the shifted-out high bits.
    ///
    /// Returns zero without accessing memory when `len == 0`.
    ///
    /// # Safety
    ///
    /// - `limbs` must cover `len` initialized readable and writable limbs.
    /// - `0 < shift < Limb::BITS`.
    #[inline]
    pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
        // SAFETY: the caller supplies the writable span and valid shift count.
        // Selection establishes the backend's CPU prerequisites.
        unsafe { selected_lshift_kernel()(limbs, len, shift) }
    }

    /// Writes `src << shift mod B^len` into `dst` and returns shifted-out high bits.
    ///
    /// Returns zero without accessing memory when `len == 0`.
    ///
    /// # Safety
    ///
    /// - `src` must cover `len` initialized readable limbs.
    /// - `dst` must cover `len` writable limbs, disjoint from `src`.
    /// - `0 < shift < Limb::BITS`.
    #[inline]
    pub unsafe fn lshift_into_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: the caller supplies disjoint readable and writable spans
        // and a valid shift count. Selection proves the CPU prerequisites.
        unsafe { selected_lshift_into_kernel()(dst, src, len, shift) }
    }

    /// Writes a left shift using the directly compiled small-span backend.
    ///
    /// Returns the shifted-out high bits, or zero for an empty span.
    ///
    /// # Safety
    ///
    /// - `src` must cover `len` initialized readable limbs.
    /// - `dst` must cover `len` writable limbs, disjoint from `src`.
    /// - `0 < shift < Limb::BITS`.
    #[inline]
    pub unsafe fn lshift_into_small_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: the caller supplies disjoint spans and a valid shift count.
        // The small-span backend uses only compiled target features.
        unsafe { selected_lshift_into_small_kernel()(dst, src, len, shift) }
    }

    /// Left-shifts `limbs[0..len]` into `limbs[offset..offset + len]`.
    ///
    /// Descending traversal preserves overlapping source limbs. Returns the
    /// shifted-out high bits, or zero for an empty span.
    ///
    /// # Safety
    ///
    /// - `offset + len` must not overflow `usize`.
    /// - `limbs` must cover `offset + len` initialized readable and writable limbs.
    /// - `0 < shift < Limb::BITS`.
    #[cfg(not(target_pointer_width = "16"))]
    #[inline]
    pub unsafe fn lshift_overlapping_unchecked(
        limbs: *mut Limb,
        len: usize,
        offset: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: the caller proves the complete span, representable offset,
        // and shift count. Selection proves the CPU prerequisites; the backend
        // traverses high to low to preserve the overlapping source.
        unsafe { selected_lshift_overlapping_kernel()(limbs, len, offset, shift) }
    }

    /// Computes `limbs >> shift` and returns shifted-out low bits aligned high.
    ///
    /// The returned bits equal `old_limbs[0] << (Limb::BITS - shift)`.
    /// Returns zero without accessing memory when `len == 0`.
    ///
    /// # Safety
    ///
    /// - `limbs` must cover `len` initialized readable and writable limbs.
    /// - `0 < shift < Limb::BITS`.
    #[inline]
    pub unsafe fn rshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
        // SAFETY: the caller supplies the writable span and valid shift count.
        // Selection establishes the backend's CPU prerequisites.
        unsafe { selected_rshift_kernel()(limbs, len, shift) }
    }

    /// Writes `src >> shift` into `dst` and returns shifted-out low bits aligned high.
    ///
    /// The returned bits equal `src[0] << (Limb::BITS - shift)`.
    /// Returns zero without accessing memory when `len == 0`.
    ///
    /// # Safety
    ///
    /// - `src` must cover `len` initialized readable limbs.
    /// - `dst` must cover `len` writable limbs, disjoint from `src`.
    /// - `0 < shift < Limb::BITS`.
    #[inline]
    pub unsafe fn rshift_into_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: the caller supplies disjoint readable and writable spans
        // and a valid shift count. Selection proves the CPU prerequisites.
        unsafe { selected_rshift_into_kernel()(dst, src, len, shift) }
    }

    /// Writes a right shift using the directly compiled small-span backend.
    ///
    /// Returns shifted-out low bits aligned high, or zero for an empty span.
    ///
    /// # Safety
    ///
    /// - `src` must cover `len` initialized readable limbs.
    /// - `dst` must cover `len` writable limbs, disjoint from `src`.
    /// - `0 < shift < Limb::BITS`.
    #[inline]
    pub unsafe fn rshift_into_small_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: the caller supplies disjoint spans and a valid shift count.
        // The small-span backend uses only compiled target features.
        unsafe { selected_rshift_into_small_kernel()(dst, src, len, shift) }
    }
}
