//! Architecture-selected limb-kernel namespace.

use super::{
    DoubleLimb, LIMB_BITS, Limb,
    add_limbs_3_unchecked::add_limbs_3_unchecked,
    add_limbs_unchecked::add_limbs_unchecked,
    add_mul_limbs_unchecked::{add_mul_limbs_unchecked, kernel as selected_add_mul_kernel},
    add_reverse_sub_limbs_unchecked::add_reverse_sub_limbs_unchecked,
    add_sub_limbs_unchecked::add_sub_limbs_unchecked,
    add_two_limbs_unchecked::add_two_limbs_unchecked,
    divrem_1_unchecked::divrem_1_unchecked,
    lshift_into_unchecked::kernel as selected_lshift_into_kernel,
    lshift_overlapping_unchecked::kernel as selected_lshift_overlapping_kernel,
    lshift_unchecked::kernel as selected_lshift_kernel,
    monty_redc_unchecked::kernel as selected_monty_redc_kernel,
    mul_basecase_unchecked::{
        mul_2x2_portable_unchecked, mul_3x3_portable_unchecked, mul_basecase_unchecked,
    },
    propagate_borrow_unchecked::propagate_borrow_unchecked,
    propagate_carry_unchecked::propagate_carry_unchecked,
    rshift_into_unchecked::kernel as selected_rshift_into_kernel,
    rshift_unchecked::kernel as selected_rshift_kernel,
    sub_limbs_3_unchecked::sub_limbs_3_unchecked,
    sub_limbs_unchecked::sub_limbs_unchecked,
    sub_mul_limbs_unchecked::{kernel as selected_sub_mul_kernel, sub_mul_limbs_unchecked},
};
with_direct_basecase_components! {
    use super::add_mul_2_limbs_unchecked::kernel as selected_add_mul_2_kernel;
    use super::mul_2_limbs_unchecked::kernel as selected_mul_2_kernel;
}
#[cfg(not(target_pointer_width = "16"))]
use super::{
    add_sub_from_limbs_unchecked::kernel as selected_add_sub_from_kernel,
    sub_shifted_high_limbs_unchecked::kernel as selected_sub_shifted_high_kernel,
};

/// Single-limb multiply-add kernel.
pub type AddMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb;
/// Single-limb multiply-subtract kernel.
pub type SubMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb);
with_direct_basecase_components! {
    /// Fused two-scalar multiply-add kernel.
    pub type AddMul2Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) -> (Limb, Limb);
}
/// Shared-source simultaneous addition/subtraction kernel.
#[cfg(not(target_pointer_width = "16"))]
pub type AddSubFromKernel = unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb);
/// One-step CIOS Montgomery reduction kernel.
pub type MontyKernel = unsafe fn(*mut Limb, *const Limb, *const Limb, usize, Limb, Limb) -> Limb;
/// In-place left-shift kernel over one writable span.
pub type LshiftKernel = unsafe fn(*mut Limb, usize, u32) -> Limb;
/// Out-of-place left-shift kernel from `src` into `dst`.
pub type LshiftIntoKernel = unsafe fn(*mut Limb, *const Limb, usize, u32) -> Limb;
/// Overlap-safe left shift from a prefix into the same or a higher suffix.
pub type LshiftOverlappingKernel = unsafe fn(*mut Limb, usize, usize, u32) -> Limb;
/// In-place right-shift kernel over one writable span.
pub type RshiftKernel = unsafe fn(*mut Limb, usize, u32) -> Limb;
/// Out-of-place right-shift kernel from `src` into `dst`.
pub type RshiftIntoKernel = unsafe fn(*mut Limb, *const Limb, usize, u32) -> Limb;
with_direct_basecase_components! {
    /// Write-only two-row multiplication kernel.
    pub type Mul2Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb);
}
/// Cross-limb shifted-high subtraction kernel.
#[cfg(not(target_pointer_width = "16"))]
pub type SubShiftedHighKernel = unsafe fn(*mut Limb, *const Limb, usize, u32, Limb) -> Limb;

/// Namespace for architecture-selected limb kernels.
///
/// Its methods preserve one architecture-neutral call surface while each
/// operation module owns its compile-time or runtime backend selection.
#[derive(Clone, Copy, Debug)]
pub struct ArchKernels;

impl ArchKernels {
    /// Computes the full double-limb product of two limbs.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Limb*Limb fits DoubleLimb; extracting its native halves is exact on every supported pointer width"
    )]
    pub const fn mul_limb_lo_hi(left: Limb, right: Limb) -> (Limb, Limb) {
        let product = (left as DoubleLimb).wrapping_mul(right as DoubleLimb);
        let low = product as Limb;
        (low, (product >> LIMB_BITS) as Limb)
    }

    /// Returns the selected shared-source addition/subtraction kernel.
    #[cfg(not(target_pointer_width = "16"))]
    #[inline]
    pub fn selected_add_sub_from_limbs_unchecked()
    -> unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb) {
        selected_add_sub_from_kernel()
    }

    /// Returns the selected Montgomery reduction-step kernel.
    #[inline]
    pub fn selected_monty_redc_step_unchecked()
    -> unsafe fn(*mut Limb, *const Limb, *const Limb, usize, Limb, Limb) -> Limb {
        selected_monty_redc_kernel()
    }

    /// Returns the selected shifted-high subtraction kernel.
    #[cfg(not(target_pointer_width = "16"))]
    #[inline]
    pub fn selected_sub_shifted_high_limbs_unchecked()
    -> unsafe fn(*mut Limb, *const Limb, usize, u32, Limb) -> Limb {
        selected_sub_shifted_high_kernel()
    }

    /// Adds `len` source limbs into the destination and returns the carry.
    ///
    /// # Safety
    ///
    /// `dst` must be writable and `src` readable for `len` limbs. The regions
    /// must satisfy the aliasing contract of the selected backend.
    #[inline]
    pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
        // SAFETY: The caller guarantees the selected kernel's pointer, length,
        // and aliasing requirements.
        unsafe { add_limbs_unchecked(dst, src, len) }
    }

    /// Subtracts `len` source limbs from the destination and returns the borrow.
    ///
    /// # Safety
    ///
    /// `dst` must be writable and `src` readable for `len` limbs. The regions
    /// must satisfy the aliasing contract of the selected backend.
    #[inline]
    pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
        // SAFETY: The caller guarantees the selected kernel's pointer, length,
        // and aliasing requirements.
        unsafe { sub_limbs_unchecked(dst, src, len) }
    }

    /// Adds two `len`-limb sources into the destination and returns the carry.
    ///
    /// # Safety
    ///
    /// `dst` must be writable and both sources readable for `len` limbs. The
    /// regions must satisfy the aliasing contract of the selected backend.
    #[inline]
    pub unsafe fn add_limbs_3_unchecked(
        dst: *mut Limb,
        src1: *const Limb,
        src2: *const Limb,
        len: usize,
    ) -> Limb {
        // SAFETY: The caller guarantees the selected kernel's pointer, length,
        // and aliasing requirements.
        unsafe { add_limbs_3_unchecked(dst, src1, src2, len) }
    }

    /// Subtracts the second source from the first into `dst`.
    ///
    /// Returns the final borrow.
    ///
    /// # Safety
    ///
    /// `dst` must be writable and both sources readable for `len` limbs. The
    /// regions must satisfy the aliasing contract of the selected backend.
    #[inline]
    pub unsafe fn sub_limbs_3_unchecked(
        dst: *mut Limb,
        src1: *const Limb,
        src2: *const Limb,
        len: usize,
    ) -> Limb {
        // SAFETY: The caller guarantees the selected kernel's pointer, length,
        // and aliasing requirements.
        unsafe { sub_limbs_3_unchecked(dst, src1, src2, len) }
    }

    /// Returns the process-stable single-limb multiply-add kernel.
    #[inline]
    pub fn selected_add_mul_limbs_unchecked()
    -> unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb {
        selected_add_mul_kernel()
    }

    /// Multiplies `src` by one limb and accumulates the product into `dst`.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must each cover `len` limbs and must satisfy the
    /// selected backend's non-aliasing contract.
    #[inline]
    pub unsafe fn add_mul_limbs_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        scalar: Limb,
    ) -> Limb {
        // SAFETY: The caller guarantees the selected kernel's pointer, length,
        // and aliasing requirements.
        unsafe { add_mul_limbs_unchecked(dst, src, len, scalar) }
    }

    /// Returns the process-stable single-limb multiply-subtract kernel.
    #[inline]
    pub fn selected_sub_mul_limbs_unchecked()
    -> unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb) {
        selected_sub_mul_kernel()
    }

    /// Multiplies `src` by one limb and subtracts the product from `dst`.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must each cover `len` limbs and must satisfy the
    /// selected backend's non-aliasing contract.
    #[inline]
    pub unsafe fn sub_mul_limbs_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        scalar: Limb,
    ) -> (Limb, Limb) {
        // SAFETY: The caller guarantees the selected kernel's pointer, length,
        // and aliasing requirements.
        unsafe { sub_mul_limbs_unchecked(dst, src, len, scalar) }
    }

    with_direct_basecase_components! {
        /// Returns the selected fused two-scalar multiply-add kernel.
        #[inline]
        pub fn selected_add_mul_2_limbs_unchecked()
        -> unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) -> (Limb, Limb) {
            selected_add_mul_2_kernel()
        }

        /// Returns the selected write-only two-row multiplication kernel.
        #[inline]
        pub fn selected_mul_2_limbs_unchecked()
        -> unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) {
            selected_mul_2_kernel()
        }

    }

    with_direct_basecase_composition! {
        /// Returns whether the selected basecase should accumulate two rows at once.
        #[inline]
        pub const fn prefer_add_mul_2_limbs() -> bool {
            !cfg!(all(
                target_arch = "x86_64",
                target_pointer_width = "64",
                target_feature = "adx",
                target_feature = "bmi2"
            ))
        }
    }

    /// Simultaneously add and subtract two disjoint limb spans.
    ///
    /// # Safety
    ///
    /// Both pointers must cover `len` writable limbs and the spans must not
    /// overlap.
    #[inline]
    pub unsafe fn add_sub_limbs_unchecked(
        sum: *mut Limb,
        difference: *mut Limb,
        len: usize,
    ) -> (Limb, Limb) {
        // SAFETY: The caller establishes both writable disjoint spans.
        unsafe { add_sub_limbs_unchecked(sum, difference, len) }
    }

    /// Simultaneously add and reverse-subtract two disjoint limb spans.
    ///
    /// # Safety
    ///
    /// Both pointers must cover `len` writable limbs and the spans must not
    /// overlap.
    #[inline]
    pub unsafe fn add_reverse_sub_limbs_unchecked(
        sum: *mut Limb,
        difference: *mut Limb,
        len: usize,
    ) -> (Limb, Limb) {
        // SAFETY: The caller establishes both writable disjoint spans.
        unsafe { add_reverse_sub_limbs_unchecked(sum, difference, len) }
    }

    /// Perform two independent same-width additions.
    ///
    /// # Safety
    ///
    /// Every pointer must cover `len` limbs and no destination may overlap
    /// another span.
    #[inline]
    pub unsafe fn add_two_limbs_unchecked(
        dst_a: *mut Limb,
        src_a: *const Limb,
        dst_b: *mut Limb,
        src_b: *const Limb,
        len: usize,
    ) -> (Limb, Limb) {
        // SAFETY: The caller establishes all four spans and non-overlap.
        unsafe { add_two_limbs_unchecked(dst_a, src_a, dst_b, src_b, len) }
    }

    /// Divide a two-limb numerator by one limb.
    ///
    /// # Safety
    ///
    /// `divisor` must be nonzero and `remainder_high < divisor`.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "Portable division backends are const-capable, but assembly backends are not; the architecture-neutral namespace requires one signature"
    )]
    #[inline]
    pub unsafe fn divrem_1_unchecked(
        limb: Limb,
        remainder_high: Limb,
        divisor: Limb,
    ) -> (Limb, Limb) {
        // SAFETY: The caller establishes the divisor and high-limb bounds.
        unsafe { divrem_1_unchecked(limb, remainder_high, divisor) }
    }

    /// Shift a nonempty limb span left in place.
    ///
    /// # Safety
    ///
    /// `limbs` must cover `len` writable limbs and `shift` must be in
    /// `1..Limb::BITS`.
    #[inline]
    pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
        // SAFETY: The caller establishes the span and shift bounds; the
        // selected backend needs no extra CPU features beyond its tier.
        unsafe { selected_lshift_kernel()(limbs, len, shift) }
    }

    /// Shift a nonempty limb span left into a separate destination.
    ///
    /// Writes `dst[0..len] = src[0..len] << shift` (merged across limb
    /// boundaries) and returns `src[len-1] >> (Limb::BITS - shift)`.
    ///
    /// # Safety
    ///
    /// `dst` must be writable and `src` readable for `len` limbs, the spans
    /// must not overlap, and `shift` must be in `1..Limb::BITS`.
    #[inline]
    pub unsafe fn lshift_into_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: The caller establishes the spans, non-aliasing, and shift
        // bounds; the selected backend needs no extra CPU features beyond its
        // tier.
        unsafe { selected_lshift_into_kernel()(dst, src, len, shift) }
    }

    /// Shift a limb prefix into the same or an overlapping higher destination.
    ///
    /// # Safety
    ///
    /// `limbs` must cover `offset + len` initialized writable limbs, `offset +
    /// len` must be representable, and `shift` must be in `1..Limb::BITS`.
    #[inline]
    pub unsafe fn lshift_overlapping_unchecked(
        limbs: *mut Limb,
        len: usize,
        offset: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: the caller establishes the complete span and shift bounds;
        // the selected backend traverses high to low to preserve overlap.
        unsafe { selected_lshift_overlapping_kernel()(limbs, len, offset, shift) }
    }

    /// Shift a nonempty limb span right in place.
    ///
    /// # Safety
    ///
    /// `limbs` must cover `len` writable limbs and `shift` must be in
    /// `1..Limb::BITS`.
    #[inline]
    pub unsafe fn rshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
        // SAFETY: The caller establishes the span and shift bounds; the
        // selected backend needs no extra CPU features beyond its tier.
        unsafe { selected_rshift_kernel()(limbs, len, shift) }
    }

    /// Shift a nonempty limb span right into a separate destination.
    ///
    /// Writes `dst[0..len] = src[0..len] >> shift` (merged across limb
    /// boundaries) and returns `src[0] << (Limb::BITS - shift)`, the bits
    /// shifted out of the bottom limb.
    ///
    /// # Safety
    ///
    /// `dst` must be writable and `src` readable for `len` limbs, the spans
    /// must not overlap, and `shift` must be in `1..Limb::BITS`.
    #[inline]
    pub unsafe fn rshift_into_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        shift: u32,
    ) -> Limb {
        // SAFETY: The caller establishes the spans, non-aliasing, and shift
        // bounds; the selected backend needs no extra CPU features beyond its
        // tier.
        unsafe { selected_rshift_into_kernel()(dst, src, len, shift) }
    }

    /// Propagate an incoming carry through a writable limb span.
    ///
    /// # Safety
    ///
    /// `dst` must cover `len` writable limbs.
    #[inline]
    pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, carry: Limb) -> Limb {
        // SAFETY: The caller establishes the writable span.
        unsafe { propagate_carry_unchecked(dst, len, carry) }
    }

    /// Propagate an incoming borrow through a writable limb span.
    ///
    /// # Safety
    ///
    /// `dst` must cover `len` writable limbs.
    #[inline]
    pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, borrow: Limb) -> Limb {
        // SAFETY: The caller establishes the writable span.
        unsafe { propagate_borrow_unchecked(dst, len, borrow) }
    }

    /// Compute a complete schoolbook product.
    ///
    /// # Safety
    ///
    /// Both inputs and the `len_a + len_b` destination must cover their stated
    /// lengths, and neither input may overlap the destination.
    #[inline]
    pub unsafe fn mul_basecase_unchecked(
        dst: *mut Limb,
        a: *const Limb,
        len_a: usize,
        b: *const Limb,
        len_b: usize,
    ) {
        // SAFETY: The caller establishes all complete input/output spans.
        unsafe { mul_basecase_unchecked(dst, a, len_a, b, len_b) }
    }

    /// Compute a portable two-by-two-limb product.
    ///
    /// # Safety
    ///
    /// Both inputs must cover two limbs and `dst` four writable limbs, with no
    /// input overlapping the destination.
    #[inline]
    pub unsafe fn mul_2x2_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
        // SAFETY: The caller establishes the exact input/output spans.
        unsafe { mul_2x2_portable_unchecked(dst, a, b) }
    }

    /// Compute a portable three-by-three-limb product.
    ///
    /// # Safety
    ///
    /// Both inputs must cover three limbs and `dst` six writable limbs, with
    /// no input overlapping the destination.
    #[inline]
    pub unsafe fn mul_3x3_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
        // SAFETY: The caller establishes the exact input/output spans.
        unsafe { mul_3x3_portable_unchecked(dst, a, b) }
    }
}
