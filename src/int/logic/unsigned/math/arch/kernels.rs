//! Architecture-selected limb-kernel namespace.
//!
//! Each limb is a base-`B` digit, where `B = 2^LIMB_BITS`.

#![expect(
    unsafe_code,
    reason = "The facade forwards the documented pointer contracts to selected kernels"
)]

use super::{
    DoubleLimb, LIMB_BITS, Limb, add_limbs_3_unchecked, add_limbs_unchecked,
    add_mul_limbs_unchecked, add_reverse_sub_limbs_unchecked, add_sub_limbs_unchecked,
    add_two_limbs_unchecked, divrem_1_unchecked, mul_2x2_portable_unchecked,
    mul_3x3_portable_unchecked, mul_4x4_unchecked, mul_8x8_unchecked, mul_basecase_unchecked,
    propagate_borrow_unchecked, propagate_carry_unchecked, selected_add_mul_kernel,
    selected_monty_redc_kernel, selected_sub_mul_kernel, sqr_basecase_unchecked,
    sub_limbs_3_unchecked, sub_limbs_unchecked, sub_mul_limbs_unchecked,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{selected_add_sub_from_kernel, selected_sub_shifted_high_kernel};
with_direct_basecase_components! {
    use super::{selected_add_mul_2_kernel, selected_mul_2_kernel};
}

/// Namespace for architecture-selected limb kernels.
///
/// Operation modules select backends at compilation or cache CPU detection.
/// All pointer spans require aligned storage and byte lengths fitting in
/// `isize::MAX`. Read spans contain initialized limbs; write spans require
/// exclusive access and may be uninitialized when the method only writes them.
/// Zero-length spans permit null pointers. Method contracts specify lengths,
/// overlap rules and arithmetic domains.
#[derive(Clone, Copy, Debug)]
pub struct ArchKernels;

impl ArchKernels {
    /// Computes the full double-limb product of two limbs.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        reason = "Limb*Limb fits DoubleLimb; extracting its native halves is exact on every supported pointer width"
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "DoubleLimb (u64/u128) narrows to Limb (u32/u64) on 32-bit and 64-bit targets"
        )
    )]
    pub const fn mul_limb_lo_hi(left: Limb, right: Limb) -> (Limb, Limb) {
        // SAFETY: (B - 1)^2 < B^2 fits DoubleLimb on 16-, 32-, and 64-bit
        // targets; widening either limb preserves its value.
        let product = unsafe { (left as DoubleLimb).unchecked_mul(right as DoubleLimb) };
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

    /// Adds `src` to `dst` over `len` limbs and returns the binary carry.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for reads and writes of `len` limbs.
    /// - `src` must be valid for reads of `len` limbs.
    /// - Spans `dst[0..len]` and `src[0..len]` must be either completely disjoint
    ///   or identical pointers (`dst == src`).
    #[inline]
    pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
        // SAFETY: the caller provides len initialized destination and source
        // limbs with disjoint spans or exact self-aliasing. Selection establishes
        // the compiled or detected CPU prerequisites.
        unsafe { add_limbs_unchecked(dst, src, len) }
    }

    /// Subtracts `src` from `dst` over `len` limbs and returns the binary borrow.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for reads and writes of `len` limbs.
    /// - `src` must be valid for reads of `len` limbs.
    /// - Spans `dst[0..len]` and `src[0..len]` must be either completely disjoint
    ///   or identical pointers (`dst == src`).
    #[inline]
    pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
        // SAFETY: the caller provides len initialized destination and source
        // limbs with disjoint spans or exact self-aliasing. Selection establishes
        // the compiled or detected CPU prerequisites.
        unsafe { sub_limbs_unchecked(dst, src, len) }
    }

    /// Writes `src1 + src2` over `len` limbs and returns the binary carry.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for writes of `len` limbs.
    /// - `src1` and `src2` must be valid for reads of `len` limbs.
    /// - `dst[0..len]` must not overlap `src1[0..len]` or `src2[0..len]`.
    /// - `src1` and `src2` may alias each other or be disjoint.
    #[inline]
    pub unsafe fn add_limbs_3_unchecked(
        dst: *mut Limb,
        src1: *const Limb,
        src2: *const Limb,
        len: usize,
    ) -> Limb {
        // SAFETY: the caller provides len readable limbs per source and len
        // writable destination limbs disjoint from both sources. Source aliasing
        // is permitted; selection establishes the CPU prerequisites.
        unsafe { add_limbs_3_unchecked(dst, src1, src2, len) }
    }

    /// Writes `src1 - src2` over `len` limbs and returns the binary borrow.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for writes of `len` limbs.
    /// - `src1` and `src2` must be valid for reads of `len` limbs.
    /// - `dst[0..len]` must not overlap `src1[0..len]` or `src2[0..len]`.
    /// - `src1` and `src2` may alias each other or be disjoint.
    #[inline]
    pub unsafe fn sub_limbs_3_unchecked(
        dst: *mut Limb,
        src1: *const Limb,
        src2: *const Limb,
        len: usize,
    ) -> Limb {
        // SAFETY: the caller provides len readable limbs per source and len
        // writable destination limbs disjoint from both sources. Source aliasing
        // is permitted; selection establishes the CPU prerequisites.
        unsafe { sub_limbs_3_unchecked(dst, src1, src2, len) }
    }

    /// Returns the process-stable single-limb multiply-add kernel.
    #[inline]
    pub fn selected_add_mul_limbs_unchecked()
    -> unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb {
        selected_add_mul_kernel()
    }

    /// Adds `src * scalar` to `dst` over `len` limbs and returns the carry limb.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for reads and writes of `len` limbs.
    /// - `src` must be valid for reads of `len` limbs.
    /// - `dst[0..len]` and `src[0..len]` must not overlap.
    #[inline]
    pub unsafe fn add_mul_limbs_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        scalar: Limb,
    ) -> Limb {
        // SAFETY: the caller provides len initialized writable destination limbs
        // and len readable source limbs in disjoint spans. Selection establishes
        // the compiled or detected CPU prerequisites.
        unsafe { add_mul_limbs_unchecked(dst, src, len, scalar) }
    }

    /// Returns the process-stable single-limb multiply-subtract kernel.
    #[inline]
    pub fn selected_sub_mul_limbs_unchecked()
    -> unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb) {
        selected_sub_mul_kernel()
    }

    /// Evaluates `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
    ///
    /// Returns the high product limb and the binary subtraction borrow.
    ///
    /// # Safety
    ///
    /// - `dst` must be valid for reads and writes of `len` limbs.
    /// - `src` must be valid for reads of `len` limbs.
    /// - `dst[0..len]` and `src[0..len]` must not overlap.
    #[inline]
    pub unsafe fn sub_mul_limbs_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        scalar: Limb,
    ) -> (Limb, Limb) {
        // SAFETY: the caller provides len initialized writable destination limbs
        // and len readable source limbs in disjoint spans. Selection establishes
        // the compiled or detected CPU prerequisites.
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

    with_direct_basecase_components! {
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

    /// Replaces `(sum, difference)` with their sum and `sum - difference`.
    ///
    /// Returns the binary addition carry and subtraction borrow.
    ///
    /// # Safety
    ///
    /// - `sum` and `difference` must be valid for reads and writes of `len` limbs.
    /// - Spans `sum[0..len]` and `difference[0..len]` must not overlap.
    #[inline]
    pub unsafe fn add_sub_limbs_unchecked(
        sum: *mut Limb,
        difference: *mut Limb,
        len: usize,
    ) -> (Limb, Limb) {
        // SAFETY: the caller provides two initialized, writable, disjoint
        // len-limb spans. Selection establishes the CPU prerequisites.
        unsafe { add_sub_limbs_unchecked(sum, difference, len) }
    }

    /// Replaces `(sum, difference)` with their sum and `difference - sum`.
    ///
    /// Returns the binary addition carry and subtraction borrow.
    ///
    /// # Safety
    ///
    /// - `sum` and `difference` must be valid for reads and writes of `len` limbs.
    /// - Spans `sum[0..len]` and `difference[0..len]` must not overlap.
    #[inline]
    pub unsafe fn add_reverse_sub_limbs_unchecked(
        sum: *mut Limb,
        difference: *mut Limb,
        len: usize,
    ) -> (Limb, Limb) {
        // SAFETY: the caller provides two initialized, writable, disjoint
        // len-limb spans. Selection establishes the CPU prerequisites.
        unsafe { add_reverse_sub_limbs_unchecked(sum, difference, len) }
    }

    /// Adds each source to its destination and returns both binary carries.
    ///
    /// # Safety
    ///
    /// - Every pointer must cover `len` initialized, readable limbs; destinations
    ///   must also be writable.
    /// - Destination spans `dst_a[0..len]` and `dst_b[0..len]` must not overlap.
    /// - `dst_a[0..len]` must not overlap `src_b[0..len]`, and `dst_b[0..len]` must not overlap `src_a[0..len]`.
    /// - In-place self-aliasing is permitted: `dst_a` may equal `src_a`, and `dst_b` may equal `src_b`.
    ///   Otherwise each destination must be disjoint from its corresponding source.
    /// - Input spans `src_a[0..len]` and `src_b[0..len]` may alias or be disjoint.
    #[inline]
    pub unsafe fn add_two_limbs_unchecked(
        dst_a: *mut Limb,
        src_a: *const Limb,
        dst_b: *mut Limb,
        src_b: *const Limb,
        len: usize,
    ) -> (Limb, Limb) {
        // SAFETY: the caller provides initialized len-limb inputs and writable
        // disjoint destinations; only each chain's exact self-alias is permitted.
        // Selection establishes the CPU prerequisites.
        unsafe { add_two_limbs_unchecked(dst_a, src_a, dst_b, src_b, len) }
    }

    /// Returns `(q, r)` with `q * divisor + r = remainder_high * B + limb`.
    ///
    /// Both results fit one limb, and `r < divisor`.
    ///
    /// # Safety
    ///
    /// `divisor` must be nonzero and `remainder_high < divisor`.
    #[cfg_attr(
        not(any(
            all(target_arch = "x86_64", target_pointer_width = "64"),
            all(target_arch = "x86", target_pointer_width = "32"),
            all(target_arch = "s390x", target_pointer_width = "64")
        )),
        expect(
            clippy::missing_const_for_fn,
            reason = "Portable division backends are const-capable, but assembly backends are not; the architecture-neutral namespace requires one signature"
        )
    )]
    #[inline]
    pub unsafe fn divrem_1_unchecked(
        limb: Limb,
        remainder_high: Limb,
        divisor: Limb,
    ) -> (Limb, Limb) {
        // SAFETY: the caller excludes division by zero and bounds the high limb
        // below divisor, so the quotient fits a limb on every backend.
        unsafe { divrem_1_unchecked(limb, remainder_high, divisor) }
    }

    /// Adds a binary carry to `dst` and returns the residual binary carry.
    ///
    /// # Safety
    ///
    /// `dst` must cover `len` initialized readable/writable limbs, and `carry <= 1`.
    #[inline]
    pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, carry: Limb) -> Limb {
        // SAFETY: the caller provides len initialized writable limbs and a binary
        // carry. Selection establishes the compiled CPU prerequisites.
        unsafe { propagate_carry_unchecked(dst, len, carry) }
    }

    /// Subtracts a binary borrow from `dst` and returns the residual binary borrow.
    ///
    /// # Safety
    ///
    /// `dst` must cover `len` initialized readable/writable limbs, and `borrow <= 1`.
    #[inline]
    pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, borrow: Limb) -> Limb {
        // SAFETY: the caller provides len initialized writable limbs and a binary
        // borrow. Selection establishes the compiled CPU prerequisites.
        unsafe { propagate_borrow_unchecked(dst, len, borrow) }
    }

    /// Writes the complete `len_a + len_b` limb product of `a` and `b`.
    ///
    /// # Safety
    ///
    /// - `a` must cover `len_a >= 2` readable limbs.
    /// - `b` must cover `len_b > 0` readable limbs.
    /// - `dst` must cover `len_a + len_b` writable limbs.
    /// - `len_a + len_b` must not overflow; neither input may overlap `dst`.
    /// - Read-only inputs may overlap each other.
    #[inline]
    pub unsafe fn mul_basecase_unchecked(
        dst: *mut Limb,
        a: *const Limb,
        len_a: usize,
        b: *const Limb,
        len_b: usize,
    ) {
        // SAFETY: the caller provides len_a >= 2 readable limbs, len_b readable
        // limbs, and a disjoint writable product span with a representable total
        // width. Selection establishes the CPU prerequisites.
        unsafe { mul_basecase_unchecked(dst, a, len_a, b, len_b) }
    }

    /// Writes the complete `2 * len` limb square of `a`.
    ///
    /// # Safety
    ///
    /// - `a` must cover `len` readable limbs.
    /// - `dst` must cover `2 * len` writable limbs.
    /// - `dst` must not alias `a`.
    /// - `2 * len` must not overflow; zero length performs no writes.
    #[inline]
    pub unsafe fn sqr_basecase_unchecked(dst: *mut Limb, a: *const Limb, len: usize) {
        // SAFETY: the caller provides len readable input limbs and 2*len disjoint
        // writable output limbs with a representable width. Selection establishes
        // the CPU prerequisites.
        unsafe { sqr_basecase_unchecked(dst, a, len) }
    }

    /// Writes the four-limb product of two two-limb operands.
    ///
    /// # Safety
    ///
    /// `a` and `b` must each cover 2 readable limbs, `dst` must cover 4 writable limbs,
    /// and neither input may overlap `dst`.
    #[inline]
    pub unsafe fn mul_2x2_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
        // SAFETY: the caller provides two readable limbs per input and four
        // writable output limbs disjoint from both inputs.
        unsafe { mul_2x2_portable_unchecked(dst, a, b) }
    }

    /// Writes the six-limb product of two three-limb operands.
    ///
    /// # Safety
    ///
    /// `a` and `b` must each cover 3 readable limbs, `dst` must cover 6 writable limbs,
    /// and neither input may overlap `dst`.
    #[inline]
    pub unsafe fn mul_3x3_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
        // SAFETY: the caller provides three readable limbs per input and six
        // writable output limbs disjoint from both inputs.
        unsafe { mul_3x3_portable_unchecked(dst, a, b) }
    }

    /// Writes the eight-limb product of two four-limb operands.
    ///
    /// # Safety
    ///
    /// `a` and `b` must each cover 4 readable limbs, `dst` must cover 8 writable limbs,
    /// and neither input may overlap `dst`.
    #[inline]
    pub unsafe fn mul_4x4_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
        // SAFETY: the caller provides four readable limbs per input and eight
        // writable output limbs disjoint from both inputs. Selection establishes
        // the CPU prerequisites.
        unsafe { mul_4x4_unchecked(dst, a, b) }
    }

    /// Writes the sixteen-limb product of two eight-limb operands.
    ///
    /// # Safety
    ///
    /// `a` and `b` must each cover 8 readable limbs, `dst` must cover 16 writable limbs,
    /// and neither input may overlap `dst`.
    #[inline]
    pub unsafe fn mul_8x8_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
        // SAFETY: the caller provides eight readable limbs per input and sixteen
        // writable output limbs disjoint from both inputs. Selection establishes
        // the CPU prerequisites.
        unsafe { mul_8x8_unchecked(dst, a, b) }
    }
}
