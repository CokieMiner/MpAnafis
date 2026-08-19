//! Architecture-selected limb-kernel namespace.

#[cfg(any(
    test,
    not(all(
        feature = "std",
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(all(target_feature = "adx", target_feature = "bmi2"))
    ))
))]
use super::add_mul_2_limbs_unchecked::kernel as selected_add_mul_2_kernel;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "adx")
))]
use super::add_sub_limbs_unchecked::runtime_dispatch::fast_add_sub_limbs_available as selected_fast_add_sub_limbs_available;
#[cfg(any(
    test,
    not(all(
        feature = "std",
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(all(target_feature = "adx", target_feature = "bmi2"))
    ))
))]
use super::mul_2_limbs_unchecked::kernel as selected_mul_2_kernel;
#[cfg(feature = "_internal-tune")]
use super::ntt_monty_u32::{radix4_dif_scalar, radix4_dit_scalar};
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
    lshift_unchecked::kernel as selected_lshift_kernel,
    monty_redc_unchecked::kernel as selected_monty_redc_kernel,
    mul_basecase_unchecked::{
        mul_2x2_portable_unchecked, mul_3x3_portable_unchecked, mul_basecase_unchecked,
    },
    ntt_digits_u32::kernel as selected_ntt_digits_kernel,
    ntt_monty_u32::kernel as selected_ntt_monty_kernel,
    propagate_borrow_unchecked::propagate_borrow_unchecked,
    propagate_carry_unchecked::propagate_carry_unchecked,
    rshift_into_unchecked::kernel as selected_rshift_into_kernel,
    rshift_unchecked::kernel as selected_rshift_kernel,
    sub_limbs_3_unchecked::sub_limbs_3_unchecked,
    sub_limbs_unchecked::sub_limbs_unchecked,
    sub_mul_limbs_unchecked::{kernel as selected_sub_mul_kernel, sub_mul_limbs_unchecked},
};
#[cfg(all(
    feature = "_internal-tune",
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "avx2")
))]
use super::{X86SimdTier, selected_x86_simd_tier};
#[cfg(not(target_pointer_width = "16"))]
use super::{
    add_sub_from_limbs_unchecked::kernel as selected_add_sub_from_kernel,
    sub_shifted_high_limbs_unchecked::kernel as selected_sub_shifted_high_kernel,
};

/// Single-limb multiply-add kernel.
pub type AddMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb;
/// Single-limb multiply-subtract kernel.
pub type SubMulKernel = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb);
/// Fused two-scalar multiply-add kernel.
#[cfg(any(
    test,
    not(all(
        feature = "std",
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(all(target_feature = "adx", target_feature = "bmi2"))
    ))
))]
pub type AddMul2Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) -> (Limb, Limb);
/// Shared-source simultaneous addition/subtraction kernel.
#[cfg(not(target_pointer_width = "16"))]
pub type AddSubFromKernel = unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb);
/// One-step CIOS Montgomery reduction kernel.
pub type MontyKernel = unsafe fn(*mut Limb, *const Limb, *const Limb, usize, Limb, Limb) -> Limb;
/// In-place left-shift kernel over one writable span.
pub type LshiftKernel = unsafe fn(*mut Limb, usize, u32) -> Limb;
/// Out-of-place left-shift kernel from `src` into `dst`.
pub type LshiftIntoKernel = unsafe fn(*mut Limb, *const Limb, usize, u32) -> Limb;
/// In-place right-shift kernel over one writable span.
pub type RshiftKernel = unsafe fn(*mut Limb, usize, u32) -> Limb;
/// Out-of-place right-shift kernel from `src` into `dst`.
pub type RshiftIntoKernel = unsafe fn(*mut Limb, *const Limb, usize, u32) -> Limb;
/// Write-only two-row multiplication kernel.
#[cfg(any(
    test,
    not(all(
        feature = "std",
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(all(target_feature = "adx", target_feature = "bmi2"))
    ))
))]
pub type Mul2Kernel = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb);
/// Cross-limb shifted-high subtraction kernel.
#[cfg(not(target_pointer_width = "16"))]
pub type SubShiftedHighKernel = unsafe fn(*mut Limb, *const Limb, usize, u32, Limb) -> Limb;
/// Fused radix-4 NTT stage over four value quarters and two twiddle quarters.
#[cfg(feature = "_internal-tune")]
pub type NttRadix4Kernel = unsafe fn(*mut u32, *const u32, usize, u32, u32);

/// Architecture-selected 31-bit Montgomery NTT kernels.
#[derive(Clone, Copy, Debug)]
pub struct NttMontyKernels {
    pub mul_slice: unsafe fn(
        dst: *mut u32,
        a: *const u32,
        b: *const u32,
        len: usize,
        prime: u32,
        neg_inverse: u32,
    ),
    pub dif_butterfly: unsafe fn(
        low: *mut u32,
        high: *mut u32,
        twiddles: *const u32,
        len: usize,
        prime: u32,
        neg_inverse: u32,
    ),
    pub dit_butterfly: unsafe fn(
        low: *mut u32,
        high: *mut u32,
        twiddles: *const u32,
        len: usize,
        prime: u32,
        neg_inverse: u32,
    ),
    pub radix4_dif: unsafe fn(*mut u32, *const u32, usize, u32, u32),
    pub radix4_dit: unsafe fn(*mut u32, *const u32, usize, u32, u32),
}

/// Provider function type for Montgomery NTT kernels.
pub type NttMontyKernel = fn() -> NttMontyKernels;

/// Architecture-selected 16-bit digit-packing NTT kernel.
#[derive(Clone, Copy, Debug)]
pub struct NttDigitsKernels {
    pub pack_16: unsafe fn(*mut u32, *const u64, usize, usize) -> usize,
}

/// Provider function type for 16-bit NTT digit packing.
pub type NttDigitsKernel = fn() -> NttDigitsKernels;

/// Namespace for architecture-selected limb kernels.
///
/// Its methods preserve one architecture-neutral call surface while each
/// operation module owns its compile-time or runtime backend selection.
#[derive(Clone, Copy, Debug)]
pub struct ArchKernels;

impl ArchKernels {
    /// Returns the runtime-selected fused radix-4 kernels for controlled tuning.
    #[cfg(feature = "_internal-tune")]
    #[must_use]
    pub fn ntt_radix4_selected_kernels() -> (NttRadix4Kernel, NttRadix4Kernel) {
        let selected = selected_ntt_monty_kernel()();
        (selected.radix4_dif, selected.radix4_dit)
    }

    /// Returns the scalar fused radix-4 reference kernels for controlled tuning.
    #[cfg(feature = "_internal-tune")]
    #[must_use]
    pub const fn ntt_radix4_scalar_kernels() -> (NttRadix4Kernel, NttRadix4Kernel) {
        (radix4_dif_scalar, radix4_dit_scalar)
    }

    /// Names the fused radix-4 backend selected for this process.
    #[cfg(feature = "_internal-tune")]
    #[must_use]
    pub fn ntt_radix4_selected_backend_name() -> &'static str {
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "avx2"
        ))]
        {
            "avx2"
        }
        #[cfg(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(target_feature = "avx2")
        ))]
        {
            match selected_x86_simd_tier() {
                X86SimdTier::Avx2 => "avx2",
                X86SimdTier::Sse2 => "scalar",
            }
        }
        #[cfg(all(not(miri), target_arch = "aarch64", target_pointer_width = "64"))]
        {
            "neon"
        }
        #[cfg(not(any(
            all(not(miri), target_arch = "x86_64", target_pointer_width = "64"),
            all(not(miri), target_arch = "aarch64", target_pointer_width = "64")
        )))]
        {
            "scalar"
        }
    }

    /// Computes the full double-limb product of two limbs.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Limb*Limb fits DoubleLimb; extracting its native halves is exact on every supported pointer width"
    )]
    pub const fn mul_limb_lo_hi(left: Limb, right: Limb) -> (Limb, Limb) {
        let product = (left as DoubleLimb).wrapping_mul(right as DoubleLimb);
        let low = product as Limb;
        (low, (product >> LIMB_BITS) as Limb)
    }

    /// Returns whether the selected add/subtract backend has independent carry
    /// chains.
    #[cfg(all(
        feature = "std",
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(target_feature = "adx")
    ))]
    #[inline]
    pub fn fast_add_sub_limbs_available() -> bool {
        selected_fast_add_sub_limbs_available()
    }

    /// Returns whether the selected add/subtract backend has independent carry
    /// chains.
    #[cfg(not(all(
        feature = "std",
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(target_feature = "adx")
    )))]
    #[inline]
    pub const fn fast_add_sub_limbs_available() -> bool {
        cfg!(all(
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx"
        ))
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

    /// Returns the selected fused two-scalar multiply-add kernel.
    #[cfg(any(
        test,
        not(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))
    ))]
    #[inline]
    pub fn selected_add_mul_2_limbs_unchecked()
    -> unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) -> (Limb, Limb) {
        selected_add_mul_2_kernel()
    }

    /// Returns the selected write-only two-row multiplication kernel.
    #[cfg(any(
        test,
        not(all(
            feature = "std",
            not(miri),
            target_arch = "x86_64",
            target_pointer_width = "64",
            not(all(target_feature = "adx", target_feature = "bmi2"))
        ))
    ))]
    #[inline]
    pub fn selected_mul_2_limbs_unchecked() -> unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb)
    {
        selected_mul_2_kernel()
    }

    /// Returns whether the selected basecase should accumulate two rows at once.
    #[cfg(not(all(
        feature = "std",
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(all(target_feature = "adx", target_feature = "bmi2"))
    )))]
    #[inline]
    pub const fn prefer_add_mul_2_limbs() -> bool {
        !cfg!(all(
            target_arch = "x86_64",
            target_pointer_width = "64",
            target_feature = "adx",
            target_feature = "bmi2"
        ))
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

    /// Pointwise Montgomery multiplication over 31-bit integer slices.
    ///
    /// # Safety
    ///
    /// - `dst` covers `len` writable elements and `a` and `b` cover `len`
    ///   readable elements.
    /// - `a[..len]` and `b[..len]` contain lazy residues in `[0, 2 * prime)`;
    ///   the selected backend canonicalizes each operand in place of a
    ///   separate normalization pass.
    /// - `prime` is odd, nonzero, and strictly less than `2^31`, and
    ///   `neg_inverse == -prime^-1 mod 2^32`.
    /// - `dst` may exactly alias either input span, as required by in-place
    ///   pointwise multiplication and squaring. Any other destination/input
    ///   overlap is forbidden.
    #[inline]
    pub unsafe fn ntt_monty_mul_slice_unchecked(
        dst: *mut u32,
        a: *const u32,
        b: *const u32,
        len: usize,
        prime: u32,
        neg_inverse: u32,
    ) {
        // SAFETY: The caller establishes all complete spans.
        unsafe {
            (selected_ntt_monty_kernel()().mul_slice)(dst, a, b, len, prime, neg_inverse);
        }
    }

    /// Packs validated 64-bit limbs into 16-bit NTT digits.
    ///
    /// The transform calls this only when `Limb` is 64 bits, `digit_bits` is
    /// 16, and the destination has room for four digits per limb.  The
    /// architecture backend itself accepts a short destination to preserve
    /// the scalar helper's truncating behavior for direct internal tests.
    ///
    /// # Safety
    /// `limbs` is readable for `len` native limbs and `dst` is writable for
    /// `dst_len` `u32`s.  The caller proves that the native limbs are 64-bit
    /// before converting the input pointer to `u64`.
    pub unsafe fn ntt_digits_16_into(
        dst: *mut u32,
        limbs: *const Limb,
        len: usize,
        dst_len: usize,
    ) -> usize {
        // SAFETY: the caller proves `Limb` is u64 for this operation, and the
        // selected backend receives the same validated spans and capacity.
        unsafe { (selected_ntt_digits_kernel()().pack_16)(dst, limbs.cast(), len, dst_len) }
    }

    /// Radix-2 Decimation-in-Frequency butterfly pass over 31-bit integer slices.
    ///
    /// # Safety
    ///
    /// - `low` and `high` cover `len` writable elements and `twiddles` covers
    ///   `len` readable elements; all three active spans are pairwise disjoint.
    /// - `low[..len]` and `high[..len]` contain lazy residues in `[0, 2 * prime)`
    ///   and `twiddles[..len]` contains Montgomery residues in `[0, prime)`.
    /// - `prime` is odd, nonzero, and strictly less than `2^31`, and
    ///   `neg_inverse == -prime^-1 mod 2^32`.
    ///
    /// On return both writable spans contain lazy residues in `[0, 2 * prime)`.
    #[inline]
    pub unsafe fn ntt_dif_butterfly_unchecked(
        low: *mut u32,
        high: *mut u32,
        twiddles: *const u32,
        len: usize,
        prime: u32,
        neg_inverse: u32,
    ) {
        // SAFETY: The caller establishes all complete spans.
        unsafe {
            (selected_ntt_monty_kernel()().dif_butterfly)(
                low,
                high,
                twiddles,
                len,
                prime,
                neg_inverse,
            );
        }
    }

    /// Radix-2 Decimation-in-Time butterfly pass over 31-bit integer slices.
    ///
    /// # Safety
    ///
    /// - `low` and `high` cover `len` writable elements and `twiddles` covers
    ///   `len` readable elements; all three active spans are pairwise disjoint.
    /// - `low[..len]` and `high[..len]` contain lazy residues in `[0, 2 * prime)`
    ///   and `twiddles[..len]` contains Montgomery residues in `[0, prime)`.
    /// - `prime` is odd, nonzero, and strictly less than `2^31`, and
    ///   `neg_inverse == -prime^-1 mod 2^32`.
    ///
    /// On return both writable spans contain lazy residues in `[0, 2 * prime)`.
    #[inline]
    pub unsafe fn ntt_dit_butterfly_unchecked(
        low: *mut u32,
        high: *mut u32,
        twiddles: *const u32,
        len: usize,
        prime: u32,
        neg_inverse: u32,
    ) {
        // SAFETY: The caller establishes all complete spans.
        unsafe {
            (selected_ntt_monty_kernel()().dit_butterfly)(
                low,
                high,
                twiddles,
                len,
                prime,
                neg_inverse,
            );
        }
    }

    /// Applies a fused two-level radix-4 DIF stage.
    ///
    /// # Safety
    /// `values` covers four disjoint quarter spans, `twiddles` covers two
    /// quarter spans, and all residues satisfy the Montgomery kernel contract.
    #[inline]
    pub unsafe fn ntt_radix4_dif_unchecked(
        values: *mut u32,
        twiddles: *const u32,
        quarter_len: usize,
        prime: u32,
        neg_inverse: u32,
    ) {
        // SAFETY: the caller establishes all complete spans.
        unsafe {
            (selected_ntt_monty_kernel()().radix4_dif)(
                values,
                twiddles,
                quarter_len,
                prime,
                neg_inverse,
            );
        }
    }

    /// Applies a fused two-level radix-4 DIT stage.
    ///
    /// # Safety
    /// `values` covers four disjoint quarter spans, `twiddles` covers two
    /// quarter spans, and all residues satisfy the Montgomery kernel contract.
    #[inline]
    pub unsafe fn ntt_radix4_dit_unchecked(
        values: *mut u32,
        twiddles: *const u32,
        quarter_len: usize,
        prime: u32,
        neg_inverse: u32,
    ) {
        // SAFETY: the caller establishes all complete spans.
        unsafe {
            (selected_ntt_monty_kernel()().radix4_dit)(
                values,
                twiddles,
                quarter_len,
                prime,
                neg_inverse,
            );
        }
    }
}
