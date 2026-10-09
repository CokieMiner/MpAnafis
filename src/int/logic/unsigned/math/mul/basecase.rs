//! Schoolbook multiplication, squaring, and raw basecase kernels.

#![expect(
    unsafe_code,
    reason = "Validated disjoint limb spans bound raw kernels; scalar products and loop indices have local nonoverflow proofs"
)]

use core::{
    mem::MaybeUninit,
    ptr::eq,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use super::{ArchKernels, DoubleLimb, LIMB_BITS, Limb};

/// Namespace for schoolbook multiplication and its raw basecase kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Schoolbook;

/// Limb storage whose complete initialization is established by a product writer.
///
/// Both ordinary limbs and `MaybeUninit<Limb>` retain the same mutable-slice
/// exclusivity. The latter permits a first writer to initialize a newly reserved
/// destination without an earlier zero pass.
///
/// # Safety
/// Implementations must have the size and alignment of `Limb`, accept every
/// initialized limb representation, and preserve that representation in
/// `from_limb`. They must have no destruction or additional validity obligations.
pub unsafe trait LimbOutput: Copy + Send {
    /// Stores an initialized limb in this destination representation.
    fn from_limb(value: Limb) -> Self;

    /// Views initialized scratch as writable product storage of the same layout.
    ///
    /// # Safety
    /// The writer must leave every element initialized before the original
    /// limb borrow resumes, including during unwinding.
    unsafe fn from_initialized_mut(output: &mut [Limb]) -> &mut [Self] {
        // SAFETY: the implementation accepts every initialized limb bit pattern
        // and has exactly its size and alignment. The exclusive borrow and
        // lifetime are preserved; no element is moved or deinitialized.
        unsafe { from_raw_parts_mut(output.as_mut_ptr().cast(), output.len()) }
    }

    /// Borrows an endpoint after its first writer initialized every element.
    ///
    /// # Safety
    /// Every element of `output` must contain an initialized `Limb`.
    unsafe fn assume_init(output: &[Self]) -> &[Limb] {
        // SAFETY: the implementation supplies the exact limb layout and the
        // caller proves full initialization. This read-only view retains the
        // original borrow and lifetime without strengthening its alias contract.
        unsafe { from_raw_parts(output.as_ptr().cast(), output.len()) }
    }

    /// Borrows a destination after its first writer initialized every element.
    ///
    /// # Safety
    /// Every element of `output` must contain an initialized `Limb`.
    unsafe fn assume_init_mut(output: &mut [Self]) -> &mut [Limb] {
        // SAFETY: the implementation contract supplies the exact limb layout;
        // the caller proves initialization of the entire borrowed span. The
        // returned slice retains the exclusive borrow and its original lifetime.
        unsafe { from_raw_parts_mut(output.as_mut_ptr().cast(), output.len()) }
    }
}

// SAFETY: Limb is its own initialized storage representation and has no drop.
unsafe impl LimbOutput for Limb {
    fn from_limb(value: Limb) -> Self {
        value
    }
}

// SAFETY: MaybeUninit<Limb> has the exact size/alignment of Limb, accepts every
// limb representation, and has no drop; new preserves the supplied limb bits.
unsafe impl LimbOutput for MaybeUninit<Limb> {
    fn from_limb(value: Limb) -> Self {
        Self::new(value)
    }
}

impl Schoolbook {
    /// Computes the square of a limb slice using schoolbook squaring.
    ///
    /// `dst` must have at least `2 * a_limbs.len()` elements. Every active
    /// destination limb is initialized before it is read.
    pub fn sqr(dst: &mut [impl LimbOutput], a_limbs: &[Limb]) {
        if a_limbs.is_empty() {
            return;
        }
        Self::sqr_nonempty(dst, a_limbs);
    }

    /// Squares a nonempty operand whose complete destination is already reserved.
    #[inline]
    pub fn sqr_nonempty(dst: &mut [impl LimbOutput], a_limbs: &[Limb]) {
        let len = a_limbs.len();
        debug_assert!(len != 0, "validated square operand must be nonempty");
        debug_assert!(
            dst.len() >= len.saturating_mul(2),
            "square destination buffer is too small"
        );

        // SAFETY:
        // - dst has length >= 2 * len and does not overlap a_limbs.
        // - a_limbs is valid for reads of length len.
        // - len > 0.
        unsafe {
            ArchKernels::sqr_basecase_unchecked(dst.as_mut_ptr().cast(), a_limbs.as_ptr(), len);
        }
    }

    /// Computes the product of two limb slices using schoolbook multiplication.
    ///
    /// `dst` must have at least `a_limbs.len() + b_limbs.len()` elements.
    pub fn mul(dst: &mut [impl LimbOutput], a_limbs: &[Limb], b_limbs: &[Limb]) {
        if a_limbs.is_empty() || b_limbs.is_empty() {
            return;
        }
        if eq(a_limbs.as_ptr(), b_limbs.as_ptr()) && a_limbs.len() == b_limbs.len() {
            Self::sqr(dst, a_limbs);
            return;
        }
        Self::mul_nonempty_distinct(dst, a_limbs, b_limbs);
    }

    /// Multiply already-validated nonempty, distinct operand slices.
    ///
    /// Callers must have excluded the equal-slice squaring case and provided a
    /// destination at least `a_limbs.len() + b_limbs.len()` limbs long. Valid Rust
    /// borrows prove that the mutable destination does not overlap either input.
    #[inline]
    pub fn mul_nonempty_distinct(dst: &mut [impl LimbOutput], a_limbs: &[Limb], b_limbs: &[Limb]) {
        debug_assert!(
            !a_limbs.is_empty() && !b_limbs.is_empty(),
            "validated schoolbook operands must be nonempty"
        );
        debug_assert!(
            dst.len() >= a_limbs.len().saturating_add(b_limbs.len()),
            "schoolbook destination is shorter than the complete product"
        );

        let (outer, inner) = if a_limbs.len() <= b_limbs.len() {
            (a_limbs, b_limbs)
        } else {
            (b_limbs, a_limbs)
        };

        let outer_len = outer.len();
        let inner_len = inner.len();

        if outer_len == inner_len {
            Self::mul_equal_nonempty_distinct(dst, inner, outer);
            return;
        }

        if outer.len() == 1 {
            // SAFETY: both operands are nonempty; dst covers inner.len()+1 limbs,
            // and schoolbook multiplication requires disjoint input/output spans.
            unsafe {
                Self::mul_limb_unchecked(
                    dst.as_mut_ptr().cast(),
                    inner.as_ptr(),
                    inner.len(),
                    *outer.get_unchecked(0),
                );
            }
            return;
        }

        // SAFETY:
        // - dst has at least outer.len() + inner.len() writable limbs. The raw
        //   basecase initializes its first row before any accumulation reads.
        // - outer has at least two limbs and inner is nonempty.
        // - both inputs are valid for their complete lengths and disjoint from dst.
        unsafe {
            ArchKernels::mul_basecase_unchecked(
                dst.as_mut_ptr().cast(),
                outer.as_ptr(),
                outer.len(),
                inner.as_ptr(),
                inner.len(),
            );
        }
    }

    /// Multiplies validated nonempty operands of the same width.
    ///
    /// The destination covers twice that width. Identical input slices retain
    /// the square kernel; callers that establish distinctness skip this check
    /// through [`Self::mul_equal_nonempty_distinct`].
    #[inline]
    pub fn mul_equal_nonempty(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb]) {
        debug_assert_eq!(a.len(), b.len(), "equal-width schoolbook operands");
        if eq(a.as_ptr(), b.as_ptr()) {
            Self::sqr_nonempty(dst, a);
        } else {
            Self::mul_equal_nonempty_distinct(dst, a, b);
        }
    }

    /// Selects a leaf for nonempty, distinct, equal-width operands.
    ///
    /// The parent establishes the shape and complete destination, so no empty,
    /// same-slice, operand-orientation, or equal-width checks remain at execution.
    #[inline]
    pub fn mul_equal_nonempty_distinct(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb]) {
        let len = a.len();
        debug_assert!(len != 0, "validated equal-width operands must be nonempty");
        debug_assert_eq!(len, b.len(), "equal-width schoolbook operands");
        debug_assert!(
            dst.len() >= len.saturating_mul(2),
            "equal-width destination must contain the complete product"
        );
        match len {
            1 => {
                // SAFETY: both inputs contain one limb and dst contains two;
                // valid Rust borrows keep the output disjoint from both inputs.
                unsafe {
                    Self::mul_limb_unchecked(
                        dst.as_mut_ptr().cast(),
                        a.as_ptr(),
                        1,
                        *b.get_unchecked(0),
                    );
                }
            }
            2 => Self::mul_fixed_equal_distinct::<2>(dst, a, b),
            3 => Self::mul_fixed_equal_distinct::<3>(dst, a, b),
            4 => Self::mul_fixed_equal_distinct::<4>(dst, a, b),
            5 => Self::mul_fixed_equal_distinct::<5>(dst, a, b),
            6 => Self::mul_fixed_equal_distinct::<6>(dst, a, b),
            8 => Self::mul_fixed_equal_distinct::<8>(dst, a, b),
            _ => {
                // SAFETY: nonempty inputs and the preceding cases prove len >= 2.
                // Both inputs contain len limbs and dst covers their complete
                // disjoint product. Architecture selection remains in arch/.
                unsafe {
                    ArchKernels::mul_basecase_unchecked(
                        dst.as_mut_ptr().cast(),
                        a.as_ptr(),
                        len,
                        b.as_ptr(),
                        len,
                    );
                }
            }
        }
    }

    /// Multiply two distinct fixed-width operands without dynamic width branches.
    ///
    /// Evaluates equal-width operands at a compile-time fixed length, bypassing
    /// dynamic length inspection and orientation checks.
    #[expect(
        clippy::inline_always,
        reason = "Fixed equal-width unrolling eliminates dynamic branch overhead for small limb counts"
    )]
    #[inline(always)]
    pub fn mul_fixed_equal_distinct<const LEN: usize>(
        dst: &mut [impl LimbOutput],
        a_limbs: &[Limb],
        b_limbs: &[Limb],
    ) {
        debug_assert_eq!(a_limbs.len(), LEN, "left operand has the wrong fixed width");
        debug_assert_eq!(
            b_limbs.len(),
            LEN,
            "right operand has the wrong fixed width"
        );
        debug_assert!(LEN >= 2, "fixed schoolbook width is below two limbs");
        debug_assert!(
            dst.len() >= LEN.saturating_mul(2),
            "fixed schoolbook destination is shorter than the product"
        );
        if LEN == 2 {
            // SAFETY: the caller proves two two-limb inputs and a four-limb destination.
            unsafe {
                ArchKernels::mul_2x2_portable_unchecked(
                    dst.as_mut_ptr().cast(),
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                );
            }
            return;
        }
        if LEN == 3 {
            // SAFETY: the caller proves two three-limb inputs and a six-limb
            // destination when this const specialization is selected.
            unsafe {
                ArchKernels::mul_3x3_portable_unchecked(
                    dst.as_mut_ptr().cast(),
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                );
            }
            return;
        }
        if LEN == 4 {
            // SAFETY: the caller proves two four-limb inputs and an eight-limb
            // destination when this const specialization is selected.
            unsafe {
                ArchKernels::mul_4x4_unchecked(
                    dst.as_mut_ptr().cast(),
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                );
            }
            return;
        }
        if LEN == 8 {
            // SAFETY: the caller proves two eight-limb inputs and a sixteen-limb
            // destination when this const specialization is selected.
            unsafe {
                ArchKernels::mul_8x8_unchecked(
                    dst.as_mut_ptr().cast(),
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                );
            }
            return;
        }

        // SAFETY: the caller proves two LEN-limb inputs and a 2*LEN-limb
        // destination; valid Rust borrows prove non-overlap.
        unsafe {
            ArchKernels::mul_basecase_unchecked(
                dst.as_mut_ptr().cast(),
                a_limbs.as_ptr(),
                LEN,
                b_limbs.as_ptr(),
                LEN,
            );
        }
    }

    /// Write one scalar product into an uninitialized destination window.
    ///
    /// # Safety
    ///
    /// `dst` must be writable for `len + 1` limbs and `src` readable for `len`
    /// limbs. The regions must either be disjoint or start at the same address;
    /// exact in-place operation is valid because limb `i` is read before it is
    /// overwritten and no later iteration reads it again.
    #[expect(
        clippy::inline_always,
        reason = "The complete scalar initializer adds only a final carry store to the shared prefix kernel"
    )]
    #[inline(always)]
    pub unsafe fn mul_limb_unchecked(dst: *mut Limb, src: *const Limb, len: usize, scalar: Limb) {
        // SAFETY: the caller reserves len+1 writable limbs and supplies the
        // initialized input. Exact aliasing consumes each input before overwrite.
        let carry = unsafe { Self::mul_limb_prefix_unchecked(dst, src, len, scalar) };
        // SAFETY: the final carry position is reserved by the caller.
        unsafe {
            *dst.add(len) = carry;
        }
    }

    /// Initializes the low `len` limbs of `a * b`.
    ///
    /// The first row initializes every output limb.
    /// Subsequent rows use `dst += a[i]*b*B^i (mod B^len)`; escaping carries
    /// are deliberately discarded by the truncated-product identity.
    ///
    /// # Safety
    ///
    /// `dst` is writable for `len` limbs and disjoint from both inputs, which
    /// each contain `len` initialized limbs and may overlap. `len` may be zero.
    #[expect(
        clippy::inline_always,
        reason = "Inlining the triangular initializer removes a separate zeroing pass and folds the first-row width into its validated callers"
    )]
    #[inline(always)]
    pub unsafe fn mullo_basecase_unchecked(
        dst: *mut Limb,
        a: *const Limb,
        b: *const Limb,
        len: usize,
    ) {
        if len == 0 {
            return;
        }
        // SAFETY: all spans cover len > 0 limbs, and the inputs are
        // disjoint from dst. This writes every limb before accumulation.
        let _ = unsafe { Self::mul_limb_prefix_unchecked(dst, b, len, *a) };
        // No accumulation row remains at width one, so selecting a backend
        // would perform a cache lookup whose result cannot be used.
        if len == 1 {
            return;
        }
        let mut index = 1_usize;
        let add_mul_limbs = ArchKernels::selected_add_mul_limbs_unchecked();
        while index < len {
            // SAFETY: index < len bounds the source limb and nonempty suffix;
            // every destination limb was initialized by the first row.
            unsafe {
                let limb = *a.add(index);
                let inner_len = len.unchecked_sub(index);
                let _ = add_mul_limbs(dst.add(index), b, inner_len, limb);
                index = index.unchecked_add(1);
            }
        }
    }

    /// Writes the low scalar-product prefix and returns its escaping limb.
    ///
    /// This is the common initialization invariant of complete scalar products
    /// and truncated first rows: no destination limb is read before it is written.
    ///
    /// # Safety
    ///
    /// Both spans cover `len` limbs and are disjoint or exactly aliased; only
    /// `src` must initially be initialized. Zero length accesses neither pointer.
    #[expect(
        clippy::as_conversions,
        reason = "Each wide product is split into a low limb and a carry strictly below the limb radix"
    )]
    #[cfg_attr(
        target_pointer_width = "32",
        expect(
            clippy::cast_possible_truncation,
            reason = "The low word truncates modulo B; the carry is proved below B"
        )
    )]
    #[inline]
    unsafe fn mul_limb_prefix_unchecked(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        scalar: Limb,
    ) -> Limb {
        let scalar_wide = scalar as DoubleLimb;
        let mut carry: DoubleLimb = 0;
        let mut index = 0_usize;
        while index < len {
            // SAFETY: the caller guarantees both spans for len limbs.
            let value = unsafe { *src.add(index) } as DoubleLimb;
            // SAFETY: with B = 2^LIMB_BITS, value and scalar are at most B-1.
            // Inductively carry <= B-2, so value*scalar+carry <= B^2-B-1.
            // DoubleLimb has at least 2*LIMB_BITS bits on 16/32/64-bit targets.
            let product = unsafe { value.unchecked_mul(scalar_wide).unchecked_add(carry) };
            // SAFETY: index < len, so this output position is within dst.
            unsafe {
                *dst.add(index) = product as Limb;
            }
            carry = product >> LIMB_BITS;
            // SAFETY: index < len and the caller supplies a valid non-ZST span,
            // so index+1 <= len <= isize::MAX/size_of::<Limb>() < usize::MAX.
            index = unsafe { index.unchecked_add(1) };
        }
        carry as Limb
    }
}
