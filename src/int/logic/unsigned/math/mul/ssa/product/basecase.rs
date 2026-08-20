//! Non-transform Fermat ring basecase multiplication, squaring, and reduction.

use core::sync::atomic::{AtomicUsize, Ordering};

use super::{
    ArchKernels, LIMB_BITS, Limb, MulPlan, Multiplication, NegacyclicPlan, SSA_BASE_MODULUS_BITS,
    SquarePlan, SsaCarry, SsaRing, TierCeiling,
};

/// Namespace for pointwise multiplication and basecase product reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaPointwise;

/// Memo table for [`SsaPointwise::fermat_basecase_scratch_len`], indexed by `mod_limbs`.
///
/// The value is a pure function of the ring width and of build-time thresholds,
/// but computing it walks the whole conventional tier tree, and that walk is
/// linear in the coefficient width: each Toom sizing step recurses into three
/// children of about a third the width. `FftPlan::from_geometry` asks for it
/// once per plan and plans nest, so without the memo a top-level multiplication
/// pays that walk once per transform level.
///
/// The array index *is* the key, so nothing can tear: a race merely has two
/// threads compute the same value and store the same result. Zero means "not
/// yet computed" and is unambiguous, because the only ring needing no scratch
/// is the empty one, which is not reachable here.
const BASECASE_MEMO_LEN: usize = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS).wrapping_add(1);
static BASECASE_SCRATCH_MEMO: [AtomicUsize; BASECASE_MEMO_LEN] =
    [const { AtomicUsize::new(0) }; BASECASE_MEMO_LEN];

impl SsaPointwise {
    /// Multiplies two ordinary Fermat residues with schoolbook multiplication and
    /// reduces the full product using `2^mod_bits = -1`.
    ///
    /// # Safety
    /// - `dst`, `left`, and `right` each have at least `SsaRing::coeff_limbs(mod_bits)` limbs.
    /// - `left` and `right` are canonical, nonzero, and are not the special residue
    ///   `2^mod_bits`.
    /// - `product_scratch` has at least [`Self::fermat_basecase_scratch_len`] limbs.
    pub unsafe fn fermat_basecase_mul_into(
        dst: &mut [Limb],
        left: &[Limb],
        right: &[Limb],
        mod_bits: usize,
        product_scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(mod_bits);
        let plan = Multiplication::select_plan(ml, ml, TierCeiling::Full);
        // SAFETY: this function's contract supplies the same fixed-width buffers
        // required by the selected-plan implementation below.
        unsafe {
            Self::fermat_basecase_mul_with_plan(dst, left, right, mod_bits, plan, product_scratch);
        }
    }

    /// Fixed-width Fermat product that overwrites its left operand after the lower
    /// multiplication tier has consumed it.
    ///
    /// # Safety
    /// - `left` and `right` are disjoint complete coefficients.
    /// - Both coefficients are canonical.
    /// - `plan` and `product_scratch` satisfy the same fixed-width contract as
    ///   [`Self::fermat_basecase_mul_with_plan`].
    #[allow(
        clippy::inline_always,
        reason = "the in-place pointwise product must inline through the coefficient loop"
    )]
    #[inline(always)]
    pub unsafe fn fermat_basecase_mul_assign_left(
        left: &mut [Limb],
        right: &[Limb],
        mod_bits: usize,
        plan: MulPlan,
        product_scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(mod_bits);
        if ml == 1 {
            // SAFETY: ml == 1, left and right each have at least 2 limbs (cl = 2).
            let (lo, hi) = unsafe {
                ArchKernels::mul_limb_lo_hi(*left.get_unchecked(0), *right.get_unchecked(0))
            };
            let (res, borrow) = lo.overflowing_sub(hi);
            // SAFETY: left has at least 2 limbs.
            unsafe {
                *left.get_unchecked_mut(0) = res;
                *left.get_unchecked_mut(1) = 0;
            }
            if borrow {
                let (c_res, c_carry) = res.overflowing_add(1);
                // SAFETY: left has at least 2 limbs.
                unsafe {
                    *left.get_unchecked_mut(0) = c_res;
                    if c_carry {
                        *left.get_unchecked_mut(1) = 1;
                    }
                }
            }
            return;
        }
        // The selected plan and its scratch layout prove this span is representable.
        let product_span = ml.wrapping_mul(2);
        debug_assert!(
            product_span != usize::MAX,
            "validated product span overflowed"
        );
        // SAFETY: the caller guarantees the ordinary basecase scratch length.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // The selected multiplication writes a disjoint product and returns before
        // reduction touches `left`. The immutable reborrow below therefore ends
        // before the in-place output overwrite begins.
        // SAFETY: product, left, and right have the exact fixed widths established
        // by the caller; tower_scratch was sized for this selected plan.
        unsafe {
            Multiplication::execute_plan(
                plan,
                product.get_unchecked_mut(..product_span),
                left.get_unchecked(..ml),
                right.get_unchecked(..ml),
                tower_scratch,
            );
        }
        // SAFETY: the tower initialized the full product. Its buffer is disjoint
        // from the complete left coefficient that now becomes the destination.
        unsafe {
            Self::reduce_full_product(left, product, ml);
        }
    }

    /// Fixed-width Fermat product with a lower-tower plan selected by the caller.
    ///
    /// # Safety
    /// The buffer and residue preconditions are the same as
    /// [`Self::fermat_basecase_mul_into`], and `plan` was selected for two `ml`-limb
    /// operands where `ml = SsaRing::mod_limbs(mod_bits)`.
    #[allow(
        clippy::inline_always,
        reason = "benchmarking shows inlining is required to hoist lower-tier dispatch"
    )]
    #[inline(always)]
    pub unsafe fn fermat_basecase_mul_with_plan(
        dst: &mut [Limb],
        left: &[Limb],
        right: &[Limb],
        mod_bits: usize,
        plan: MulPlan,
        product_scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(mod_bits);
        // The selected plan and its scratch layout prove this span is representable.
        let product_span = ml.wrapping_mul(2);
        debug_assert!(
            product_span != usize::MAX,
            "validated product span overflowed"
        );
        // SAFETY: the caller guarantees product_scratch has the basecase scratch
        // length, whose first product_span limbs hold the complete product.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // FFT coefficients occupy fixed-width slots and become dense after the
        // first butterfly. Multiplying the complete data spans avoids two backward
        // active-length scans, a second tier selection for scratch sizing, and any
        // product clearing: every lower multiplication tier overwrites the exact
        // `2 * ml`-limb result. Leading zero limbs remain algebraically harmless.
        // SAFETY: all three ranges have their exact fixed widths by construction;
        // tower_scratch was sized for this full-width product.
        let product_output = unsafe { product.get_unchecked_mut(..product_span) };
        // SAFETY: the caller guarantees `left` contains the complete data span.
        let left_input = unsafe { left.get_unchecked(..ml) };
        // SAFETY: the caller guarantees `right` contains the complete data span.
        let right_input = unsafe { right.get_unchecked(..ml) };
        Multiplication::execute_plan(plan, product_output, left_input, right_input, tower_scratch);
        // SAFETY: the multiplication overwrote the complete 2*ml-limb product and
        // dst is a disjoint complete coefficient.
        unsafe {
            Self::reduce_full_product(dst, product, ml);
        }
    }

    /// Fixed-width Fermat square, `dst = value^2 mod (2^mod_bits + 1)`.
    ///
    /// The squaring counterpart of [`Self::fermat_basecase_mul_into`]. Calling that with
    /// `value` for both operands selects a *product* plan and drives
    /// `Multiplication::execute_plan`, so the aliasing is never noticed and the pointwise stage
    /// of a squaring transform paid full product price at every coefficient.
    ///
    /// # Safety
    /// The buffer and residue preconditions are those of
    /// [`Self::fermat_basecase_mul_into`], with one operand.
    pub unsafe fn fermat_basecase_sqr_into(
        dst: &mut [Limb],
        value: &[Limb],
        mod_bits: usize,
        product_scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(mod_bits);
        if ml == 1 {
            // SAFETY: ml == 1, value has at least 1 limb and dst has at least 2 limbs.
            let val = unsafe { *value.get_unchecked(0) };
            let (lo, hi) = ArchKernels::mul_limb_lo_hi(val, val);
            let (res, borrow) = lo.overflowing_sub(hi);
            // SAFETY: dst has at least 2 limbs.
            unsafe {
                *dst.get_unchecked_mut(0) = res;
                *dst.get_unchecked_mut(1) = 0;
            }
            if borrow {
                let (c_res, c_carry) = res.overflowing_add(1);
                // SAFETY: dst has at least 2 limbs.
                unsafe {
                    *dst.get_unchecked_mut(0) = c_res;
                    if c_carry {
                        *dst.get_unchecked_mut(1) = 1;
                    }
                }
            }
            return;
        }
        let product_span = ml.wrapping_mul(2);
        debug_assert!(
            product_span != usize::MAX,
            "validated product span overflowed"
        );
        // SAFETY: product_scratch was sized by `fermat_basecase_scratch_len`, which
        // allocates the full 2*ml-limb product prefix for either operation.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // The squaring tower takes its scratch from the common basecase layout;
        // `uncached_fermat_basecase_scratch_len` sizes the tail for whichever of
        // the two towers is larger.
        // SAFETY: product covers product_span limbs by split above.
        let product_output = unsafe { product.get_unchecked_mut(..product_span) };
        // SAFETY: the caller guarantees `value` contains the complete data span.
        let value_input = unsafe { value.get_unchecked(..ml) };
        Multiplication::execute_square_plan(
            Multiplication::select_square_plan(ml, TierCeiling::Full),
            product_output,
            value_input,
            tower_scratch,
        );
        // SAFETY: the square overwrote the complete 2*ml-limb product and dst is a
        // disjoint complete coefficient.
        unsafe {
            Self::reduce_full_product(dst, product, ml);
        }
    }

    /// Fixed-width Fermat square that overwrites its operand in place.
    ///
    /// # Safety
    /// - `val` is a canonical complete coefficient with at least `SsaRing::coeff_limbs(mod_bits)` limbs.
    /// - `product_scratch` has at least `Self::fermat_basecase_scratch_len(mod_bits)` limbs.
    #[allow(
        clippy::inline_always,
        reason = "the in-place pointwise square must inline through the coefficient loop"
    )]
    #[inline(always)]
    pub unsafe fn fermat_basecase_sqr_assign(
        val: &mut [Limb],
        mod_bits: usize,
        plan: SquarePlan,
        product_scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(mod_bits);
        if ml == 1 {
            // SAFETY: ml == 1, val has at least 2 limbs (cl = 2).
            let x = unsafe { *val.get_unchecked(0) };
            let (lo, hi) = ArchKernels::mul_limb_lo_hi(x, x);
            let (res, borrow) = lo.overflowing_sub(hi);
            // SAFETY: val has at least 2 limbs.
            unsafe {
                *val.get_unchecked_mut(0) = res;
                *val.get_unchecked_mut(1) = 0;
            }
            if borrow {
                let (c_res, c_carry) = res.overflowing_add(1);
                // SAFETY: val has at least 2 limbs.
                unsafe {
                    *val.get_unchecked_mut(0) = c_res;
                    if c_carry {
                        *val.get_unchecked_mut(1) = 1;
                    }
                }
            }
            return;
        }
        let product_span = ml.wrapping_mul(2);
        debug_assert!(
            product_span != usize::MAX,
            "validated product span overflowed"
        );
        // SAFETY: the caller guarantees product_scratch has the basecase scratch length.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // SAFETY: product covers product_span limbs by split above.
        let product_output = unsafe { product.get_unchecked_mut(..product_span) };
        // SAFETY: the caller guarantees val contains at least cl > ml limbs.
        let value_input = unsafe { val.get_unchecked(..ml) };
        Multiplication::execute_square_plan(plan, product_output, value_input, tower_scratch);
        // SAFETY: the square overwrote the complete 2*ml product, disjoint from val.
        unsafe {
            Self::reduce_full_product(val, product, ml);
        }
    }

    /// Reduces a fixed-width `2 * ml` product modulo `2^(ml * LIMB_BITS) + 1`.
    ///
    /// # Safety
    /// - `dst` contains at least `ml + 1` writable limbs.
    /// - `product` contains at least `2 * ml` initialized limbs and is disjoint
    ///   from `dst`.
    #[allow(
        clippy::inline_always,
        reason = "shared pointwise reduction must inline after the lower multiplication tier"
    )]
    #[inline(always)]
    pub unsafe fn reduce_full_product(dst: &mut [Limb], product: &[Limb], ml: usize) {
        // The low `ml` destination limbs are overwritten below. The guard is the
        // only destination word whose zero value must be established up front; a
        // borrow correction may replace it with the canonical `2^n` residue.
        // SAFETY: ml < cl and caller guarantees dst has at least cl limbs.
        unsafe {
            *dst.get_unchecked_mut(ml) = 0;
        }
        // Reduce with one three-operand subtraction pass instead of copying the low
        // half and then subtracting the high half in place. The architecture layer
        // selects the native kernel while every target retains the same fixed-width
        // `low - high` proof.
        // SAFETY: dst covers ml writable limbs; product covers two disjoint ml-limb
        // halves, and all three pointers are valid for exactly ml elements.
        let borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                dst.as_mut_ptr(),
                product.as_ptr(),
                product.as_ptr().add(ml),
                ml,
            )
        };
        if borrow != 0 {
            // SAFETY: the low-minus-high subtraction borrowed exactly once and the
            // caller guarantees dst has cl > ml limbs.
            unsafe {
                SsaCarry::correct_wrapped_shift_difference(dst, ml);
            }
        }
    }

    /// Scratch required for one non-recursive Fermat-ring point product.
    pub fn fermat_basecase_scratch_len(mod_bits: usize) -> usize {
        let ml = SsaRing::mod_limbs(mod_bits);
        let Some(slot) = BASECASE_SCRATCH_MEMO.get(ml) else {
            // Above the basecase width the transform runs instead, so this is only
            // reachable from a caller asking about a ring it will not use.
            return uncached_fermat_basecase_scratch_len(ml);
        };
        let cached = slot.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }
        let len = uncached_fermat_basecase_scratch_len(ml);
        slot.store(len, Ordering::Relaxed);
        len
    }
}

fn uncached_fermat_basecase_scratch_len(ml: usize) -> usize {
    if ml == 0 {
        return 0;
    }
    let product_len = ml.wrapping_mul(2);
    let mul_plan = Multiplication::select_plan(ml, ml, TierCeiling::Full);
    let sqr_plan = Multiplication::select_square_plan(ml, TierCeiling::Full);
    let tower_scratch = Multiplication::scratch_len(mul_plan, ml, ml)
        .max(Multiplication::square_scratch_len(sqr_plan, ml));
    let base_scratch = product_len.wrapping_add(tower_scratch);
    NegacyclicPlan::new(ml).map_or(base_scratch, |plan| base_scratch.max(plan.scratch_len()))
}
