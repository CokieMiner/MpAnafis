//! Non-transform Fermat ring basecase multiplication, squaring, and reduction.

#![expect(
    unsafe_code,
    reason = "Complete canonical coefficients and sized disjoint product arenas bound scalar specialization and Fermat reduction"
)]

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::parallel::SequentialExecutor;

use super::{
    ArchKernels, LIMB_BITS, Limb, LimbOutput, MulPlan, Multiplication, NegacyclicPlan,
    SSA_BASE_MODULUS_BITS, SquarePlan, SsaCarry, SsaRing, TierCeiling,
};

/// Namespace for pointwise multiplication and basecase product reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaPointwise;

/// Memo table for [`SsaPointwise::fermat_basecase_scratch_len`], indexed by `mod_limbs`.
///
/// Width and build-time thresholds determine each value. The index is the
/// complete key; racing stores publish identical sizing results. Zero marks
/// an uncomputed entry because every admitted ring needs positive scratch.
const BASECASE_MEMO_LEN: usize = SSA_BASE_MODULUS_BITS.div_euclid(LIMB_BITS) + 1;
static BASECASE_SCRATCH_MEMO: [AtomicUsize; BASECASE_MEMO_LEN] =
    [const { AtomicUsize::new(0) }; BASECASE_MEMO_LEN];

impl SsaPointwise {
    /// Multiplies two ordinary Fermat residues with the lower multiplication tower and
    /// reduces the full product using `2^mod_bits = -1`.
    ///
    /// # Safety
    /// - `dst`, `left`, and `right` each have at least `SsaRing::coeff_limbs(mod_bits)` limbs.
    /// - `left` and `right` are canonical, nonzero, and are not the special residue
    ///   `2^mod_bits`.
    /// - `product_scratch` has at least [`Self::fermat_basecase_scratch_len`] limbs.
    pub unsafe fn fermat_basecase_mul_into(
        dst: &mut [impl LimbOutput],
        left: &[Limb],
        right: &[Limb],
        mod_bits: usize,
        product_scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(mod_bits);
        if ml == 1 {
            // SAFETY: ml == 1, left and right have at least 1 limb.
            let (r0, r1) =
                unsafe { Self::mul_1x1(*left.get_unchecked(0), *right.get_unchecked(0)) };
            // SAFETY: dst has at least cl = 2 limbs.
            unsafe {
                *dst.get_unchecked_mut(0) = LimbOutput::from_limb(r0);
                *dst.get_unchecked_mut(1) = LimbOutput::from_limb(r1);
            }
            return;
        }
        if ml == 2 {
            // SAFETY: ml == 2, left and right have at least 2 limbs.
            let product = unsafe { Self::mul_2x2(left, right) };
            // SAFETY: dst covers at least cl = 3 limbs, product has 4 limbs.
            unsafe {
                Self::reduce_full_product(dst, &product, 2);
            }
            return;
        }
        let plan = Multiplication::select_plan(ml, ml, TierCeiling::Full);
        // SAFETY: the complete product prefix belongs to the caller's checked
        // basecase arena; independently ml=mod_bits/LIMB_BITS and LIMB_BITS>=16.
        let product_span = unsafe { ml.unchecked_mul(2) };
        // SAFETY: the caller guarantees product_scratch has the basecase scratch
        // length, whose first product_span limbs hold the complete product.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // FFT coefficients occupy fixed-width slots and become dense after the
        // first butterfly. Multiplying the complete data spans avoids two backward
        // active-length scans, a second tier selection for scratch sizing, and any
        // product clearing: every lower multiplication tier overwrites the exact
        // `2 * ml`-limb result. Leading zero limbs remain algebraically harmless.
        // SAFETY: the caller guarantees `left` contains the complete data span.
        let left_input = unsafe { left.get_unchecked(..ml) };
        // SAFETY: the caller guarantees `right` contains the complete data span.
        let right_input = unsafe { right.get_unchecked(..ml) };
        Multiplication::execute_plan_with_executor(
            plan,
            product,
            left_input,
            right_input,
            tower_scratch,
            &SequentialExecutor,
        );
        // SAFETY: the multiplication overwrote the complete 2*ml-limb product and
        // dst is a disjoint complete coefficient.
        unsafe {
            Self::reduce_full_product(dst, product, ml);
        }
    }

    /// Fixed-width Fermat product that overwrites its left operand after the lower
    /// multiplication tier has consumed it.
    ///
    /// # Safety
    /// - `left` and `right` are disjoint complete coefficients.
    /// - Both coefficients are canonical.
    /// - `plan` and `product_scratch` satisfy the same fixed-width contract as
    ///   [`Self::fermat_basecase_mul_into`].
    #[expect(
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
            // SAFETY: ml == 1, left and right have at least 1 limb.
            let (r0, r1) =
                unsafe { Self::mul_1x1(*left.get_unchecked(0), *right.get_unchecked(0)) };
            // SAFETY: left has at least cl = 2 limbs.
            unsafe {
                *left.get_unchecked_mut(0) = r0;
                *left.get_unchecked_mut(1) = r1;
            }
            return;
        }
        if ml == 2 {
            // SAFETY: ml == 2, left and right have at least 2 limbs.
            let product = unsafe { Self::mul_2x2(left, right) };
            // SAFETY: left covers at least cl = 3 limbs, product has 4 limbs.
            unsafe {
                Self::reduce_full_product(left, &product, 2);
            }
            return;
        }
        // SAFETY: the caller's checked arena reserves the complete 2*ml product;
        // ml=mod_bits/LIMB_BITS also bounds this multiplication independently.
        let product_span = unsafe { ml.unchecked_mul(2) };
        // SAFETY: the caller guarantees the ordinary basecase scratch length.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // The selected multiplication writes a disjoint product and returns before
        // reduction touches `left`. The immutable reborrow below therefore ends
        // before the in-place output overwrite begins.
        // SAFETY: product, left, and right have the exact fixed widths established
        // by the caller; tower_scratch was sized for this selected plan.
        unsafe {
            Multiplication::execute_plan_with_executor(
                plan,
                product,
                left.get_unchecked(..ml),
                right.get_unchecked(..ml),
                tower_scratch,
                &SequentialExecutor,
            );
        }
        // SAFETY: the tower initialized the full product. Its buffer is disjoint
        // from the complete left coefficient that now becomes the destination.
        unsafe {
            Self::reduce_full_product(left, product, ml);
        }
    }

    /// Fixed-width Fermat square, `dst = value^2 mod (2^mod_bits + 1)`.
    ///
    /// Uses the square tower and one full-width square before Fermat reduction.
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
            // SAFETY: ml == 1, value has at least 1 limb.
            let scalar = unsafe { *value.get_unchecked(0) };
            let (r0, r1) = Self::mul_1x1(scalar, scalar);
            // SAFETY: dst has at least cl = 2 limbs.
            unsafe {
                *dst.get_unchecked_mut(0) = r0;
                *dst.get_unchecked_mut(1) = r1;
            }
            return;
        }
        if ml == 2 {
            // SAFETY: ml == 2, value has at least 2 limbs.
            let product = unsafe { Self::sqr_2x2(value) };
            // SAFETY: dst covers at least cl = 3 limbs, product has 4 limbs.
            unsafe {
                Self::reduce_full_product(dst, &product, 2);
            }
            return;
        }
        // SAFETY: the caller's checked arena reserves the complete 2*ml square;
        // ml=mod_bits/LIMB_BITS also bounds this multiplication independently.
        let product_span = unsafe { ml.unchecked_mul(2) };
        // SAFETY: product_scratch was sized by `fermat_basecase_scratch_len`, which
        // allocates the full 2*ml-limb product prefix for either operation.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // The squaring tower takes its scratch from the common basecase layout;
        // `uncached_fermat_basecase_scratch_len` sizes the tail for whichever of
        // the two towers is larger.
        // SAFETY: the caller guarantees `value` contains the complete data span.
        let value_input = unsafe { value.get_unchecked(..ml) };
        Multiplication::execute_square_plan_with_executor(
            Multiplication::select_square_plan(ml, TierCeiling::Full),
            product,
            value_input,
            tower_scratch,
            &SequentialExecutor,
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
    #[expect(
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
            // SAFETY: ml == 1, val has at least 1 limb.
            let scalar = unsafe { *val.get_unchecked(0) };
            let (r0, r1) = Self::mul_1x1(scalar, scalar);
            // SAFETY: val has at least cl = 2 limbs.
            unsafe {
                *val.get_unchecked_mut(0) = r0;
                *val.get_unchecked_mut(1) = r1;
            }
            return;
        }
        if ml == 2 {
            // SAFETY: ml == 2, val has at least 2 limbs.
            let product = unsafe { Self::sqr_2x2(val) };
            // SAFETY: val covers at least cl = 3 limbs, product has 4 limbs.
            unsafe {
                Self::reduce_full_product(val, &product, 2);
            }
            return;
        }
        // SAFETY: the caller's checked arena reserves the complete 2*ml square;
        // ml=mod_bits/LIMB_BITS also bounds this multiplication independently.
        let product_span = unsafe { ml.unchecked_mul(2) };
        // SAFETY: the caller guarantees product_scratch has the basecase scratch length.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // SAFETY: the caller guarantees val contains at least cl > ml limbs.
        let value_input = unsafe { val.get_unchecked(..ml) };
        Multiplication::execute_square_plan_with_executor(
            plan,
            product,
            value_input,
            tower_scratch,
            &SequentialExecutor,
        );
        // SAFETY: the square overwrote the complete 2*ml product, disjoint from val.
        unsafe {
            Self::reduce_full_product(val, product, ml);
        }
    }

    #[inline]
    fn mul_1x1(a: Limb, b: Limb) -> (Limb, Limb) {
        let (lo, hi) = ArchKernels::mul_limb_lo_hi(a, b);
        let (res, borrow) = lo.overflowing_sub(hi);
        // B == -1 modulo B+1. A borrowed difference already includes B;
        // adding its borrow supplies the remaining +1. An escaping carry
        // denotes B canonically and can occur only when that borrow is one.
        let (corrected, carry) = res.overflowing_add(Limb::from(borrow));
        (corrected, Limb::from(carry))
    }

    #[inline]
    unsafe fn mul_2x2(a: &[Limb], b: &[Limb]) -> [Limb; 4] {
        // SAFETY: caller guarantees a has at least 2 limbs.
        let a0 = unsafe { *a.get_unchecked(0) };
        // SAFETY: caller guarantees a has at least 2 limbs.
        let a1 = unsafe { *a.get_unchecked(1) };
        // SAFETY: caller guarantees b has at least 2 limbs.
        let b0 = unsafe { *b.get_unchecked(0) };
        // SAFETY: caller guarantees b has at least 2 limbs.
        let b1 = unsafe { *b.get_unchecked(1) };

        let (p0, c0) = ArchKernels::mul_limb_lo_hi(a0, b0);
        let (m0, c1) = ArchKernels::mul_limb_lo_hi(a0, b1);
        let (m1, c2) = ArchKernels::mul_limb_lo_hi(a1, b0);
        let (p2_base, p3_base) = ArchKernels::mul_limb_lo_hi(a1, b1);

        let (s0, k0) = m0.overflowing_add(c0);
        let (p1, k1) = s0.overflowing_add(m1);

        let (c_mid, k2) = c1.overflowing_add(c2);
        let (c_mid2, k3) = c_mid.overflowing_add(Limb::from(k0));
        let (c_mid3, k4) = c_mid2.overflowing_add(Limb::from(k1));
        // SAFETY: three one-bit carries sum to at most three.
        let carry_hi = unsafe {
            Limb::from(k2)
                .unchecked_add(Limb::from(k3))
                .unchecked_add(Limb::from(k4))
        };

        let (p2, k5) = p2_base.overflowing_add(c_mid3);
        // SAFETY: these nonnegative terms are the high digit of the exact
        // product a*b<B^4. Every partial sum is therefore below B.
        let p3 = unsafe {
            p3_base
                .unchecked_add(carry_hi)
                .unchecked_add(Limb::from(k5))
        };

        [p0, p1, p2, p3]
    }

    #[inline]
    unsafe fn sqr_2x2(a: &[Limb]) -> [Limb; 4] {
        // SAFETY: caller guarantees a has at least 2 limbs.
        let a0 = unsafe { *a.get_unchecked(0) };
        // SAFETY: caller guarantees a has at least 2 limbs.
        let a1 = unsafe { *a.get_unchecked(1) };

        let (p0, c0) = ArchKernels::mul_limb_lo_hi(a0, a0);
        let (m0, c1) = ArchKernels::mul_limb_lo_hi(a0, a1);
        let (p2_base, p3_base) = ArchKernels::mul_limb_lo_hi(a1, a1);

        let (s0, k0) = m0.overflowing_add(c0);
        let (p1, k1) = s0.overflowing_add(m0);

        let c1_hi = c1 >> (LIMB_BITS - 1);
        let c1_lo = c1 << 1;
        // c1_lo is even and at most B-2, while k0 is binary. Their digit
        // fields are disjoint, so their sum is an exact OR with no extra carry.
        let carry_limb1 = c1_lo | Limb::from(k0);
        let (carry_limb2, carry_extra2) = carry_limb1.overflowing_add(Limb::from(k1));

        let (p2, k2) = p2_base.overflowing_add(carry_limb2);
        // SAFETY: the terms are nonnegative and sum to the high digit of
        // a^2<B^4, so neither their final nor their partial sums reach B.
        let p3 = unsafe {
            p3_base
                .unchecked_add(c1_hi)
                .unchecked_add(Limb::from(carry_extra2))
                .unchecked_add(Limb::from(k2))
        };

        [p0, p1, p2, p3]
    }

    /// Reduces a fixed-width `2 * ml` product modulo `2^(ml * LIMB_BITS) + 1`.
    ///
    /// # Safety
    /// - `dst` contains at least `ml + 1` writable limbs.
    /// - `product` contains at least `2 * ml` initialized limbs and is disjoint
    ///   from `dst`.
    #[expect(
        clippy::inline_always,
        reason = "shared pointwise reduction must inline after the lower multiplication tier"
    )]
    #[inline(always)]
    pub unsafe fn reduce_full_product(dst: &mut [impl LimbOutput], product: &[Limb], ml: usize) {
        // Reduce with one three-operand subtraction pass instead of copying the low
        // half and then subtracting the high half in place. The architecture layer
        // selects the native kernel while every target retains the same fixed-width
        // `low - high` proof.
        // SAFETY: dst covers ml writable limbs; product covers two disjoint ml-limb
        // halves, and all three pointers are valid for exactly ml elements.
        let borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                dst.as_mut_ptr().cast(),
                product.as_ptr(),
                product.as_ptr().add(ml),
                ml,
            )
        };
        if borrow != 0 {
            // SAFETY: the low-minus-high subtraction borrowed exactly once and the
            // caller guarantees dst has cl > ml limbs.
            unsafe {
                let _ = SsaCarry::correct_wrapped_shift_difference(dst, ml);
            }
        } else {
            // SAFETY: dst has ml data limbs and a guard; reduction consumes
            // only the data prefix, so this is the sole guard write.
            unsafe {
                *dst.get_unchecked_mut(ml) = LimbOutput::from_limb(0);
            }
        }
    }

    /// Scratch required for one non-recursive Fermat-ring point product.
    pub fn fermat_basecase_scratch_len(mod_bits: usize) -> usize {
        let ml = SsaRing::mod_limbs(mod_bits);
        let Some(slot) = BASECASE_SCRATCH_MEMO.get(ml) else {
            // Above the basecase width the transform runs instead, so this is only
            // reachable from a caller asking about a ring it will not use.
            return Self::uncached_fermat_basecase_scratch_len(ml);
        };
        let cached = slot.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }
        let len = Self::uncached_fermat_basecase_scratch_len(ml);
        slot.store(len, Ordering::Relaxed);
        len
    }

    fn uncached_fermat_basecase_scratch_len(ml: usize) -> usize {
        if ml == 0 {
            return 0;
        }
        // SAFETY: the sole caller derives ml=mod_bits/LIMB_BITS with LIMB_BITS>=16.
        let product_len = unsafe { ml.unchecked_mul(2) };
        let mul_plan = Multiplication::select_plan(ml, ml, TierCeiling::Full);
        let sqr_plan = Multiplication::select_square_plan(ml, TierCeiling::Full);
        let tower_scratch = Multiplication::scratch_len(mul_plan, ml, ml)
            .max(Multiplication::square_scratch_len(sqr_plan, ml));
        // Saturation preserves the sizing sentinel for an unrepresentable arena.
        let base_scratch = product_len.saturating_add(tower_scratch);
        NegacyclicPlan::select_factor(ml)
            .and_then(|factor| NegacyclicPlan::for_factor(ml, factor))
            .map_or(base_scratch, |plan| base_scratch.max(plan.scratch_len))
    }
}
