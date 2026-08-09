//! Pointwise multiplication and basecase product reduction for SSA.

use core::sync::atomic::{AtomicUsize, Ordering};

use super::{
    ArchKernels, LIMB_BITS, Limb, MulPlan, Multiplication, NegacyclicPlan, Residue,
    SSA_BASE_MODULUS_BITS, SsaCarry, SsaRing, SsaTransform, TierCeiling,
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
    /// Writes the product when either operand is a special residue, and reports
    /// whether it did.
    ///
    /// Zero absorbs and `-1` negates, so both cases are a fill or a negated copy.
    /// Neither needs a transform, and both are reachable at every level of the
    /// recursion, so this guard sits in front of the basecase as well as in front
    /// of the transform.
    ///
    /// # Safety
    /// `dst`, `left`, and `right` each span at least `SsaRing::mod_limbs(mod_bits) + 1`
    /// limbs, and `dst` is disjoint from both operands.
    pub unsafe fn write_special_residue_product(
        dst: &mut [Limb],
        left: &[Limb],
        right: &[Limb],
        mod_bits: usize,
    ) -> bool {
        let ml = SsaRing::mod_limbs(mod_bits);
        let cl = ml.wrapping_add(1);
        // SAFETY: ml < cl and the caller guarantees both spans have cl limbs.
        let left_class = unsafe { SsaRing::classify_residue(left, ml) };
        // SAFETY: same bounds proof as left.
        let right_class = unsafe { SsaRing::classify_residue(right, ml) };

        if left_class == Residue::Zero || right_class == Residue::Zero {
            // SAFETY: the caller guarantees dst spans cl limbs. A slice fill
            // lowers to one memset; the iterator form did not.
            unsafe { dst.get_unchecked_mut(..cl) }.fill(0);
            return true;
        }
        if left_class != Residue::NegOne && right_class != Residue::NegOne {
            return false;
        }
        // Multiplying by -1 is a negated copy of the other operand.
        let source = if left_class == Residue::NegOne {
            right
        } else {
            left
        };
        // SAFETY: both spans are complete, disjoint cl-limb coefficients.
        unsafe { dst.get_unchecked_mut(..cl) }
            .copy_from_slice(unsafe { source.get_unchecked(..cl) });
        // SAFETY: dst has cl limbs and mod_bits matches.
        unsafe {
            SsaRing::negate(dst, mod_bits);
        }
        true
    }

    /// The squaring counterpart: `0^2 = 0` and `(-1)^2 = 1`.
    ///
    /// # Safety
    /// `dst` spans at least `cl` limbs and `cl` is non-zero.
    pub unsafe fn write_special_residue_square(
        dst: &mut [Limb],
        cl: usize,
        class: Residue,
    ) -> bool {
        if class == Residue::Ordinary {
            return false;
        }
        // SAFETY: the caller guarantees dst spans cl limbs.
        unsafe { dst.get_unchecked_mut(..cl) }.fill(0);
        if class == Residue::NegOne {
            // SAFETY: cl is non-zero, so index 0 is in bounds.
            unsafe {
                *dst.get_unchecked_mut(0) = 1;
            }
        }
        true
    }

    /// Multiplies every coefficient pair in two Fermat-ring matrices.
    ///
    /// Normalizes coefficients in-place, multiplies into a shared result buffer,
    /// and writes results back into `left_matrix`. `right_matrix` is mutated
    /// during normalization and is dead after this call.
    ///
    /// # Safety
    /// - `left_matrix` and `right_matrix` each have `transform_len * cl` limbs.
    /// - `product_scratch` has at least `cl` limbs for result staging plus recursive
    ///   scratch as determined by the tier dispatcher.
    pub unsafe fn pointwise_multiply(
        left_matrix: &mut [Limb],
        right_matrix: &mut [Limb],
        transform_len: usize,
        mod_bits: usize,
        product_scratch: &mut [Limb],
    ) {
        let cl = SsaRing::coeff_limbs(mod_bits);
        let basecase_plan = (mod_bits <= SSA_BASE_MODULUS_BITS).then(|| {
            let ml = SsaRing::mod_limbs(mod_bits);
            Multiplication::select_plan(ml, ml, TierCeiling::Full)
        });
        let negacyclic_plan = (mod_bits <= SSA_BASE_MODULUS_BITS)
            .then(|| NegacyclicPlan::new(SsaRing::mod_limbs(mod_bits)))
            .flatten();

        let (result, mul_scratch) = product_scratch.split_at_mut(cl);

        for i in 0..transform_len {
            let offset = i.wrapping_mul(cl);

            // SAFETY: offset + cl <= matrix length by construction; both matrices
            // are disjoint and each holds transform_len complete coefficients.
            let left = unsafe { left_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };
            // SAFETY: same bounds proof as left, applied to the right matrix.
            let right = unsafe { right_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };

            // SAFETY: both spans are complete cl-limb coefficients.
            unsafe {
                SsaRing::normalize(left, mod_bits);
                SsaRing::normalize(right, mod_bits);
                if write_pointwise_special_product(left, right, mod_bits) {
                    continue;
                }
                if let Some(plan) = negacyclic_plan {
                    plan.mul_assign_left(left, right, mul_scratch);
                } else if let Some(plan) = basecase_plan {
                    fermat_basecase_mul_assign_left(left, right, mod_bits, plan, mul_scratch);
                } else {
                    fermat_mul_into(result, left, right, mod_bits, None, mul_scratch);
                    left.copy_from_slice(result);
                }
            }
        }
    }

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
            fermat_basecase_mul_with_plan(dst, left, right, mod_bits, plan, product_scratch);
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
        let product_span = ml.wrapping_mul(2);
        // SAFETY: the caller guarantees product_scratch has the basecase scratch
        // length, whose first product_span limbs hold the complete product.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // SAFETY: both ranges have their exact fixed widths by construction, and
        // `uncached_fermat_basecase_scratch_len` sizes the tail for whichever of
        // the two towers is larger.
        Multiplication::execute_square_plan(
            Multiplication::select_square_plan(ml, TierCeiling::Full),
            unsafe { product.get_unchecked_mut(..product_span) },
            unsafe { value.get_unchecked(..ml) },
            tower_scratch,
        );
        // SAFETY: the square overwrote the complete 2*ml-limb product and dst is a
        // disjoint complete coefficient.
        unsafe {
            Self::reduce_full_product(dst, product, ml);
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
        let computed = uncached_fermat_basecase_scratch_len(ml);
        slot.store(computed, Ordering::Relaxed);
        computed
    }
}

fn uncached_fermat_basecase_scratch_len(ml: usize) -> usize {
    // One buffer serves both the product and the square, because a Fermat
    // coefficient reaches whichever of the two the caller asked for and the
    // two towers size their workspaces independently.
    let tower =
        Multiplication::required_scratch(ml, ml).max(Multiplication::required_sqr_scratch(ml));
    let exact = ml.wrapping_mul(2).wrapping_add(tower);
    NegacyclicPlan::new(ml).map_or(exact, |plan| exact.max(plan.scratch_len()))
}

/// Handles the sole pointwise special representation that cannot flow through
/// an ordinary fixed-width multiplication.
///
/// Canonical zero has a zero guard and needs no branch: every multiplication
/// tier naturally writes a zero product. Canonical `-1` is the unique value
/// with guard one, so it is identified without scanning the data limbs.
///
/// # Safety
/// `left` and `right` are disjoint, canonical `SsaRing::coeff_limbs(mod_bits)` spans.
#[allow(
    clippy::inline_always,
    reason = "one guard check per coefficient replaces two full-width classification scans"
)]
#[inline(always)]
unsafe fn write_pointwise_special_product(
    left: &mut [Limb],
    right: &[Limb],
    mod_bits: usize,
) -> bool {
    let ml = SsaRing::mod_limbs(mod_bits);
    // SAFETY: both complete coefficients include the guard at ml.
    let left_guard = unsafe { *left.get_unchecked(ml) };
    // SAFETY: same coefficient-width proof as left.
    let right_guard = unsafe { *right.get_unchecked(ml) };
    debug_assert!(
        left_guard <= 1 && right_guard <= 1,
        "pointwise inputs were canonicalized"
    );

    if left_guard != 0 {
        let cl = ml.wrapping_add(1);
        // SAFETY: both complete coefficient spans are disjoint.
        unsafe { left.get_unchecked_mut(..cl) }
            .copy_from_slice(unsafe { right.get_unchecked(..cl) });
        // SAFETY: left now contains right's canonical residue.
        unsafe {
            SsaRing::negate(left, mod_bits);
        }
        return true;
    }
    if right_guard != 0 {
        // SAFETY: left is canonical, so in-place negation is valid.
        unsafe {
            SsaRing::negate(left, mod_bits);
        }
        return true;
    }
    false
}

/// Multiplies two Fermat-ring elements and stores the reduced result.
///
/// Computes `(a * b) mod (2^mod_bits + 1)` and writes the result to `dst`.
///
/// For small `mod_bits` (≤ `SSA_BASE_MODULUS_BITS`), uses the standard
/// multiplication tower. For larger values, recursively applies the FFT.
///
/// # Safety
/// - `dst`, `left`, `right` each have `cl = SsaRing::coeff_limbs(mod_bits)` limbs.
/// - `mul_scratch` has at least `2 * cl` limbs.
#[allow(
    clippy::inline_always,
    reason = "benchmarking shows the fixed pointwise plan must propagate through the coefficient loop"
)]
#[inline(always)]
unsafe fn fermat_mul_into(
    dst: &mut [Limb],
    left: &[Limb],
    right: &[Limb],
    mod_bits: usize,
    basecase_plan: Option<MulPlan>,
    mul_scratch: &mut [Limb],
) {
    // SAFETY: the caller guarantees three complete coefficient spans.
    if unsafe { SsaPointwise::write_special_residue_product(dst, left, right, mod_bits) } {
        return;
    }

    if mod_bits <= SSA_BASE_MODULUS_BITS {
        let Some(plan) = basecase_plan else {
            debug_assert!(false, "the pointwise loop must preselect its basecase plan");
            return;
        };
        // SAFETY: all coefficient buffers have cl limbs, the special -1 cases
        // returned above, and mul_scratch has at least 2*cl > 2*ml limbs.
        unsafe {
            fermat_basecase_mul_with_plan(dst, left, right, mod_bits, plan, mul_scratch);
        }
    } else {
        // SAFETY: all buffers correctly sized for recursive call.
        unsafe {
            SsaTransform::fft_mul_mod_slices(
                dst,
                left,
                right,
                mod_bits,
                None,
                false,
                None,
                mul_scratch,
            );
        }
    }
}

/// Fixed-width Fermat product that overwrites its left operand after the lower
/// multiplication tier has consumed it.
///
/// # Safety
/// - `left` and `right` are disjoint complete coefficients.
/// - Both coefficients are canonical.
/// - `plan` and `product_scratch` satisfy the same fixed-width contract as
///   [`fermat_basecase_mul_with_plan`].
#[allow(
    clippy::inline_always,
    reason = "the in-place pointwise product must inline through the coefficient loop"
)]
#[inline(always)]
unsafe fn fermat_basecase_mul_assign_left(
    left: &mut [Limb],
    right: &[Limb],
    mod_bits: usize,
    plan: MulPlan,
    product_scratch: &mut [Limb],
) {
    let ml = SsaRing::mod_limbs(mod_bits);
    let product_span = ml.wrapping_mul(2);
    // SAFETY: the caller guarantees the ordinary basecase scratch length.
    let (product, tower_scratch) = unsafe { product_scratch.split_at_mut_unchecked(product_span) };
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
        SsaPointwise::reduce_full_product(left, product, ml);
    }
}

/// Fixed-width Fermat product with a lower-tower plan selected by the caller.
///
/// # Safety
/// The buffer and residue preconditions are the same as
/// [`SsaPointwise::fermat_basecase_mul_into`], and `plan` was selected for two `ml`-limb
/// operands where `ml = SsaRing::mod_limbs(mod_bits)`.
#[allow(
    clippy::inline_always,
    reason = "benchmarking shows inlining is required to hoist lower-tier dispatch"
)]
#[inline(always)]
unsafe fn fermat_basecase_mul_with_plan(
    dst: &mut [Limb],
    left: &[Limb],
    right: &[Limb],
    mod_bits: usize,
    plan: MulPlan,
    product_scratch: &mut [Limb],
) {
    let ml = SsaRing::mod_limbs(mod_bits);
    let product_span = ml.wrapping_mul(2);
    // SAFETY: the caller guarantees product_scratch has the basecase scratch
    // length, whose first product_span limbs hold the complete product.
    let (product, tower_scratch) = unsafe { product_scratch.split_at_mut_unchecked(product_span) };
    // FFT coefficients occupy fixed-width slots and become dense after the
    // first butterfly. Multiplying the complete data spans avoids two backward
    // active-length scans, a second tier selection for scratch sizing, and any
    // product clearing: every lower multiplication tier overwrites the exact
    // `2 * ml`-limb result. Leading zero limbs remain algebraically harmless.
    // SAFETY: all three ranges have their exact fixed widths by construction;
    // tower_scratch was sized for this full-width product.
    Multiplication::execute_plan(
        plan,
        unsafe { product.get_unchecked_mut(..product_span) },
        unsafe { left.get_unchecked(..ml) },
        unsafe { right.get_unchecked(..ml) },
        tower_scratch,
    );
    // SAFETY: the multiplication overwrote the complete 2*ml-limb product and
    // dst is a disjoint complete coefficient.
    unsafe {
        SsaPointwise::reduce_full_product(dst, product, ml);
    }
}
