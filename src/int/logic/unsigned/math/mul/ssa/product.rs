//! Pointwise multiplication and basecase product reduction for SSA.

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{
    ArchKernels, FftPlan, LIMB_BITS, Limb, MulPlan, Multiplication, NegacyclicPlan, Residue,
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
        let source_coefficient = if left_class == Residue::NegOne {
            right
        } else {
            left
        };
        // SAFETY: `dst` contains the complete writable coefficient span.
        let destination = unsafe { dst.get_unchecked_mut(..cl) };
        // SAFETY: `source` contains the complete initialized coefficient span.
        let source = unsafe { source_coefficient.get_unchecked(..cl) };
        destination.copy_from_slice(source);
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

    /// Multiplies every coefficient pair using the supplied synchronous executor.
    ///
    /// Both sequential and parallel paths use the caller-provided scratch.
    /// Parallel leaves receive disjoint coefficient-aligned regions; nested
    /// coefficient products use a sequential child executor to avoid recursive
    /// oversubscription.
    ///
    /// # Safety
    /// The caller must satisfy the validated matrix and scratch preconditions
    /// described above.
    pub unsafe fn pointwise_multiply_with_executor<E: ParallelExecutor>(
        left_matrix: &mut [Limb],
        right_matrix: &mut [Limb],
        transform_len: usize,
        mod_bits: usize,
        executor: &E,
        product_scratch: &mut [Limb],
    ) {
        let basecase_plan = (mod_bits <= SSA_BASE_MODULUS_BITS).then(|| {
            let ml = SsaRing::mod_limbs(mod_bits);
            Multiplication::select_plan(ml, ml, TierCeiling::Full)
        });
        let negacyclic_plan = (mod_bits <= SSA_BASE_MODULUS_BITS)
            .then(|| NegacyclicPlan::new(SsaRing::mod_limbs(mod_bits)))
            .flatten();

        if executor.parallelism().get() > 1 && transform_len >= 16 {
            let cl = SsaRing::coeff_limbs(mod_bits);
            let needed_scratch_result = if mod_bits <= SSA_BASE_MODULUS_BITS {
                cl.checked_add(Self::fermat_basecase_scratch_len(mod_bits))
            } else {
                cl.checked_add(FftPlan::new(mod_bits).transform_mul_scratch())
            };
            let Some(needed_scratch) = needed_scratch_result else {
                debug_assert!(
                    false,
                    "pointwise scratch size overflowed during preparation"
                );
                return;
            };
            let workers =
                FftPlan::pointwise_parallelism_budget(transform_len, executor.parallelism().get());
            let leaf_len = transform_len.div_ceil(workers).max(1);
            let leaf_count = FftPlan::pointwise_leaf_count(transform_len, leaf_len);
            let Some(required_scratch) = needed_scratch.checked_mul(leaf_count) else {
                debug_assert!(
                    false,
                    "pointwise leaf arena size overflowed during preparation"
                );
                return;
            };
            debug_assert!(
                product_scratch.len() >= required_scratch,
                "pointwise scratch must be partitioned at the outer transform boundary"
            );
            // SAFETY: the matrices and caller-owned scratch contain complete,
            // coefficient-aligned partitions for every leaf.
            unsafe {
                pointwise_multiply_parallel(
                    left_matrix,
                    right_matrix,
                    transform_len,
                    mod_bits,
                    basecase_plan,
                    negacyclic_plan,
                    needed_scratch,
                    leaf_len,
                    product_scratch,
                    executor,
                );
            }
            return;
        }

        // SAFETY: left_matrix and right_matrix are disjoint slices.
        unsafe {
            pointwise_multiply_sequential(
                left_matrix,
                right_matrix,
                transform_len,
                mod_bits,
                basecase_plan,
                negacyclic_plan,
                product_scratch,
            );
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
        // The validated basecase layout always contains the full two-limb product.
        let product_span = ml.wrapping_mul(2);
        debug_assert!(
            product_span != usize::MAX,
            "validated product span overflowed"
        );
        // SAFETY: the caller guarantees product_scratch has the basecase scratch
        // length, whose first product_span limbs hold the complete product.
        let (product, tower_scratch) =
            unsafe { product_scratch.split_at_mut_unchecked(product_span) };
        // SAFETY: both ranges have their exact fixed widths by construction, and
        // `uncached_fermat_basecase_scratch_len` sizes the tail for whichever of
        // the two towers is larger.
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
    let Some(exact) = ml.checked_mul(2).and_then(|n| n.checked_add(tower)) else {
        return usize::MAX;
    };
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
        // SAFETY: `left` contains the complete writable coefficient span.
        let destination = unsafe { left.get_unchecked_mut(..cl) };
        // SAFETY: `right` is a disjoint complete initialized coefficient span.
        let source = unsafe { right.get_unchecked(..cl) };
        destination.copy_from_slice(source);
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
            SsaTransform::fft_mul_mod_slices_with_executor(
                dst,
                left,
                right,
                mod_bits,
                None,
                false,
                None,
                &SequentialExecutor,
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
    // The selected plan and its scratch layout prove this span is representable.
    let product_span = ml.wrapping_mul(2);
    debug_assert!(
        product_span != usize::MAX,
        "validated product span overflowed"
    );
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
    // The selected plan and its scratch layout prove this span is representable.
    let product_span = ml.wrapping_mul(2);
    debug_assert!(
        product_span != usize::MAX,
        "validated product span overflowed"
    );
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
    let product_output = unsafe { product.get_unchecked_mut(..product_span) };
    // SAFETY: the caller guarantees `left` contains the complete data span.
    let left_input = unsafe { left.get_unchecked(..ml) };
    // SAFETY: the caller guarantees `right` contains the complete data span.
    let right_input = unsafe { right.get_unchecked(..ml) };
    Multiplication::execute_plan(plan, product_output, left_input, right_input, tower_scratch);
    // SAFETY: the multiplication overwrote the complete 2*ml-limb product and
    // dst is a disjoint complete coefficient.
    unsafe {
        SsaPointwise::reduce_full_product(dst, product, ml);
    }
}

/// Sequential loop for pointwise multiplication over a contiguous chunk of coefficients.
///
/// # Safety
/// - `left_matrix` and `right_matrix` each have `transform_len * SsaRing::coeff_limbs(mod_bits)` limbs.
/// - `product_scratch` has at least `cl` limbs plus recursive scratch.
unsafe fn pointwise_multiply_sequential(
    left_matrix: &mut [Limb],
    right_matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    basecase_plan: Option<MulPlan>,
    negacyclic_plan: Option<NegacyclicPlan>,
    product_scratch: &mut [Limb],
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
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

/// Executes pointwise products in disjoint coefficient ranges.
///
/// # Safety
/// The two matrices contain `transform_len` complete coefficients. Recursive
/// splits occur only on coefficient boundaries; every leaf owns an independent
/// caller-provided scratch range before calling the shared sequential kernel.
#[allow(
    clippy::too_many_arguments,
    reason = "The recursive worker carries one immutable plan and one executor alongside the two disjoint matrix ranges"
)]
unsafe fn pointwise_multiply_parallel<E: ParallelExecutor>(
    left_matrix: &mut [Limb],
    right_matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    basecase_plan: Option<MulPlan>,
    negacyclic_plan: Option<NegacyclicPlan>,
    needed_scratch: usize,
    leaf_len: usize,
    scratch: &mut [Limb],
    executor: &E,
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
    if transform_len <= leaf_len {
        debug_assert!(
            scratch.len() >= needed_scratch,
            "validated pointwise leaf scratch must cover its product workspace"
        );
        // SAFETY: the outer preparation proved one complete arena per leaf.
        let leaf_scratch = unsafe { scratch.get_unchecked_mut(..needed_scratch) };
        // SAFETY: this leaf is a complete coefficient-aligned matrix partition.
        unsafe {
            pointwise_multiply_sequential(
                left_matrix,
                right_matrix,
                transform_len,
                mod_bits,
                basecase_plan,
                negacyclic_plan,
                leaf_scratch,
            );
        }
        return;
    }

    let left_count = transform_len.div_euclid(2);
    // Matrix validity proves every coefficient boundary is representable.
    let left_limbs = left_count.wrapping_mul(cl);
    let (left_first, left_second) = left_matrix.split_at_mut(left_limbs);
    let (right_first, right_second) = right_matrix.split_at_mut(left_limbs);
    // `transform_len > leaf_len >= 1` proves the split has at least two
    // coefficients, so this subtraction cannot underflow.
    let right_count = transform_len.wrapping_sub(left_count);
    let left_leaves = FftPlan::pointwise_leaf_count(left_count, leaf_len);
    // The outer preparation checked the complete leaf arena before recursion.
    let left_scratch_len = needed_scratch.wrapping_mul(left_leaves);
    debug_assert!(
        left_scratch_len != usize::MAX,
        "validated pointwise leaf arena must fit the caller scratch"
    );
    let (left_scratch, right_scratch) = scratch.split_at_mut(left_scratch_len);
    let ((), ()) = executor.join(
        || {
            // SAFETY: the first matrix ranges are disjoint and complete.
            unsafe {
                pointwise_multiply_parallel(
                    left_first,
                    right_first,
                    left_count,
                    mod_bits,
                    basecase_plan,
                    negacyclic_plan,
                    needed_scratch,
                    leaf_len,
                    left_scratch,
                    executor,
                );
            }
        },
        || {
            // SAFETY: the second matrix ranges are disjoint and complete.
            unsafe {
                pointwise_multiply_parallel(
                    left_second,
                    right_second,
                    right_count,
                    mod_bits,
                    basecase_plan,
                    negacyclic_plan,
                    needed_scratch,
                    leaf_len,
                    right_scratch,
                    executor,
                );
            }
        },
    );
}
