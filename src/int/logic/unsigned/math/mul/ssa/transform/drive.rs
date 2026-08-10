//! End-to-end multiplication and squaring orchestration for SSA.

use super::{FftPlan, Limb, SSA_BASE_MODULUS_BITS, SsaCoefficients, SsaPointwise, SsaRing};

/// Namespace for SSA transform operations and layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaTransform;

/// The complete multiply and square transforms, and the operand measurement
/// their ring sizing starts from.
///
/// Other transform files contribute their operations to this namespace.
impl SsaTransform {
    /// Core recursive FFT multiplication in `Z/(2^modulus_bits + 1)`.
    ///
    /// A `forced_plan` overrides the planner's geometry. It is passed as a built
    /// plan rather than as a bare exponent so that the only way to reach this
    /// function with a forced geometry is to have already constructed a valid one
    /// for `modulus_bits` — an exponent that does not divide this ring width simply
    /// cannot be expressed here.
    ///
    /// # Safety
    /// - `dst` has at least `SsaRing::coeff_limbs(modulus_bits)` limbs.
    /// - Each operand either has that complete guarded width, or has exactly
    ///   `SsaRing::mod_limbs(modulus_bits)` limbs with an implicit zero guard and a nonzero
    ///   exact width supplied through `significant_bits`.
    /// - Values in `significant_bits`, when present, are upper bounds on the
    ///   represented operands' exact significant widths.
    /// - `forced_plan`, when present, was built for this exact `modulus_bits`, and
    ///   `scratch` is sized from it.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "FFT orchestration linear pass"
    )]
    pub unsafe fn fft_mul_mod_slices(
        dst: &mut [Limb],
        left: &[Limb],
        right: &[Limb],
        modulus_bits: usize,
        significant_bits: Option<(usize, usize)>,
        force_transform: bool,
        forced_plan: Option<&FftPlan>,
        scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(modulus_bits);
        let guarded_operands = left.len() > ml && right.len() > ml;
        if guarded_operands {
            // SAFETY: both operands include the complete guard limb in this branch.
            if unsafe {
                SsaPointwise::write_special_residue_product(dst, left, right, modulus_bits)
            } {
                return;
            }
        } else {
            debug_assert_eq!(left.len(), ml, "short Fermat operand has exact data width");
            debug_assert_eq!(right.len(), ml, "short Fermat operand has exact data width");
            debug_assert!(
                significant_bits
                    .is_some_and(|(left_bits, right_bits)| { left_bits != 0 && right_bits != 0 }),
                "implicit-guard operands must carry nonzero exact widths"
            );
        }

        if modulus_bits <= SSA_BASE_MODULUS_BITS && !force_transform {
            debug_assert!(
                guarded_operands,
                "implicit guards are only accepted by the transform path"
            );
            let bc_len = SsaPointwise::fermat_basecase_scratch_len(modulus_bits);
            let (bc, _) = scratch.split_at_mut(bc_len);
            // SAFETY: all three coefficients have cl limbs, not -1 or zero.
            unsafe {
                SsaPointwise::fermat_basecase_mul_into(dst, left, right, modulus_bits, bc);
            }
            return;
        }

        // ── Recursive FFT case ───────────────────────────────────────────────
        let plan = forced_plan
            .copied()
            .unwrap_or_else(|| FftPlan::new(modulus_bits));

        let half_transform_bits = plan.chunk_bits.wrapping_mul(plan.transform_len >> 1);
        let (left_upper_half_zero, right_upper_half_zero) =
            significant_bits.map_or((false, false), |(left_bits, right_bits)| {
                (
                    left_bits <= half_transform_bits,
                    right_bits <= half_transform_bits,
                )
            });

        // This point is past the basecase early return, so the transform layout is
        // the one that must fit regardless of how narrow the ring is. Every caller
        // sizes its buffer from the same plan, so this never re-allocates; the
        // assertion catches a mis-sized caller in debug and test builds rather than
        // hiding an allocation in the hot path.
        debug_assert!(
            scratch.len() >= plan.transform_mul_scratch(),
            "SSA transform scratch is undersized; size it with transform_mul_scratch"
        );

        // Partition caller scratch: [left] [right] [transpose] [twiddle] [product] [recon]
        let (left_matrix, after_left) = scratch.split_at_mut(plan.mat_limbs);
        let (right_matrix, after_right) = after_left.split_at_mut(plan.mat_limbs);
        let (transpose_scratch, after_trans) = after_right.split_at_mut(plan.trans_len);
        let (twiddle_scratch, after_twid) = after_trans.split_at_mut(plan.inner_cl.wrapping_mul(2));
        let (product_scratch, recon_scratch) = after_twid.split_at_mut(plan.prod_scratch_size);
        twiddle_scratch.fill(0);

        // A whole-bit twist can be applied while splitting, so each matrix is
        // written once instead of split and then rewritten in a second RAM pass.
        // SAFETY: both matrices have the plan's complete layout and the twiddle
        // buffer is disjoint and contains at least one complete coefficient.
        let fused_twist = unsafe {
            SsaCoefficients::split_twisted(
                left,
                left_matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                plan.twist_step_half,
                twiddle_scratch,
            ) && SsaCoefficients::split_twisted(
                right,
                right_matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                plan.twist_step_half,
                twiddle_scratch,
            )
        };
        if !fused_twist {
            for (operand, matrix) in [(left, &mut *left_matrix), (right, &mut *right_matrix)] {
                SsaCoefficients::split(
                    operand,
                    matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                );
            }
            // SAFETY: both matrices were just filled with transform_len complete
            // coefficients and twiddle_scratch is disjoint from them.
            unsafe {
                apply_forward_twist(left_matrix, Some(right_matrix), &plan, twiddle_scratch);
            }
        }

        // SAFETY: matrices correctly sized, twiddle_scratch has inner_cl limbs.
        unsafe {
            Self::fft_in_place(
                left_matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                false,
                left_upper_half_zero,
                twiddle_scratch,
                transpose_scratch,
            );
            Self::fft_in_place(
                right_matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                false,
                right_upper_half_zero,
                twiddle_scratch,
                transpose_scratch,
            );
        }

        // ── Pointwise products ───────────────────────────────────────────────
        // SAFETY: the matrices are perfectly sized for exactly transform_len entries.
        unsafe {
            SsaPointwise::pointwise_multiply(
                left_matrix,
                right_matrix,
                plan.transform_len,
                plan.inner_bits,
                product_scratch,
            );
        }

        // SAFETY: matrices correctly sized, twiddle_scratch has inner_cl limbs.
        unsafe {
            Self::fft_in_place(
                left_matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                true,
                false,
                twiddle_scratch,
                transpose_scratch,
            );
        }

        // The inverse twiddle rides along with the reconstruction sweep, so the
        // coefficient matrix never makes a separate correction pass through RAM.
        SsaCoefficients::reconstruct(
            left_matrix,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_bits,
            modulus_bits,
            dst,
            recon_scratch,
            Some((plan.inverse_twist(), twiddle_scratch)),
        );

        // SAFETY: dst has cl limbs, modulus_bits matches.
        unsafe {
            SsaRing::normalize(dst, modulus_bits);
        }
    }

    /// Core recursive FFT squaring: one forward transform instead of two.
    ///
    /// # Safety
    /// `dst` and `a` each have at least `SsaRing::coeff_limbs(modulus_bits)` limbs and are
    /// disjoint, and `scratch` is sized from the plan this ring width selects.
    pub unsafe fn fft_sqr_mod_slices(
        dst: &mut [Limb],
        a: &[Limb],
        modulus_bits: usize,
        force_transform: bool,
        scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(modulus_bits);
        let cl = SsaRing::coeff_limbs(modulus_bits);

        // SAFETY: ml < cl and the caller guarantees a has cl limbs.
        let class = unsafe { SsaRing::classify_residue(a, ml) };
        // SAFETY: dst has cl limbs and cl = ml + 1 is non-zero.
        if unsafe { SsaPointwise::write_special_residue_square(dst, cl, class) } {
            return;
        }

        if modulus_bits <= SSA_BASE_MODULUS_BITS && !force_transform {
            // SAFETY: coefficient is canonical, not -1 or zero, buffers correctly sized.
            unsafe {
                SsaPointwise::fermat_basecase_sqr_into(dst, a, modulus_bits, scratch);
            }
            return;
        }

        let plan = FftPlan::new(modulus_bits);
        debug_assert!(
            scratch.len() >= plan.transform_sqr_scratch(),
            "SSA square scratch is undersized; size it with transform_sqr_scratch"
        );

        let (matrix, after_matrix) = scratch.split_at_mut(plan.mat_limbs);
        let (transpose_scratch, after_transpose) = after_matrix.split_at_mut(plan.trans_len);
        let (twiddle_scratch, after_twiddle) =
            after_transpose.split_at_mut(plan.inner_cl.wrapping_mul(2));
        let (sqr_scratch, recon_scratch) = after_twiddle.split_at_mut(plan.sqr_scratch_size);
        twiddle_scratch.fill(0);

        // SAFETY: matrix and twiddle scratch are complete, disjoint plan-sized spans.
        let fused_twist = unsafe {
            SsaCoefficients::split_twisted(
                a,
                matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                plan.twist_step_half,
                twiddle_scratch,
            )
        };
        if !fused_twist {
            SsaCoefficients::split(
                a,
                matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
            );
            // SAFETY: matrix and twiddle scratch are complete and disjoint.
            unsafe {
                apply_forward_twist(matrix, None, &plan, twiddle_scratch);
            }
        }

        // SAFETY: all matrix and scratch spans were partitioned from the plan.
        unsafe {
            Self::fft_in_place(
                matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                false,
                false,
                twiddle_scratch,
                transpose_scratch,
            );
            pointwise_square(matrix, plan.transform_len, plan.inner_bits, sqr_scratch);
            Self::fft_in_place(
                matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                true,
                false,
                twiddle_scratch,
                transpose_scratch,
            );
        }

        // The inverse twiddle rides along with the reconstruction sweep, so the
        // coefficient matrix never makes a separate correction pass through RAM.
        SsaCoefficients::reconstruct(
            matrix,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_bits,
            modulus_bits,
            dst,
            recon_scratch,
            Some((plan.inverse_twist(), twiddle_scratch)),
        );
        // SAFETY: dst has cl limbs and modulus_bits matches.
        unsafe {
            SsaRing::normalize(dst, modulus_bits);
        }
    }
}

/// Applies the forward pre-twist to one or more coefficient matrices.
/// Accumulating the half-bit exponent inside one ring period prevents wrapping.
///
/// # Safety
/// Every matrix is complete; `twiddle` is a disjoint `inner_cl`-limb buffer.
unsafe fn apply_forward_twist(
    matrix: &mut [Limb],
    mut second_matrix: Option<&mut [Limb]>,
    plan: &FftPlan,
    twiddle: &mut [Limb],
) {
    let period = plan.inner_bits.wrapping_mul(4);
    let mut shift = 0_usize;
    for index in 1_usize..plan.transform_len {
        shift = shift.wrapping_add(plan.twist_step_half);
        if shift >= period {
            shift = shift.wrapping_sub(period);
        }
        // SAFETY: index < transform_len and matrix holds complete coefficients.
        let slot = unsafe { SsaTransform::coeff_mut(matrix, index, plan.inner_cl) };
        // SAFETY: slot is canonical and disjoint from twiddle after splitting.
        unsafe {
            SsaRing::shift_half(slot, shift, plan.inner_bits, twiddle);
        }
        if let Some(second) = second_matrix.as_deref_mut() {
            // SAFETY: the second matrix has the same complete, disjoint layout.
            let second_slot = unsafe { SsaTransform::coeff_mut(second, index, plan.inner_cl) };
            // SAFETY: second_slot is canonical and disjoint from twiddle.
            unsafe {
                SsaRing::shift_half(second_slot, shift, plan.inner_bits, twiddle);
            }
        }
    }
}

// ── Private helper functions ─────────────────────────────────────────────────

/// Squares every coefficient of a transformed matrix in place.
///
/// # Safety
/// `matrix` holds `transform_len` complete `SsaRing::coeff_limbs(mod_bits)`-limb
/// coefficients and `sqr_scratch` is sized from the same plan.
unsafe fn pointwise_square(
    matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    sqr_scratch: &mut [Limb],
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
    let (result, mul_scratch) = sqr_scratch.split_at_mut(cl);

    for index in 0..transform_len {
        // SAFETY: index is inside the complete coefficient matrix.
        let slot = unsafe { SsaTransform::coeff_mut(matrix, index, cl) };
        // SAFETY: slot and scratch are complete disjoint coefficients.
        unsafe {
            SsaRing::normalize(slot, mod_bits);
            fermat_sqr_into(result, slot, mod_bits, mul_scratch);
        }
        slot.copy_from_slice(result);
    }
}

/// Squares one Fermat residue, recursing into the transform above the basecase.
///
/// # Safety
/// `dst` and `value` are disjoint complete coefficients and `mul_scratch` is
/// sized for the path this ring width takes.
unsafe fn fermat_sqr_into(
    dst: &mut [Limb],
    value: &[Limb],
    mod_bits: usize,
    mul_scratch: &mut [Limb],
) {
    let ml = SsaRing::mod_limbs(mod_bits);
    let cl = ml.wrapping_add(1);

    // SAFETY: value is one complete coefficient.
    let class = unsafe { SsaRing::classify_residue(value, ml) };
    // SAFETY: dst has cl limbs and cl is nonzero.
    if unsafe { SsaPointwise::write_special_residue_square(dst, cl, class) } {
        return;
    }

    if mod_bits <= SSA_BASE_MODULUS_BITS {
        // SAFETY: the coefficient is ordinary and buffers are complete.
        unsafe {
            SsaPointwise::fermat_basecase_sqr_into(dst, value, mod_bits, mul_scratch);
        }
    } else {
        // Recurse into the squaring driver, not the product one. An inner ring
        // above the basecase width has its own transform, and handing it to
        // `fft_mul_mod_slices` with two aliased operands ran two forward
        // transforms over identical data — losing the square's discount at
        // exactly the widths where the pointwise stage nests, which since the
        // inner-ring rounding fix is every RAM-resident size.
        //
        // The scratch is sized for the product recursion by `from_geometry`,
        // and a squaring transform holds one coefficient matrix where a product
        // holds two, so it is a strict over-allocation rather than a shortfall.
        //
        // SAFETY: the operand is one complete coefficient disjoint from `dst`,
        // and `mul_scratch` exceeds this ring's squaring layout.
        unsafe {
            SsaTransform::fft_sqr_mod_slices(dst, value, mod_bits, false, mul_scratch);
        }
    }
}
