//! Sequential dense convolution with input borrows ending before reconstruction.

#![expect(
    unsafe_code,
    reason = "Retained operation plans bound staging, convolution, and reconstruction partitions with explicit phase lifetimes"
)]

use crate::parallel::SequentialExecutor;

use super::{
    FftPlan, Limb, LimbOutput, MulTransformPlan, Residue, SquareTransformPlan, SsaCoefficients,
    SsaRing, SsaTransform,
};

/// An inverted coefficient matrix and the arena available for reconstruction.
///
/// Only the plan and scratch are retained. Operand borrows end when convolution
/// returns, so reconstruction may overwrite either consumed input. Multiplication
/// also releases its right matrix into the reconstruction arena.
pub struct DenseWorkspace<'plan, 'work> {
    plan: &'plan FftPlan,
    matrix: &'work mut [Limb],
    scratch: &'work mut [Limb],
}

impl MulTransformPlan {
    /// Multiplies a canonical coefficient in place using the retained transform.
    ///
    /// # Safety
    /// Both operands are complete coefficients for this plan's modulus, with
    /// zero guards. `scratch` covers `self.transform_mul_scratch(1)`. All three
    /// slices are disjoint; the right operand remains available after execution.
    pub unsafe fn mul_assign_left(&self, left: &mut [Limb], right: &[Limb], scratch: &mut [Limb]) {
        let ml = SsaRing::mod_limbs(self.modulus_bits);
        // SAFETY: both canonical coefficients contain their complete data and
        // guard spans. The caller has already handled the -1 representation.
        unsafe {
            debug_assert_eq!(*left.get_unchecked(ml), 0, "ordinary left coefficient");
            debug_assert_eq!(*right.get_unchecked(ml), 0, "ordinary right coefficient");
            if SsaRing::classify_residue(left, ml) == Residue::Zero {
                return;
            }
            if SsaRing::classify_residue(right, ml) == Residue::Zero {
                left.fill(0);
                return;
            }
        }
        // SAFETY: canonical zero-guard inputs occupy the complete ring width.
        // Convolution retains references only to its plan and disjoint scratch;
        // the shared reborrow of left ends before its mutable reconstruction.
        unsafe {
            DenseWorkspace::mul(left, right, self, scratch).reconstruct(left);
        }
    }
}

impl SquareTransformPlan {
    /// Squares a canonical coefficient in place using the retained transform.
    ///
    /// # Safety
    /// `value` is a complete coefficient for this plan's modulus with a zero
    /// guard. Disjoint `scratch` covers `self.transform_sqr_scratch(1)`.
    pub unsafe fn sqr_assign(&self, value: &mut [Limb], scratch: &mut [Limb]) {
        let ml = SsaRing::mod_limbs(self.modulus_bits);
        // SAFETY: the canonical coefficient contains its data and zero guard;
        // the pointwise boundary has already handled the -1 representation.
        unsafe {
            debug_assert_eq!(*value.get_unchecked(ml), 0, "ordinary square coefficient");
            if SsaRing::classify_residue(value, ml) == Residue::Zero {
                return;
            }
        }
        // SAFETY: the complete canonical input and plan-sized scratch are
        // disjoint. The workspace retains no input borrow after convolution,
        // so reconstruction exclusively borrows the consumed coefficient.
        unsafe {
            DenseWorkspace::sqr(value, self, scratch).reconstruct(value);
        }
    }
}

impl<'plan, 'work> DenseWorkspace<'plan, 'work> {
    /// Stages both inputs, multiplies their spectra, and inverts the left matrix.
    ///
    /// # Safety
    /// Both operands have canonical zero guards or implicit zero high limbs.
    /// The selected transform uses every input chunk. `scratch` is disjoint
    /// from both operands and covers `plan.transform_mul_scratch(1)`.
    pub unsafe fn mul(
        left: &[Limb],
        right: &[Limb],
        plan: &'plan MulTransformPlan,
        scratch: &'work mut [Limb],
    ) -> Self {
        debug_assert!(
            scratch.len() >= plan.transform_mul_scratch(1),
            "dense product arena must cover the retained sequential plan"
        );
        // SAFETY: the validated layout contains two complete matrices followed
        // by the staging/convolution arena. Splitting retains disjoint borrows.
        let (left_matrix, after_left) =
            unsafe { scratch.split_at_mut_unchecked(plan.mat_limbs.get()) };
        // SAFETY: the second matrix is reserved after the first in the same plan.
        let (right_matrix, after_right) =
            unsafe { after_left.split_at_mut_unchecked(plan.mat_limbs.get()) };
        // SAFETY: the forward arena reserves at least two complete coefficients;
        // its representable size bounds this product and the staging prefix.
        let stage = unsafe {
            let len = plan.inner_cl.get().unchecked_mul(2);
            after_right.get_unchecked_mut(..len)
        };
        // SAFETY: staging writes complete initialized matrices, disjoint from
        // both operands and the two-coefficient staging span. Each synchronous
        // split consumes its input before the staging span is reused.
        unsafe {
            SsaCoefficients::split_twisted(
                left,
                left_matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_cl,
                plan.periods,
                plan.twist_step_half,
                stage,
            );
            SsaCoefficients::split_twisted(
                right,
                right_matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_cl,
                plan.periods,
                plan.twist_step_half,
                stage,
            );
        }
        // SAFETY: staging has ended. Both complete matrices and the retained
        // pointwise plan share this ring, and after_right covers its leaf arena.
        unsafe {
            SsaTransform::convolve_subtrees(
                left_matrix,
                right_matrix,
                plan.transform_len,
                plan.twist_step_half,
                plan.pointwise(),
                &SequentialExecutor,
                after_right,
            );
        }
        Self {
            plan,
            matrix: left_matrix,
            scratch: after_left,
        }
    }

    /// Stages an input, squares its spectrum, and inverts the same matrix.
    ///
    /// # Safety
    /// The operand has a canonical zero guard or implicit zero high limbs and
    /// uses every input chunk. Disjoint scratch covers the sequential square plan.
    pub unsafe fn sqr(
        value: &[Limb],
        plan: &'plan SquareTransformPlan,
        scratch: &'work mut [Limb],
    ) -> Self {
        debug_assert!(
            scratch.len() >= plan.transform_sqr_scratch(1),
            "dense square arena must cover the retained sequential plan"
        );
        // SAFETY: the validated square layout reserves one complete matrix and
        // the reusable staging, pointwise, and reconstruction arena after it.
        let (matrix, after_matrix) =
            unsafe { scratch.split_at_mut_unchecked(plan.mat_limbs.get()) };
        // SAFETY: two staging coefficients fit the validated square arena, so
        // the length product is representable and the prefix is initialized.
        let stage = unsafe {
            let len = plan.inner_cl.get().unchecked_mul(2);
            after_matrix.get_unchecked_mut(..len)
        };
        // SAFETY: the input, complete matrix, and staging span are disjoint.
        // After splitting, no operation retains a borrow of the input.
        unsafe {
            SsaCoefficients::split_twisted(
                value,
                matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_cl,
                plan.periods,
                plan.twist_step_half,
                stage,
            );
            SsaTransform::convolve_square_subtrees(
                matrix,
                plan.transform_len,
                plan.twist_step_half,
                plan.pointwise(),
                &SequentialExecutor,
                after_matrix,
            );
        }
        Self {
            plan,
            matrix,
            scratch: after_matrix,
        }
    }

    /// Consumes the inverse matrix and folds directly into the output coefficient.
    ///
    /// # Safety
    /// `dst` contains a complete coefficient for the plan's modulus, or a
    /// shorter guard-free span proved to contain the exact unreduced product.
    pub unsafe fn reconstruct(self, dst: &mut [impl LimbOutput]) {
        let plan = self.plan;
        // SAFETY: the plan reserves sequential inverse-twiddle slots and the
        // reconstruction arena after the live matrix. Multiplication releases
        // the consumed right matrix into this span before constructing Self.
        let (twiddle, reconstruction) = unsafe {
            let len = plan.inner_cl.get().unchecked_mul(plan.parallel_slots(1));
            self.scratch.split_at_mut_unchecked(len)
        };
        // SAFETY: convolution established every inverse coefficient and its
        // signed reconstruction bound. Matrix, twiddle, accumulator, and output
        // are disjoint; no operand references remain in this workspace.
        unsafe {
            SsaCoefficients::reconstruct(
                self.matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                plan.modulus_bits,
                dst,
                reconstruction,
                Some((plan.inverse_twist(), twiddle)),
                &SequentialExecutor,
            );
        }
    }
}
