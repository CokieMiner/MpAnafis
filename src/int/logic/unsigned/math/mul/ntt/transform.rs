//! The multi-prime NTT driver: split, forward transforms, pointwise, inverse.

use alloc::vec;

use crate::parallel::ParallelExecutor;

use super::{LIMB_BITS, Limb, MODULI, NttMultiplicationPlan, TransformPlan};

/// Namespace for the exact multi-prime number-theoretic transform tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ntt;

/// Disjoint per-modulus storage used while a transform plan is executing.
///
/// Each modulus receives its own output, scratch, and twiddle span. This is
/// the ownership boundary that lets an executor join independent transforms
/// without raw pointers or shared mutable state. The workspace is only built
/// for two- and three-prime plans; the one-prime plan has a separate Goldilocks
/// workspace because it has no second or third modulus span.
pub struct PrimeWorkspace<'workspace> {
    outputs: &'workspace mut [u32],
    workers: &'workspace mut [u32],
    twiddles: &'workspace mut [u32],
    digits_out: &'workspace mut [u32],
    transform_len: usize,
    modulus_count: usize,
}

impl<'workspace> PrimeWorkspace<'workspace> {
    /// Creates disjoint views over one validated multi-prime layout.
    pub const fn new(
        outputs: &'workspace mut [u32],
        workers: &'workspace mut [u32],
        twiddles: &'workspace mut [u32],
        digits_out: &'workspace mut [u32],
        transform_len: usize,
        modulus_count: usize,
    ) -> Self {
        PrimeWorkspace {
            outputs,
            workers,
            twiddles,
            digits_out,
            transform_len,
            modulus_count,
        }
    }

    fn square<E: ParallelExecutor>(&mut self, digits_a: &[u32], executor: &E) {
        // This workspace is constructed only for two- or three-prime plans;
        // the one-prime path uses its dedicated Goldilocks workspace below.
        debug_assert!(
            (2..=3).contains(&self.modulus_count),
            "prime workspace requires two or three moduli"
        );
        let (first, output_tail_after_first) = self.outputs.split_at_mut(self.transform_len);
        let (second, output_tail_after_second) =
            output_tail_after_first.split_at_mut(self.transform_len);
        let twiddle_len = self.transform_len.div_ceil(2);
        let (twiddle_first, rest_twiddles) = self.twiddles.split_at_mut(twiddle_len);
        let (twiddle_second, twiddle_third) = rest_twiddles.split_at_mut(twiddle_len);

        if self.modulus_count == 2 {
            executor.join(
                || {
                    Ntt::square_mod_slice_with_executor(
                        first,
                        digits_a,
                        MODULI[0],
                        twiddle_first,
                        executor,
                    );
                },
                || {
                    Ntt::square_mod_slice_with_executor(
                        second,
                        digits_a,
                        MODULI[1],
                        twiddle_second,
                        executor,
                    );
                },
            );
            return;
        }

        let (third, _) = output_tail_after_second.split_at_mut(self.transform_len);
        executor.join(
            || {
                Ntt::square_mod_slice_with_executor(
                    first,
                    digits_a,
                    MODULI[0],
                    twiddle_first,
                    executor,
                );
            },
            || {
                executor.join(
                    || {
                        Ntt::square_mod_slice_with_executor(
                            second,
                            digits_a,
                            MODULI[1],
                            twiddle_second,
                            executor,
                        );
                    },
                    || {
                        Ntt::square_mod_slice_with_executor(
                            third,
                            digits_a,
                            MODULI[2],
                            twiddle_third,
                            executor,
                        );
                    },
                );
            },
        );
    }

    pub fn multiply<E: ParallelExecutor>(
        &mut self,
        digits_a: &[u32],
        digits_b: &[u32],
        executor: &E,
    ) {
        // This workspace is constructed only for two- or three-prime plans;
        // the one-prime path uses its dedicated Goldilocks workspace below.
        debug_assert!(
            (2..=3).contains(&self.modulus_count),
            "prime workspace requires two or three moduli"
        );
        let (first, output_tail_after_first) = self.outputs.split_at_mut(self.transform_len);
        let (second, output_tail_after_second) =
            output_tail_after_first.split_at_mut(self.transform_len);
        let (scratch_first, rest_workers) = self.workers.split_at_mut(self.transform_len);
        let (scratch_second, scratch_rest) = rest_workers.split_at_mut(self.transform_len);
        let twiddle_len = self.transform_len.div_ceil(2);
        let (twiddle_first, rest_twiddles) = self.twiddles.split_at_mut(twiddle_len);
        let (twiddle_second, twiddle_third) = rest_twiddles.split_at_mut(twiddle_len);

        if self.modulus_count == 2 {
            executor.join(
                || {
                    Ntt::convolve_mod_slice_with_executor(
                        first,
                        scratch_first,
                        digits_a,
                        digits_b,
                        MODULI[0],
                        twiddle_first,
                        executor,
                    );
                },
                || {
                    Ntt::convolve_mod_slice_with_executor(
                        second,
                        scratch_second,
                        digits_a,
                        digits_b,
                        MODULI[1],
                        twiddle_second,
                        executor,
                    );
                },
            );
            return;
        }

        let (third, _) = output_tail_after_second.split_at_mut(self.transform_len);
        let (scratch_third, _) = scratch_rest.split_at_mut(self.transform_len);
        executor.join(
            || {
                Ntt::convolve_mod_slice_with_executor(
                    first,
                    scratch_first,
                    digits_a,
                    digits_b,
                    MODULI[0],
                    twiddle_first,
                    executor,
                );
            },
            || {
                executor.join(
                    || {
                        Ntt::convolve_mod_slice_with_executor(
                            second,
                            scratch_second,
                            digits_a,
                            digits_b,
                            MODULI[1],
                            twiddle_second,
                            executor,
                        );
                    },
                    || {
                        Ntt::convolve_mod_slice_with_executor(
                            third,
                            scratch_third,
                            digits_a,
                            digits_b,
                            MODULI[2],
                            twiddle_third,
                            executor,
                        );
                    },
                );
            },
        );
    }

    pub fn reconstruct_two(&mut self, convolution_len: usize, digit_bits: u32) -> &[u32] {
        let (first, rest_outputs) = self.outputs.split_at(self.transform_len);
        let (second, _) = rest_outputs.split_at(self.transform_len);
        // SAFETY: the transform planner proves convolution_len <= transform_len.
        let first_slice = unsafe { first.get_unchecked(..convolution_len) };
        // SAFETY: the transform planner proves convolution_len <= transform_len.
        let second_slice = unsafe { second.get_unchecked(..convolution_len) };
        let count =
            Ntt::reconstruct_two_slices(self.digits_out, first_slice, second_slice, digit_bits);
        // SAFETY: CRT reconstruction returns at most the supplied output width.
        unsafe { self.digits_out.get_unchecked(..count) }
    }

    pub fn reconstruct_three(&mut self, convolution_len: usize, digit_bits: u32) -> &[u32] {
        let (first, rest_outputs) = self.outputs.split_at(self.transform_len);
        let (second, third) = rest_outputs.split_at(self.transform_len);
        let count = Ntt::reconstruct_three_slices(
            self.digits_out,
            // SAFETY: the transform planner proves convolution_len <= transform_len.
            unsafe { first.get_unchecked(..convolution_len) },
            // SAFETY: the transform planner proves convolution_len <= transform_len.
            unsafe { second.get_unchecked(..convolution_len) },
            // SAFETY: the transform planner proves convolution_len <= transform_len.
            unsafe { third.get_unchecked(..convolution_len) },
            digit_bits,
        );
        // SAFETY: CRT reconstruction returns at most the supplied output width.
        unsafe { self.digits_out.get_unchecked(..count) }
    }
}

impl Ntt {
    /// Whether the executor-aware NTT entry point can compute these widths.
    ///
    /// The capability counterpart of [`Ssa::admits_mul`](super::super::Ssa::admits_mul):
    /// it reports whether the fixed prime set carries a transform long enough for
    /// this product, and says nothing about whether the transform is the fastest
    /// tier. Empty operands are accepted because the product is then a fill.
    pub fn admits_mul(len_a: usize, len_b: usize) -> bool {
        if len_a == 0 || len_b == 0 {
            return true;
        }
        Self::choose_transform_plan(len_a, len_b).is_some_and(|plan| {
            Self::estimated_transform_len(len_a, len_b, plan.digit_bits).is_some_and(
                |transform_len| {
                    Self::MAX_TRANSFORM_LEN.is_none_or(|max_len| transform_len <= max_len)
                        && Self::coefficient_range_fits(
                            transform_len,
                            plan.digit_bits,
                            plan.modulus_count,
                        )
                },
            )
        })
    }

    /// Square using the supplied synchronous execution policy.
    pub fn try_sqr_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        plan: TransformPlan,
        executor: &E,
    ) -> bool {
        if !plan.is_valid() {
            return false;
        }
        if a.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(capacity_a) = digit_capacity(a.len(), plan.digit_bits) else {
            return false;
        };
        let Some(max_transform_len) = transform_capacity(capacity_a, capacity_a) else {
            return false;
        };
        if Self::MAX_TRANSFORM_LEN.is_some_and(|max_len| max_transform_len > max_len) {
            return false;
        }

        let twiddle_len = max_transform_len.div_ceil(2);
        let Some(prime_workspace_len) = max_transform_len.checked_mul(plan.modulus_count) else {
            return false;
        };
        let Some(twiddle_workspace) = twiddle_len.checked_mul(plan.modulus_count) else {
            return false;
        };
        let Some(scratch_len) = capacity_a
            .checked_add(prime_workspace_len)
            .and_then(|total| total.checked_add(max_transform_len))
            .and_then(|total| total.checked_add(max_transform_len))
            .and_then(|total| total.checked_add(twiddle_workspace))
        else {
            return false;
        };
        let mut scratch_u32 = vec![0_u32; scratch_len];
        let (buf_a, rest_all) = scratch_u32.split_at_mut(capacity_a);
        let (rest_transforms, scratch_tail) = rest_all.split_at_mut(prime_workspace_len);
        let (workers, digit_tail) = scratch_tail.split_at_mut(max_transform_len);
        let (digits_out, twiddle_buf) = digit_tail.split_at_mut(max_transform_len);

        // SAFETY: validated capacities reserve every possible digit for `a`.
        let len_a = unsafe { Self::limbs_to_digits_into(buf_a, a, plan.digit_bits) };
        if len_a == 0 {
            dst.fill(0);
            return true;
        }
        // SAFETY: len_a <= buf_a.len() returned by limbs_to_digits_into.
        let digits_a = unsafe { buf_a.get_unchecked(..len_a) };

        let Some(convolution_len) = len_a.checked_mul(2).and_then(|sum| sum.checked_sub(1)) else {
            return false;
        };
        let Some(transform_len) = convolution_len.checked_next_power_of_two() else {
            return false;
        };
        if !Self::coefficient_range_fits(transform_len, plan.digit_bits, plan.modulus_count) {
            return false;
        }

        if plan.modulus_count == 1 {
            let Some(single_scratch_len) = transform_len.checked_mul(2) else {
                return false;
            };
            let mut scratch_u64 = vec![0_u64; single_scratch_len];
            let (single_output, _) = rest_transforms.split_at_mut(transform_len);
            let count = Self::square_digits_into(
                single_output,
                digits_a,
                convolution_len,
                transform_len,
                plan.digit_bits,
                &mut scratch_u64,
                executor,
            );
            // SAFETY: count <= single_output.len().
            let valid_digits = unsafe { single_output.get_unchecked(..count) };
            // SAFETY: the destination is the validated product width and the
            // reconstructed digits fit that width.
            unsafe {
                Self::digits_to_limbs(dst, valid_digits, plan.digit_bits);
            }
            return true;
        }

        let mut prime_workspace = PrimeWorkspace::new(
            rest_transforms,
            workers,
            twiddle_buf,
            digits_out,
            transform_len,
            plan.modulus_count,
        );
        prime_workspace.square(digits_a, executor);
        let valid_digits = if plan.modulus_count == 2 {
            prime_workspace.reconstruct_two(convolution_len, plan.digit_bits)
        } else {
            prime_workspace.reconstruct_three(convolution_len, plan.digit_bits)
        };
        // SAFETY: the destination is the validated product width and the
        // reconstructed digits fit that width.
        unsafe {
            Self::digits_to_limbs(dst, valid_digits, plan.digit_bits);
        }
        true
    }

    /// Multiply using the supplied synchronous execution policy.
    pub fn try_mul_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        plan: TransformPlan,
        executor: &E,
    ) -> bool {
        let Some(prepared) = NttMultiplicationPlan::try_new(a, b, plan) else {
            return false;
        };
        if dst.len() < prepared.destination_len() {
            return false;
        }
        // SAFETY: this boundary checked the complete destination width and the
        // prepared plan owns the immutable operand/geometry proof.
        unsafe { prepared.run_allocating(dst, executor) }
        true
    }
}

pub fn digit_capacity(limb_len: usize, digit_bits: u32) -> Option<usize> {
    let digit_width = usize::try_from(digit_bits).ok()?;
    limb_len
        .checked_mul(LIMB_BITS)
        .map(|bits| bits.div_ceil(digit_width))
        .and_then(|digits| digits.checked_add(1))
}

pub fn transform_capacity(capacity_a: usize, capacity_b: usize) -> Option<usize> {
    capacity_a
        .checked_add(capacity_b)
        .and_then(|sum| sum.checked_sub(1))
        .and_then(usize::checked_next_power_of_two)
}
