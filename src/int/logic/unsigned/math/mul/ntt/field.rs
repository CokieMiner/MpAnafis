//! Field-level NTT stages, pointwise operations, and coefficient conversion.

use crate::parallel::ParallelExecutor;

use super::{ArchKernels, Modulus, Ntt, NttExecutionPolicy};

#[derive(Clone, Copy)]
enum StageWidth {
    Radix2,
    Radix4,
}

#[derive(Clone, Copy)]
struct StageParameters<'twiddles> {
    block_len: usize,
    modulus: Modulus,
    stage_twiddles: &'twiddles [u32],
    width: StageWidth,
    policy: NttExecutionPolicy,
}

struct StageSetup<'twiddles> {
    block_len: usize,
    block_root: u32,
    modulus: Modulus,
    twiddle_buf: &'twiddles mut [u32],
    width: StageWidth,
    policy: NttExecutionPolicy,
}

impl Ntt {
    fn forward_dif_stage_with_executor<E: ParallelExecutor>(
        values: &mut [u32],
        setup: &mut StageSetup<'_>,
        executor: &E,
    ) {
        let half_len = setup.block_len >> 1;
        Self::generate_stage_twiddles(setup.twiddle_buf, half_len, setup.block_root, setup.modulus);
        if matches!(setup.width, StageWidth::Radix4)
            && values.len() == setup.block_len
            && setup.policy.should_split(values.len())
        {
            // A single radix-4 block has no block boundary to partition.  Its
            // two radix-2 passes are disjoint by butterfly index, so expand
            // the fused pass only at this cache-sized scheduling boundary.
            Self::apply_forward_radix4_ranges(values, setup, executor);
            return;
        }
        // SAFETY: stage twiddles were generated for exactly half_len entries.
        let stage_twiddles = unsafe { setup.twiddle_buf.get_unchecked(..half_len) };
        let stage = StageParameters {
            block_len: setup.block_len,
            modulus: setup.modulus,
            stage_twiddles,
            width: setup.width,
            policy: setup.policy,
        };
        Self::apply_forward_stage(values, stage, executor);
    }

    fn apply_forward_radix4_ranges<E: ParallelExecutor>(
        values: &mut [u32],
        setup: &mut StageSetup<'_>,
        executor: &E,
    ) {
        let half_len = setup.block_len >> 1;
        // The first DIF pass uses the twiddles already generated for the
        // fused stage.  The second pass needs W^(2j), which is generated only
        // after the first pass has consumed the original twiddle span.
        // SAFETY: generation initialized the first half_len entries.
        let first_twiddles = unsafe { setup.twiddle_buf.get_unchecked(..half_len) };
        Self::apply_radix2_stage(
            values,
            setup.block_len,
            setup.modulus,
            first_twiddles,
            setup.policy,
            executor,
            false,
        );
        let quarter_len = half_len >> 1;
        let second_root = Self::montgomery_mul(setup.block_root, setup.block_root, setup.modulus);
        Self::generate_stage_twiddles(setup.twiddle_buf, quarter_len, second_root, setup.modulus);
        // SAFETY: generation initialized the first quarter_len entries.
        let second_twiddles = unsafe { setup.twiddle_buf.get_unchecked(..quarter_len) };
        Self::apply_radix2_stage(
            values,
            half_len,
            setup.modulus,
            second_twiddles,
            setup.policy,
            executor,
            false,
        );
    }

    fn apply_forward_stage<E: ParallelExecutor>(
        values: &mut [u32],
        stage: StageParameters<'_>,
        executor: &E,
    ) {
        let block_count = values.len().div_euclid(stage.block_len);
        if !stage.policy.should_split(values.len()) || block_count < 2 {
            match stage.width {
                StageWidth::Radix2 => Self::forward_dif_stage_with_twiddles(
                    values,
                    stage.block_len,
                    stage.modulus,
                    stage.stage_twiddles,
                ),
                StageWidth::Radix4 => Self::forward_dif_radix4_stage_with_twiddles(
                    values,
                    stage.block_len,
                    stage.modulus,
                    stage.stage_twiddles,
                ),
            }
            return;
        }
        let split_blocks = block_count.div_euclid(2);
        // The validated transform geometry makes this product fit in the slice.
        let split_len = split_blocks.wrapping_mul(stage.block_len);
        let (left, right) = values.split_at_mut(split_len);
        executor.join(
            || Self::apply_forward_stage(left, stage, executor),
            || Self::apply_forward_stage(right, stage, executor),
        );
    }

    fn apply_radix2_stage<E: ParallelExecutor>(
        values: &mut [u32],
        block_len: usize,
        modulus: Modulus,
        stage_twiddles: &[u32],
        policy: NttExecutionPolicy,
        executor: &E,
        inverse: bool,
    ) {
        let half_len = block_len >> 1;
        let block_count = values.len().div_euclid(block_len);
        if policy.should_split(values.len()) && block_count >= 2 {
            let split_len = block_count.div_euclid(2).wrapping_mul(block_len);
            let (left, right) = values.split_at_mut(split_len);
            executor.join(
                || {
                    Self::apply_radix2_stage(
                        left,
                        block_len,
                        modulus,
                        stage_twiddles,
                        policy,
                        executor,
                        inverse,
                    );
                },
                || {
                    Self::apply_radix2_stage(
                        right,
                        block_len,
                        modulus,
                        stage_twiddles,
                        policy,
                        executor,
                        inverse,
                    );
                },
            );
            return;
        }
        if block_count == 1 && policy.should_split(half_len) {
            let (low, high) = values.split_at_mut(half_len);
            Self::apply_radix2_range(
                low,
                high,
                stage_twiddles,
                modulus,
                policy,
                executor,
                inverse,
            );
            return;
        }
        // SAFETY: validated stage geometry gives complete low/high and
        // twiddle spans; the architecture contract covers each direction.
        unsafe {
            if inverse {
                ArchKernels::ntt_dit_butterfly_unchecked(
                    values.as_mut_ptr(),
                    values.as_mut_ptr().add(half_len),
                    stage_twiddles.as_ptr(),
                    half_len,
                    modulus.prime,
                    modulus.neg_inverse,
                );
            } else {
                ArchKernels::ntt_dif_butterfly_unchecked(
                    values.as_mut_ptr(),
                    values.as_mut_ptr().add(half_len),
                    stage_twiddles.as_ptr(),
                    half_len,
                    modulus.prime,
                    modulus.neg_inverse,
                );
            }
        }
    }

    fn apply_radix2_range<E: ParallelExecutor>(
        low: &mut [u32],
        high: &mut [u32],
        stage_twiddles: &[u32],
        modulus: Modulus,
        policy: NttExecutionPolicy,
        executor: &E,
        inverse: bool,
    ) {
        if policy.should_split(low.len()) {
            let midpoint = low.len().div_euclid(2);
            let (low_left, low_right) = low.split_at_mut(midpoint);
            let (high_left, high_right) = high.split_at_mut(midpoint);
            let (twiddle_left, twiddle_right) = stage_twiddles.split_at(midpoint);
            executor.join(
                || {
                    Self::apply_radix2_range(
                        low_left,
                        high_left,
                        twiddle_left,
                        modulus,
                        policy,
                        executor,
                        inverse,
                    );
                },
                || {
                    Self::apply_radix2_range(
                        low_right,
                        high_right,
                        twiddle_right,
                        modulus,
                        policy,
                        executor,
                        inverse,
                    );
                },
            );
            return;
        }
        // SAFETY: low, high, and stage_twiddles are equal-sized disjoint
        // slices produced by the validated range split above.
        unsafe {
            let kernel = if inverse {
                ArchKernels::ntt_dit_butterfly_unchecked
            } else {
                ArchKernels::ntt_dif_butterfly_unchecked
            };
            kernel(
                low.as_mut_ptr(),
                high.as_mut_ptr(),
                stage_twiddles.as_ptr(),
                low.len(),
                modulus.prime,
                modulus.neg_inverse,
            );
        }
    }

    fn inverse_dit_stage_with_executor<E: ParallelExecutor>(
        values: &mut [u32],
        setup: &mut StageSetup<'_>,
        executor: &E,
    ) {
        let half_len = setup.block_len >> 1;
        Self::generate_stage_twiddles(setup.twiddle_buf, half_len, setup.block_root, setup.modulus);
        if matches!(setup.width, StageWidth::Radix4)
            && values.len() == setup.block_len
            && setup.policy.should_split(values.len())
        {
            Self::apply_inverse_radix4_ranges(values, setup, executor);
            return;
        }
        // SAFETY: stage twiddles were generated for exactly half_len entries.
        let stage_twiddles = unsafe { setup.twiddle_buf.get_unchecked(..half_len) };
        let stage = StageParameters {
            block_len: setup.block_len,
            modulus: setup.modulus,
            stage_twiddles,
            width: setup.width,
            policy: setup.policy,
        };
        Self::apply_inverse_stage(values, stage, executor);
    }

    fn apply_inverse_radix4_ranges<E: ParallelExecutor>(
        values: &mut [u32],
        setup: &mut StageSetup<'_>,
        executor: &E,
    ) {
        let half_len = setup.block_len >> 1;
        let quarter_len = half_len >> 1;
        let first_root = Self::montgomery_mul(setup.block_root, setup.block_root, setup.modulus);
        Self::generate_stage_twiddles(setup.twiddle_buf, quarter_len, first_root, setup.modulus);
        // SAFETY: generation initialized the first quarter_len entries.
        let first_twiddles = unsafe { setup.twiddle_buf.get_unchecked(..quarter_len) };
        Self::apply_radix2_stage(
            values,
            half_len,
            setup.modulus,
            first_twiddles,
            setup.policy,
            executor,
            true,
        );
        Self::generate_stage_twiddles(setup.twiddle_buf, half_len, setup.block_root, setup.modulus);
        // SAFETY: generation initialized the first half_len entries.
        let second_twiddles = unsafe { setup.twiddle_buf.get_unchecked(..half_len) };
        Self::apply_radix2_stage(
            values,
            setup.block_len,
            setup.modulus,
            second_twiddles,
            setup.policy,
            executor,
            true,
        );
    }

    fn apply_inverse_stage<E: ParallelExecutor>(
        values: &mut [u32],
        stage: StageParameters<'_>,
        executor: &E,
    ) {
        let block_count = values.len().div_euclid(stage.block_len);
        if !stage.policy.should_split(values.len()) || block_count < 2 {
            match stage.width {
                StageWidth::Radix2 => Self::inverse_dit_stage_with_twiddles(
                    values,
                    stage.block_len,
                    stage.modulus,
                    stage.stage_twiddles,
                ),
                StageWidth::Radix4 => Self::inverse_dit_radix4_stage_with_twiddles(
                    values,
                    stage.block_len,
                    stage.modulus,
                    stage.stage_twiddles,
                ),
            }
            return;
        }
        let split_blocks = block_count.div_euclid(2);
        // The validated transform geometry makes this product fit in the slice.
        let split_len = split_blocks.wrapping_mul(stage.block_len);
        let (left, right) = values.split_at_mut(split_len);
        executor.join(
            || Self::apply_inverse_stage(left, stage, executor),
            || Self::apply_inverse_stage(right, stage, executor),
        );
    }

    pub fn pointwise_monty_mul_with_executor<E: ParallelExecutor>(
        dst: &mut [u32],
        src: &mut [u32],
        modulus: Modulus,
        executor: &E,
    ) {
        let len = dst.len().min(src.len());
        let (dst_prefix, _) = dst.split_at_mut(len);
        let (src_prefix, _) = src.split_at_mut(len);
        let policy = NttExecutionPolicy::for_executor(executor);
        Self::apply_pointwise_mul(dst_prefix, src_prefix, modulus, policy, executor);
    }

    fn apply_pointwise_mul<E: ParallelExecutor>(
        dst: &mut [u32],
        src: &[u32],
        modulus: Modulus,
        policy: NttExecutionPolicy,
        executor: &E,
    ) {
        if !policy.should_split(dst.len()) {
            Self::pointwise_monty_mul_canonical(dst, src, modulus);
            return;
        }
        let midpoint = dst.len().div_euclid(2);
        let (dst_left, dst_right) = dst.split_at_mut(midpoint);
        let (src_left, src_right) = src.split_at(midpoint);
        executor.join(
            || Self::apply_pointwise_mul(dst_left, src_left, modulus, policy, executor),
            || Self::apply_pointwise_mul(dst_right, src_right, modulus, policy, executor),
        );
    }

    pub fn pointwise_monty_sqr_with_executor<E: ParallelExecutor>(
        dst: &mut [u32],
        modulus: Modulus,
        executor: &E,
    ) {
        let policy = NttExecutionPolicy::for_executor(executor);
        Self::apply_pointwise_sqr(dst, modulus, policy, executor);
    }

    fn apply_pointwise_sqr<E: ParallelExecutor>(
        dst: &mut [u32],
        modulus: Modulus,
        policy: NttExecutionPolicy,
        executor: &E,
    ) {
        if !policy.should_split(dst.len()) {
            Self::pointwise_monty_sqr_canonical(dst, modulus);
            return;
        }
        let midpoint = dst.len().div_euclid(2);
        let (left, right) = dst.split_at_mut(midpoint);
        executor.join(
            || Self::apply_pointwise_sqr(left, modulus, policy, executor),
            || Self::apply_pointwise_sqr(right, modulus, policy, executor),
        );
    }

    fn scale_inverse_with_executor<E: ParallelExecutor>(
        values: &mut [u32],
        inverse_len: u32,
        modulus: Modulus,
        policy: NttExecutionPolicy,
        executor: &E,
    ) {
        if !policy.should_split(values.len()) {
            for value in values {
                *value = Self::montgomery_mul(
                    Self::montgomery_mul(*value, inverse_len, modulus),
                    1,
                    modulus,
                );
            }
            return;
        }
        let midpoint = values.len().div_euclid(2);
        let (left, right) = values.split_at_mut(midpoint);
        executor.join(
            || Self::scale_inverse_with_executor(left, inverse_len, modulus, policy, executor),
            || Self::scale_inverse_with_executor(right, inverse_len, modulus, policy, executor),
        );
    }

    /// Forward DIF transform using the supplied stage scheduler.
    pub fn forward_transform_single_with_executor<E: ParallelExecutor>(
        values: &mut [u32],
        modulus: Modulus,
        twiddle_buf: &mut [u32],
        executor: &E,
    ) {
        for value in values.iter_mut() {
            *value = Self::to_montgomery(*value, modulus);
        }
        let root = Self::to_montgomery(modulus.primitive_root, modulus);
        let policy = NttExecutionPolicy::for_executor(executor);
        let mut block_len = values.len();
        while block_len >= 2 {
            let width = if block_len >= 4 {
                StageWidth::Radix4
            } else {
                StageWidth::Radix2
            };
            // SAFETY: block_len is bounded by transform length ≤ 2^28, so the
            // conversion is nonzero and fits in u32.
            let block_len_u32 = unsafe { u32::try_from(block_len).unwrap_unchecked() };
            let exponent = modulus.prime.wrapping_sub(1).div_euclid(block_len_u32);
            let block_root = Self::montgomery_pow(root, exponent, modulus);
            let mut setup = StageSetup {
                block_len,
                block_root,
                modulus,
                twiddle_buf,
                width,
                policy,
            };
            Self::forward_dif_stage_with_executor(values, &mut setup, executor);
            block_len >>= if matches!(width, StageWidth::Radix4) {
                2
            } else {
                1
            };
        }
    }

    /// Forward DIF transform on a pair using the supplied stage scheduler.
    pub fn forward_transform_pair_with_executor<E: ParallelExecutor>(
        left: &mut [u32],
        right: &mut [u32],
        modulus: Modulus,
        twiddle_buf: &mut [u32],
        executor: &E,
    ) {
        for (left_value, right_value) in left.iter_mut().zip(right.iter_mut()) {
            *left_value = Self::to_montgomery(*left_value, modulus);
            *right_value = Self::to_montgomery(*right_value, modulus);
        }
        let root = Self::to_montgomery(modulus.primitive_root, modulus);
        let policy = NttExecutionPolicy::for_executor(executor);
        let mut block_len = left.len();
        while block_len >= 2 {
            let width = if block_len >= 4 {
                StageWidth::Radix4
            } else {
                StageWidth::Radix2
            };
            // SAFETY: block_len is bounded by transform length ≤ 2^28, so the
            // conversion is nonzero and fits in u32.
            let block_len_u32 = unsafe { u32::try_from(block_len).unwrap_unchecked() };
            let exponent = modulus.prime.wrapping_sub(1).div_euclid(block_len_u32);
            let block_root = Self::montgomery_pow(root, exponent, modulus);
            let half_len = block_len >> 1;
            Self::generate_stage_twiddles(twiddle_buf, half_len, block_root, modulus);
            // SAFETY: stage twiddles were generated for exactly half_len entries.
            let stage_twiddles = unsafe { twiddle_buf.get_unchecked(..half_len) };
            let stage = StageParameters {
                block_len,
                modulus,
                stage_twiddles,
                width,
                policy,
            };
            executor.join(
                || Self::apply_forward_stage(left, stage, executor),
                || Self::apply_forward_stage(right, stage, executor),
            );
            block_len >>= if matches!(width, StageWidth::Radix4) {
                2
            } else {
                1
            };
        }
    }

    /// Inverse DIT transform using the supplied stage scheduler.
    pub fn inverse_transform_with_executor<E: ParallelExecutor>(
        values: &mut [u32],
        modulus: Modulus,
        twiddle_buf: &mut [u32],
        executor: &E,
    ) {
        let root = Self::montgomery_pow(
            Self::to_montgomery(modulus.primitive_root, modulus),
            modulus.prime.wrapping_sub(2),
            modulus,
        );
        let policy = NttExecutionPolicy::for_executor(executor);
        let mut block_len = 2;
        while block_len <= values.len() {
            let width = if block_len <= values.len().div_euclid(2) {
                StageWidth::Radix4
            } else {
                StageWidth::Radix2
            };
            let stage_len = if matches!(width, StageWidth::Radix4) {
                block_len.wrapping_mul(2)
            } else {
                block_len
            };
            // SAFETY: stage_len is bounded by transform length ≤ 2^28, so the
            // conversion is nonzero and fits in u32.
            let stage_len_u32 = unsafe { u32::try_from(stage_len).unwrap_unchecked() };
            let exponent = modulus.prime.wrapping_sub(1).div_euclid(stage_len_u32);
            let block_root = Self::montgomery_pow(root, exponent, modulus);
            let mut setup = StageSetup {
                block_len: stage_len,
                block_root,
                modulus,
                twiddle_buf,
                width,
                policy,
            };
            Self::inverse_dit_stage_with_executor(values, &mut setup, executor);
            block_len = if matches!(width, StageWidth::Radix4) {
                block_len.wrapping_mul(4)
            } else {
                block_len.wrapping_mul(2)
            };
        }
        // SAFETY: values.len() ≤ transform length ≤ 2^28, so the conversion
        // is nonzero and fits in u32.
        let values_len_u32 = unsafe { u32::try_from(values.len()).unwrap_unchecked() };
        let inverse_len = Self::montgomery_pow(
            Self::to_montgomery(values_len_u32, modulus),
            modulus.prime.wrapping_sub(2),
            modulus,
        );
        Self::scale_inverse_with_executor(values, inverse_len, modulus, policy, executor);
    }
}
