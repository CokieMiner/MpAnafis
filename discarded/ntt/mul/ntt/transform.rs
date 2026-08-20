//! Forward, inverse, and full 3-prime 50-bit floating-point Harvey NTT multiplication and squaring drivers.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated transform buffers"
)]

use alloc::vec;

use crate::parallel::ParallelExecutor;

use super::{
    ArchKernels, FLOAT_PINV_1, FLOAT_PINV_2, FLOAT_PINV_3, FLOAT_PRIME_1, FLOAT_PRIME_1_INT,
    FLOAT_PRIME_2, FLOAT_PRIME_2_INT, FLOAT_PRIME_3, FLOAT_PRIME_3_INT, FLOAT_ROOT_1, FLOAT_ROOT_2,
    FLOAT_ROOT_3, LIMB_BITS, Limb, Ntt,
};

impl Ntt {
    fn dif_stage_blocks<E: ParallelExecutor>(
        data: &mut [f64],
        stage_twiddles: &[f64],
        quarter: usize,
        step: usize,
        prime: f64,
        pinv: f64,
        executor: &E,
    ) {
        if data.len() <= 16384 || data.len() == step {
            let data_ptr = data.as_mut_ptr();
            let tw_ptr = stage_twiddles.as_ptr();
            let mut offset = 0_usize;
            while offset < data.len() {
                // SAFETY:
                // - `offset + 4 * quarter <= data.len()`.
                // - `stage_twiddles` covers at least `3 * quarter` elements.
                // - `data` and `stage_twiddles` do not alias.
                unsafe {
                    ArchKernels::ntt_float_radix4_dif_unchecked(
                        data_ptr.add(offset),
                        tw_ptr,
                        quarter,
                        prime,
                        pinv,
                    );
                }
                offset = offset.wrapping_add(step);
            }
        } else {
            let step_shift = step.trailing_zeros();
            let num_blocks = data.len() >> step_shift;
            let mid_blocks = num_blocks >> 1;
            let mid = mid_blocks << step_shift;
            let (left, right) = data.split_at_mut(mid);
            executor.join(
                || {
                    Self::dif_stage_blocks(
                        left,
                        stage_twiddles,
                        quarter,
                        step,
                        prime,
                        pinv,
                        executor,
                    );
                },
                || {
                    Self::dif_stage_blocks(
                        right,
                        stage_twiddles,
                        quarter,
                        step,
                        prime,
                        pinv,
                        executor,
                    );
                },
            );
        }
    }

    fn dit_stage_blocks<E: ParallelExecutor>(
        data: &mut [f64],
        stage_twiddles: &[f64],
        quarter: usize,
        step: usize,
        prime: f64,
        pinv: f64,
        executor: &E,
    ) {
        if data.len() <= 16384 || data.len() == step {
            let data_ptr = data.as_mut_ptr();
            let tw_ptr = stage_twiddles.as_ptr();
            let mut offset = 0_usize;
            while offset < data.len() {
                // SAFETY:
                // - `offset + 4 * quarter <= data.len()`.
                // - `stage_twiddles` covers at least `3 * quarter` elements.
                // - `data` and `stage_twiddles` do not alias.
                unsafe {
                    ArchKernels::ntt_float_radix4_dit_unchecked(
                        data_ptr.add(offset),
                        tw_ptr,
                        quarter,
                        prime,
                        pinv,
                    );
                }
                offset = offset.wrapping_add(step);
            }
        } else {
            let step_shift = step.trailing_zeros();
            let num_blocks = data.len() >> step_shift;
            let mid_blocks = num_blocks >> 1;
            let mid = mid_blocks << step_shift;
            let (left, right) = data.split_at_mut(mid);
            executor.join(
                || {
                    Self::dit_stage_blocks(
                        left,
                        stage_twiddles,
                        quarter,
                        step,
                        prime,
                        pinv,
                        executor,
                    );
                },
                || {
                    Self::dit_stage_blocks(
                        right,
                        stage_twiddles,
                        quarter,
                        step,
                        prime,
                        pinv,
                        executor,
                    );
                },
            );
        }
    }

    /// Executes forward Radix-4 / Radix-2 hybrid decimation-in-frequency transform on `data`.
    pub fn forward_transform<E: ParallelExecutor>(
        data: &mut [f64],
        fwd_twiddles: &[f64],
        transform_len: usize,
        prime: f64,
        pinv: f64,
        executor: &E,
    ) {
        let is_odd_power_of_two = !transform_len.trailing_zeros().is_multiple_of(2);
        let mut tw_offset = 0_usize;
        let data_ptr = data.as_mut_ptr();
        let tw_ptr = fwd_twiddles.as_ptr();

        if is_odd_power_of_two {
            let half = transform_len >> 1;
            let mut i = 0_usize;
            while i < half {
                // SAFETY: i + half < transform_len and tw_offset + i < fwd_twiddles.len().
                unsafe {
                    let u = *data_ptr.add(i);
                    let v = *data_ptr.add(i.wrapping_add(half));
                    let w = *tw_ptr.add(tw_offset.wrapping_add(i));

                    let sum = u + v;
                    let q_s = ((sum * pinv) + 6_755_399_441_055_744.0) - 6_755_399_441_055_744.0;
                    *data_ptr.add(i) = sum - q_s * prime;

                    let diff = u - v;
                    *data_ptr.add(i.wrapping_add(half)) = Self::mulmod(diff, w, prime, pinv);
                }
                i = i.wrapping_add(1);
            }
            tw_offset = tw_offset.wrapping_add(half);
        }

        let start_step = if is_odd_power_of_two {
            transform_len >> 1
        } else {
            transform_len
        };
        let mut step = start_step;

        while step >= 4 {
            let quarter = step >> 2;
            let stage_len = quarter.wrapping_mul(3);
            // SAFETY: `tw_offset + stage_len <= fwd_twiddles.len()`.
            let stage_twiddles =
                unsafe { fwd_twiddles.get_unchecked(tw_offset..tw_offset.wrapping_add(stage_len)) };
            Self::dif_stage_blocks(data, stage_twiddles, quarter, step, prime, pinv, executor);
            tw_offset = tw_offset.wrapping_add(stage_len);
            step >>= 2;
        }
    }

    /// Executes inverse Radix-4 / Radix-2 hybrid decimation-in-time transform on `data`.
    pub fn inverse_transform<E: ParallelExecutor>(
        data: &mut [f64],
        inv_twiddles: &[f64],
        transform_len: usize,
        prime: f64,
        pinv: f64,
        executor: &E,
    ) {
        let is_odd_power_of_two = !transform_len.trailing_zeros().is_multiple_of(2);
        let max_r4_step = if is_odd_power_of_two {
            transform_len >> 1
        } else {
            transform_len
        };

        let mut step = 4_usize;
        let mut tw_offset = 0_usize;
        let data_ptr = data.as_mut_ptr();
        let tw_ptr = inv_twiddles.as_ptr();

        while step <= max_r4_step {
            let quarter = step >> 2;
            let stage_len = quarter.wrapping_mul(3);
            // SAFETY: `tw_offset + stage_len <= inv_twiddles.len()`.
            let stage_twiddles =
                unsafe { inv_twiddles.get_unchecked(tw_offset..tw_offset.wrapping_add(stage_len)) };
            Self::dit_stage_blocks(data, stage_twiddles, quarter, step, prime, pinv, executor);
            tw_offset = tw_offset.wrapping_add(stage_len);
            step <<= 2;
        }

        if is_odd_power_of_two {
            let half = transform_len >> 1;
            let mut i = 0_usize;
            while i < half {
                // SAFETY: i + half < transform_len and tw_offset + i < inv_twiddles.len().
                unsafe {
                    let u = *data_ptr.add(i);
                    let v = *data_ptr.add(i.wrapping_add(half));
                    let w = *tw_ptr.add(tw_offset.wrapping_add(i));

                    let v_tw = Self::mulmod(v, w, prime, pinv);

                    let sum = u + v_tw;
                    let q_s = ((sum * pinv) + 6_755_399_441_055_744.0) - 6_755_399_441_055_744.0;
                    *data_ptr.add(i) = sum - q_s * prime;

                    let diff = u - v_tw;
                    let q_d = ((diff * pinv) + 6_755_399_441_055_744.0) - 6_755_399_441_055_744.0;
                    *data_ptr.add(i.wrapping_add(half)) = diff - q_d * prime;
                }
                i = i.wrapping_add(1);
            }
        }
    }

    /// Multiplies two large limb operands using 3-prime 50-bit floating-point NTT with caller-supplied scratch.
    ///
    /// # Performance
    /// Operates completely allocation-free when `scratch` has capacity at least `scratch_len(transform_len)`
    /// (the 3-output, 3-worker, 3-twiddle buffer layout = `9 * transform_len`).
    #[allow(
        clippy::too_many_lines,
        reason = "Explicit 3-way parallel executor transform and CRT dispatch"
    )]
    pub fn try_mul_with_scratch<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [f64],
        executor: &E,
    ) -> bool {
        if a.is_empty() || b.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(dst_len) = a.len().checked_add(b.len()) else {
            return false;
        };
        if dst.len() < dst_len {
            return false;
        }
        let Some(cap_a) = a
            .len()
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.div_ceil(50).checked_add(1))
        else {
            return false;
        };
        let Some(cap_b) = b
            .len()
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.div_ceil(50).checked_add(1))
        else {
            return false;
        };
        let conv_len = cap_a.saturating_add(cap_b).saturating_sub(1);
        let Some(transform_len) = Self::transform_capacity(cap_a, cap_b) else {
            return false;
        };
        let required_len = Self::scratch_len(transform_len);
        if scratch.len() < required_len {
            return false;
        }

        let (workspace, _) = scratch.split_at_mut(required_len);
        let (outputs, rest) = workspace.split_at_mut(transform_len.wrapping_mul(3));
        let (workers, twiddles) = rest.split_at_mut(transform_len.wrapping_mul(3));

        let (out1, out_rest) = outputs.split_at_mut(transform_len);
        let (out2, out3) = out_rest.split_at_mut(transform_len);

        let (work1, work_rest) = workers.split_at_mut(transform_len);
        let (work2, work3) = work_rest.split_at_mut(transform_len);

        let (tw1, tw_rest) = twiddles.split_at_mut(transform_len);
        let (tw2, tw3) = tw_rest.split_at_mut(transform_len);

        // SAFETY: out1 and work1 have capacity transform_len.
        unsafe {
            let _ = Self::limbs_to_digits_50_into(out1, a);
            let _ = Self::limbs_to_digits_50_into(work1, b);
        }
        out2.copy_from_slice(out1);
        out3.copy_from_slice(out1);
        work2.copy_from_slice(work1);
        work3.copy_from_slice(work1);

        executor.join(
            || {
                Self::generate_stage_twiddles(
                    tw1,
                    transform_len,
                    FLOAT_ROOT_1,
                    FLOAT_PRIME_1_INT,
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                    false,
                );
                executor.join(
                    || {
                        Self::forward_transform(
                            out1,
                            tw1,
                            transform_len,
                            FLOAT_PRIME_1,
                            FLOAT_PINV_1,
                            executor,
                        );
                    },
                    || {
                        Self::forward_transform(
                            work1,
                            tw1,
                            transform_len,
                            FLOAT_PRIME_1,
                            FLOAT_PINV_1,
                            executor,
                        );
                    },
                );
                // SAFETY: out1 and work1 have capacity transform_len.
                unsafe {
                    ArchKernels::ntt_float_pointwise_mul_unchecked(
                        out1.as_mut_ptr(),
                        work1.as_ptr(),
                        transform_len,
                        FLOAT_PRIME_1,
                        FLOAT_PINV_1,
                    );
                }
                Self::generate_stage_twiddles(
                    tw1,
                    transform_len,
                    FLOAT_ROOT_1,
                    FLOAT_PRIME_1_INT,
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                    true,
                );
                Self::inverse_transform(
                    out1,
                    tw1,
                    transform_len,
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                    executor,
                );
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_precision_loss,
                    reason = "transform_len fits within f64 mantissa"
                )]
                let inv_n_1 = Self::pow_mod_float(
                    transform_len as f64,
                    FLOAT_PRIME_1_INT.wrapping_sub(2),
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                );
                // SAFETY: out1 has capacity transform_len >= conv_len.
                unsafe {
                    ArchKernels::ntt_float_scale_unchecked(
                        out1.as_mut_ptr(),
                        conv_len,
                        inv_n_1,
                        FLOAT_PRIME_1,
                        FLOAT_PINV_1,
                    );
                }
            },
            || {
                executor.join(
                    || {
                        Self::generate_stage_twiddles(
                            tw2,
                            transform_len,
                            FLOAT_ROOT_2,
                            FLOAT_PRIME_2_INT,
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                            false,
                        );
                        executor.join(
                            || {
                                Self::forward_transform(
                                    out2,
                                    tw2,
                                    transform_len,
                                    FLOAT_PRIME_2,
                                    FLOAT_PINV_2,
                                    executor,
                                );
                            },
                            || {
                                Self::forward_transform(
                                    work2,
                                    tw2,
                                    transform_len,
                                    FLOAT_PRIME_2,
                                    FLOAT_PINV_2,
                                    executor,
                                );
                            },
                        );
                        // SAFETY: out2 and work2 have capacity transform_len.
                        unsafe {
                            ArchKernels::ntt_float_pointwise_mul_unchecked(
                                out2.as_mut_ptr(),
                                work2.as_ptr(),
                                transform_len,
                                FLOAT_PRIME_2,
                                FLOAT_PINV_2,
                            );
                        }
                        Self::generate_stage_twiddles(
                            tw2,
                            transform_len,
                            FLOAT_ROOT_2,
                            FLOAT_PRIME_2_INT,
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                            true,
                        );
                        Self::inverse_transform(
                            out2,
                            tw2,
                            transform_len,
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                            executor,
                        );
                        #[allow(
                            clippy::as_conversions,
                            clippy::cast_precision_loss,
                            reason = "transform_len fits within f64 mantissa"
                        )]
                        let inv_n_2 = Self::pow_mod_float(
                            transform_len as f64,
                            FLOAT_PRIME_2_INT.wrapping_sub(2),
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                        );
                        // SAFETY: out2 has capacity transform_len >= conv_len.
                        unsafe {
                            ArchKernels::ntt_float_scale_unchecked(
                                out2.as_mut_ptr(),
                                conv_len,
                                inv_n_2,
                                FLOAT_PRIME_2,
                                FLOAT_PINV_2,
                            );
                        }
                    },
                    || {
                        Self::generate_stage_twiddles(
                            tw3,
                            transform_len,
                            FLOAT_ROOT_3,
                            FLOAT_PRIME_3_INT,
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                            false,
                        );
                        executor.join(
                            || {
                                Self::forward_transform(
                                    out3,
                                    tw3,
                                    transform_len,
                                    FLOAT_PRIME_3,
                                    FLOAT_PINV_3,
                                    executor,
                                );
                            },
                            || {
                                Self::forward_transform(
                                    work3,
                                    tw3,
                                    transform_len,
                                    FLOAT_PRIME_3,
                                    FLOAT_PINV_3,
                                    executor,
                                );
                            },
                        );
                        // SAFETY: out3 and work3 have capacity transform_len.
                        unsafe {
                            ArchKernels::ntt_float_pointwise_mul_unchecked(
                                out3.as_mut_ptr(),
                                work3.as_ptr(),
                                transform_len,
                                FLOAT_PRIME_3,
                                FLOAT_PINV_3,
                            );
                        }
                        Self::generate_stage_twiddles(
                            tw3,
                            transform_len,
                            FLOAT_ROOT_3,
                            FLOAT_PRIME_3_INT,
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                            true,
                        );
                        Self::inverse_transform(
                            out3,
                            tw3,
                            transform_len,
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                            executor,
                        );
                        #[allow(
                            clippy::as_conversions,
                            clippy::cast_precision_loss,
                            reason = "transform_len fits within f64 mantissa"
                        )]
                        let inv_n_3 = Self::pow_mod_float(
                            transform_len as f64,
                            FLOAT_PRIME_3_INT.wrapping_sub(2),
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                        );
                        // SAFETY: out3 has capacity transform_len >= conv_len.
                        unsafe {
                            ArchKernels::ntt_float_scale_unchecked(
                                out3.as_mut_ptr(),
                                conv_len,
                                inv_n_3,
                                FLOAT_PRIME_3,
                                FLOAT_PINV_3,
                            );
                        }
                    },
                );
            },
        );

        // SAFETY: Garner CRT reconstruction converts residues in [0, P) into destination limbs.
        unsafe {
            Self::reconstruct_into_limbs(dst, out1, out2, out3, conv_len, 50);
        }
        true
    }

    /// Squares a large limb operand using 3-prime 50-bit floating-point NTT with caller-supplied scratch.
    ///
    /// # Performance
    /// Requires only 3 forward transforms (instead of 6) and needs only `scratch_sqr_len(transform_len)`
    /// (`6 * transform_len`) workspace elements.
    #[allow(
        clippy::too_many_lines,
        reason = "Explicit 3-way parallel executor transform and CRT dispatch"
    )]
    pub fn try_sqr_with_scratch<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [f64],
        executor: &E,
    ) -> bool {
        if a.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(dst_len) = a.len().checked_mul(2) else {
            return false;
        };
        if dst.len() < dst_len {
            return false;
        }
        let Some(cap_a) = a
            .len()
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.div_ceil(50).checked_add(1))
        else {
            return false;
        };
        let conv_len = cap_a.saturating_mul(2).saturating_sub(1);
        let Some(transform_len) = Self::transform_capacity(cap_a, cap_a) else {
            return false;
        };
        let required_len = Self::scratch_sqr_len(transform_len);
        if scratch.len() < required_len {
            return false;
        }

        let (workspace, _) = scratch.split_at_mut(required_len);
        let (outputs, twiddles) = workspace.split_at_mut(transform_len.wrapping_mul(3));

        let (out1, out_rest) = outputs.split_at_mut(transform_len);
        let (out2, out3) = out_rest.split_at_mut(transform_len);

        let (tw1, tw_rest) = twiddles.split_at_mut(transform_len);
        let (tw2, tw3) = tw_rest.split_at_mut(transform_len);

        // SAFETY: out1 has capacity transform_len.
        unsafe {
            let _ = Self::limbs_to_digits_50_into(out1, a);
        }
        out2.copy_from_slice(out1);
        out3.copy_from_slice(out1);

        executor.join(
            || {
                Self::generate_stage_twiddles(
                    tw1,
                    transform_len,
                    FLOAT_ROOT_1,
                    FLOAT_PRIME_1_INT,
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                    false,
                );
                Self::forward_transform(
                    out1,
                    tw1,
                    transform_len,
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                    executor,
                );
                // SAFETY: out1 has capacity transform_len.
                unsafe {
                    ArchKernels::ntt_float_pointwise_sqr_unchecked(
                        out1.as_mut_ptr(),
                        transform_len,
                        FLOAT_PRIME_1,
                        FLOAT_PINV_1,
                    );
                }
                Self::generate_stage_twiddles(
                    tw1,
                    transform_len,
                    FLOAT_ROOT_1,
                    FLOAT_PRIME_1_INT,
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                    true,
                );
                Self::inverse_transform(
                    out1,
                    tw1,
                    transform_len,
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                    executor,
                );
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_precision_loss,
                    reason = "transform_len fits within f64 mantissa"
                )]
                let inv_n_1 = Self::pow_mod_float(
                    transform_len as f64,
                    FLOAT_PRIME_1_INT.wrapping_sub(2),
                    FLOAT_PRIME_1,
                    FLOAT_PINV_1,
                );
                // SAFETY: out1 has capacity transform_len >= conv_len.
                unsafe {
                    ArchKernels::ntt_float_scale_unchecked(
                        out1.as_mut_ptr(),
                        conv_len,
                        inv_n_1,
                        FLOAT_PRIME_1,
                        FLOAT_PINV_1,
                    );
                }
            },
            || {
                executor.join(
                    || {
                        Self::generate_stage_twiddles(
                            tw2,
                            transform_len,
                            FLOAT_ROOT_2,
                            FLOAT_PRIME_2_INT,
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                            false,
                        );
                        Self::forward_transform(
                            out2,
                            tw2,
                            transform_len,
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                            executor,
                        );
                        // SAFETY: out2 has capacity transform_len.
                        unsafe {
                            ArchKernels::ntt_float_pointwise_sqr_unchecked(
                                out2.as_mut_ptr(),
                                transform_len,
                                FLOAT_PRIME_2,
                                FLOAT_PINV_2,
                            );
                        }
                        Self::generate_stage_twiddles(
                            tw2,
                            transform_len,
                            FLOAT_ROOT_2,
                            FLOAT_PRIME_2_INT,
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                            true,
                        );
                        Self::inverse_transform(
                            out2,
                            tw2,
                            transform_len,
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                            executor,
                        );
                        #[allow(
                            clippy::as_conversions,
                            clippy::cast_precision_loss,
                            reason = "transform_len fits within f64 mantissa"
                        )]
                        let inv_n_2 = Self::pow_mod_float(
                            transform_len as f64,
                            FLOAT_PRIME_2_INT.wrapping_sub(2),
                            FLOAT_PRIME_2,
                            FLOAT_PINV_2,
                        );
                        // SAFETY: out2 has capacity transform_len >= conv_len.
                        unsafe {
                            ArchKernels::ntt_float_scale_unchecked(
                                out2.as_mut_ptr(),
                                conv_len,
                                inv_n_2,
                                FLOAT_PRIME_2,
                                FLOAT_PINV_2,
                            );
                        }
                    },
                    || {
                        Self::generate_stage_twiddles(
                            tw3,
                            transform_len,
                            FLOAT_ROOT_3,
                            FLOAT_PRIME_3_INT,
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                            false,
                        );
                        Self::forward_transform(
                            out3,
                            tw3,
                            transform_len,
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                            executor,
                        );
                        // SAFETY: out3 has capacity transform_len.
                        unsafe {
                            ArchKernels::ntt_float_pointwise_sqr_unchecked(
                                out3.as_mut_ptr(),
                                transform_len,
                                FLOAT_PRIME_3,
                                FLOAT_PINV_3,
                            );
                        }
                        Self::generate_stage_twiddles(
                            tw3,
                            transform_len,
                            FLOAT_ROOT_3,
                            FLOAT_PRIME_3_INT,
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                            true,
                        );
                        Self::inverse_transform(
                            out3,
                            tw3,
                            transform_len,
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                            executor,
                        );
                        #[allow(
                            clippy::as_conversions,
                            clippy::cast_precision_loss,
                            reason = "transform_len fits within f64 mantissa"
                        )]
                        let inv_n_3 = Self::pow_mod_float(
                            transform_len as f64,
                            FLOAT_PRIME_3_INT.wrapping_sub(2),
                            FLOAT_PRIME_3,
                            FLOAT_PINV_3,
                        );
                        // SAFETY: out3 has capacity transform_len >= conv_len.
                        unsafe {
                            ArchKernels::ntt_float_scale_unchecked(
                                out3.as_mut_ptr(),
                                conv_len,
                                inv_n_3,
                                FLOAT_PRIME_3,
                                FLOAT_PINV_3,
                            );
                        }
                    },
                );
            },
        );

        // SAFETY: Garner CRT reconstruction converts residues in [0, P) into destination limbs.
        unsafe {
            Self::reconstruct_into_limbs(dst, out1, out2, out3, conv_len, 50);
        }
        true
    }

    /// Multiplies two large limb operands using 3-prime 50-bit floating-point NTT with optional caller scratch.
    pub fn try_mul_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: Option<&mut [Limb]>,
        executor: &E,
    ) -> bool {
        if a.is_empty() || b.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(cap_a) = a
            .len()
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.div_ceil(50).checked_add(1))
        else {
            return false;
        };
        let Some(cap_b) = b
            .len()
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.div_ceil(50).checked_add(1))
        else {
            return false;
        };
        let Some(transform_len) = Self::transform_capacity(cap_a, cap_b) else {
            return false;
        };
        let required_len = Self::scratch_len(transform_len);
        if let Some(scratch_limbs) = scratch
            && let Some(f64_slice) = Self::align_scratch_limbs_to_f64(scratch_limbs, required_len)
        {
            return Self::try_mul_with_scratch(dst, a, b, f64_slice, executor);
        }
        let mut workspace = vec![0.0_f64; required_len];
        Self::try_mul_with_scratch(dst, a, b, &mut workspace, executor)
    }

    /// Squares a large limb operand using 3-prime 50-bit floating-point NTT with optional caller scratch.
    pub fn try_sqr_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        scratch: Option<&mut [Limb]>,
        executor: &E,
    ) -> bool {
        if a.is_empty() {
            dst.fill(0);
            return true;
        }
        let Some(cap_a) = a
            .len()
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.div_ceil(50).checked_add(1))
        else {
            return false;
        };
        let Some(transform_len) = Self::transform_capacity(cap_a, cap_a) else {
            return false;
        };
        let required_len = Self::scratch_sqr_len(transform_len);
        if let Some(scratch_limbs) = scratch
            && let Some(f64_slice) = Self::align_scratch_limbs_to_f64(scratch_limbs, required_len)
        {
            return Self::try_sqr_with_scratch(dst, a, f64_slice, executor);
        }
        let mut workspace = vec![0.0_f64; required_len];
        Self::try_sqr_with_scratch(dst, a, &mut workspace, executor)
    }
}
