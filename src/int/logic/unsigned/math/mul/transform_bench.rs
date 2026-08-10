//! SSA forced-transform benchmark support.
//!
//! Everything here exists so a measurement can reach a configuration production
//! never picks: a transform on a ring the planner would leave to the basecase, a
//! pinned exponent, or one Fermat product with the CRT split taken out of the way.
//! Keeping it on `TransformBench` rather than on [`Ssa`](super::Ssa) stops the
//! production namespace from carrying entry points the tower cannot reach.
//!
//! These forward to the *same* implementations production uses, so a forced
//! measurement times the real kernel rather than a parallel copy of it. They
//! allocate their own scratch, which is why none of them belongs on a hot path.
//!
//! The namespace is feature-gated at the multiplication boundary and is
//! consumed only by the raw benchmark facade. Stateful crossover measurement
//! remains on the reusable `Tuner` types in `tune_api`.
//! None of these methods participates in production tier selection.
//! The root module exposes it only under the internal tuning configuration.

#![allow(
    unsafe_code,
    reason = "Benchmark entry points forward into validated SSA transform kernels"
)]

use alloc::vec::Vec;

use super::{FftPlan, LIMB_BITS, Limb, SsaCrt, SsaPlan, SsaRing, SsaTransform};

/// Namespace for forced transform benchmark entry points.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformBench;

impl TransformBench {
    /// Collect the exact inner ring widths visited by the planner up to a limit.
    pub fn collect_ssa_inner_rings(max_modulus_bits: usize) -> Vec<usize> {
        let mut widths = Vec::new();
        let mut current = 256;
        while current <= max_modulus_bits {
            let plan = SsaPlan::best_exponent(current, 0);
            if let Some((_, geometry)) = plan {
                let inner = geometry.inner_bits;
                if !widths.contains(&inner) {
                    widths.push(inner);
                }
            }
            current = current.saturating_add(256);
        }
        widths.sort_unstable();
        widths
    }

    /// Scratch required when forcing a specific transform exponent.
    ///
    /// Returns `None` if the requested geometry is invalid for these operand sizes.
    pub fn ssa_mul_scratch_len_for_plan(
        len_a: usize,
        len_b: usize,
        transform_exponent: u32,
    ) -> Option<usize> {
        let half_width = SsaPlan::crt_half_width_for_operands(len_a, len_b)?;
        let ring_bits = half_width.checked_mul(LIMB_BITS)?;
        let ring_scratch =
            FftPlan::try_forced(ring_bits, transform_exponent)?.transform_mul_scratch();
        Some(SsaCrt::layout_len(half_width, ring_scratch))
    }

    /// The top-level `modulus_bits` used for SSA multiplication of the given lengths.
    pub fn ssa_mul_modulus_bits(len_a: usize, len_b: usize) -> Option<usize> {
        let half_width = SsaPlan::crt_half_width_for_operands(len_a, len_b)?;
        let ring_bits = half_width.checked_mul(LIMB_BITS)?;
        Some(ring_bits)
    }

    /// The top-level `modulus_bits` used for SSA squaring of the given length.
    pub fn ssa_sqr_modulus_bits(len: usize) -> Option<usize> {
        let half_width = SsaPlan::crt_half_width_for_operands(len, len)?;
        let ring_bits = half_width.checked_mul(LIMB_BITS)?;
        Some(ring_bits)
    }

    /// Multiply two full-width residues modulo `2^(len * LIMB_BITS) + 1`.
    ///
    /// `transform_exponent` selects between the planner's geometry and a forced one;
    /// validation, scratch sizing, padding, and the call into the transform are
    /// identical either way. Returns `false` when the operand widths do not form such
    /// a ring, or when a forced exponent yields no usable geometry.
    pub fn ssa_fermat_mul(
        dst: &mut [Limb],
        left: &[Limb],
        right: &[Limb],
        transform_exponent: Option<u32>,
    ) -> bool {
        let Some(expected_dst_len) = left.len().checked_add(1) else {
            return false;
        };
        if left.len() != right.len() || dst.len() != expected_dst_len {
            return false;
        }
        let n = left.len();
        if n == 0 {
            // SAFETY: dst.len() == n + 1 == 1 on the zero-width path.
            *unsafe { dst.get_unchecked_mut(0) } = 0;
            return true;
        }
        let Some(modulus_bits) = n.checked_mul(LIMB_BITS) else {
            return false;
        };
        if !modulus_bits.is_power_of_two() || modulus_bits < LIMB_BITS {
            return false;
        }

        let ml = SsaRing::mod_limbs(modulus_bits);
        let cl = SsaRing::coeff_limbs(modulus_bits);

        // A forced transform always runs, so the transform layout for that specific
        // exponent is what has to fit; the planned path sizes from its own choice.
        let forced_plan = match transform_exponent {
            Some(exponent) => match FftPlan::try_forced(modulus_bits, exponent) {
                Some(plan) => Some(plan),
                None => return false,
            },
            None => None,
        };
        let scratch_size = forced_plan.as_ref().map_or_else(
            || FftPlan::new(modulus_bits).required_mul_scratch(),
            FftPlan::transform_mul_scratch,
        );

        let Some(total_scratch) = cl
            .checked_mul(2)
            .and_then(|padded_operands| padded_operands.checked_add(scratch_size))
        else {
            return false;
        };
        let mut scratch = vec![0; total_scratch];

        let (left_padded, after_left) = scratch.split_at_mut(cl);
        let (right_padded, ring_scratch) = after_left.split_at_mut(cl);

        // SAFETY: `modulus_bits == n * LIMB_BITS` gives `ml == n == left.len()`;
        // `cl == ml + 1`, so this destination prefix has the same exact width.
        unsafe { left_padded.get_unchecked_mut(..ml) }.copy_from_slice(left);
        // SAFETY: `left_padded.len() == cl == ml + 1`, hence `ml < len`.
        *unsafe { left_padded.get_unchecked_mut(ml) } = 0;
        // SAFETY: `right.len() == left.len() == ml` and the second padded
        // buffer also has exactly `cl == ml + 1` limbs.
        unsafe { right_padded.get_unchecked_mut(..ml) }.copy_from_slice(right);
        // SAFETY: `right_padded.len() == ml + 1`, hence `ml < len`.
        *unsafe { right_padded.get_unchecked_mut(ml) } = 0;

        let mut product = vec![0; cl];
        // SAFETY: every buffer is exactly `cl` limbs, the ring scratch is sized from
        // the same geometry that is passed in, and a forced plan was built for this
        // exact `modulus_bits` above.
        unsafe {
            SsaTransform::fft_mul_mod_slices(
                &mut product,
                left_padded,
                right_padded,
                modulus_bits,
                None,
                forced_plan.is_some(),
                forced_plan.as_ref(),
                ring_scratch,
            );
        }
        dst.copy_from_slice(&product);
        true
    }

    /// Scratch one [`Self::ssa_mul_mod_bnm1`] call on `n`-limb operands needs.
    pub fn ssa_mul_mod_bnm1_scratch_len(n: usize) -> usize {
        SsaCrt::mul_mod_bnm1_scratch_len(n)
    }

    /// The `B^n - 1` half of the top-level CRT split, on its own.
    ///
    /// Production only ever reaches this from inside [`Ssa::try_mul`](super::entry::Ssa::try_mul), where its
    /// cost is entangled with the Fermat half beside it. Exposing it lets the tuner
    /// measure the Mersenne recursion's own crossover.
    pub fn ssa_mul_mod_bnm1(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        SsaCrt::mul_mod_bnm1(dst, a, b, scratch);
    }
}
