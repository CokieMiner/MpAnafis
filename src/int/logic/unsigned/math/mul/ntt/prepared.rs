//! Operand-bound NTT product planning and reusable workspace execution.

use alloc::vec;

use crate::parallel::ParallelExecutor;

use super::{
    GoldilocksProduct, LIMB_BITS, Limb, Ntt, PrimeWorkspace, TransformPlan, digit_capacity,
};

/// Operand-bound NTT product plan with all fallible geometry work completed.
///
/// The immutable operand borrow prevents a caller from substituting a shape or
/// significant width after digit counts, CRT range, and workspace offsets have
/// been proved. Execution can therefore reuse exact workspace without plan
/// selection, allocation, or fallible branches.
#[derive(Debug)]
pub struct NttMultiplicationPlan<'operands> {
    a_limbs: &'operands [Limb],
    b_limbs: &'operands [Limb],
    destination_len: usize,
    product: Option<NttProductPlan>,
}

#[derive(Clone, Copy, Debug)]
struct NttProductPlan {
    transform: TransformPlan,
    active_left_len: usize,
    active_right_len: usize,
    left_capacity: usize,
    right_capacity: usize,
    left_digit_len: usize,
    right_digit_len: usize,
    convolution_len: usize,
    transform_len: usize,
    prime_workspace_len: usize,
    twiddle_workspace_len: usize,
    scratch_u32_len: usize,
    scratch_u64_len: usize,
}

impl<'operands> NttMultiplicationPlan<'operands> {
    /// Builds one exact product plan for immutable operands.
    ///
    /// Returns `None` when the transform parameters are invalid, a dimension
    /// overflows, the fixed roots are too short, or the selected CRT range
    /// cannot represent every convolution coefficient.
    pub fn try_new(
        a_limbs: &'operands [Limb],
        b_limbs: &'operands [Limb],
        transform: TransformPlan,
    ) -> Option<Self> {
        if !transform.is_valid() {
            return None;
        }
        let destination_len = a_limbs.len().checked_add(b_limbs.len())?;
        let Some(left_width) = significant_width(a_limbs) else {
            return Some(Self {
                a_limbs,
                b_limbs,
                destination_len,
                product: None,
            });
        };
        let Some(right_width) = significant_width(b_limbs) else {
            return Some(Self {
                a_limbs,
                b_limbs,
                destination_len,
                product: None,
            });
        };

        let digit_width = usize::try_from(transform.digit_bits).ok()?;
        let left_digit_len = left_width.bits.div_ceil(digit_width);
        let right_digit_len = right_width.bits.div_ceil(digit_width);
        let left_capacity = digit_capacity(left_width.limb_len, transform.digit_bits)?;
        let right_capacity = digit_capacity(right_width.limb_len, transform.digit_bits)?;
        let convolution_len = left_digit_len
            .checked_add(right_digit_len)?
            .checked_sub(1)?;
        let transform_len = convolution_len.checked_next_power_of_two()?;
        if Ntt::MAX_TRANSFORM_LEN.is_some_and(|maximum| transform_len > maximum)
            || !Ntt::coefficient_range_fits(
                transform_len,
                transform.digit_bits,
                transform.modulus_count,
            )
        {
            return None;
        }

        let prime_workspace_len = transform_len.checked_mul(transform.modulus_count)?;
        let twiddle_workspace_len = transform_len
            .div_ceil(2)
            .checked_mul(transform.modulus_count)?;
        let transform_workspace_len = if transform.modulus_count == 1 {
            transform_len
        } else {
            prime_workspace_len
                .checked_mul(2)?
                .checked_add(twiddle_workspace_len)?
                .checked_add(transform_len)?
        };
        let scratch_u32_len = left_capacity
            .checked_add(right_capacity)?
            .checked_add(transform_workspace_len)?;
        let scratch_u64_len = if transform.modulus_count == 1 {
            transform_len.checked_mul(3)?
        } else {
            0
        };

        Some(Self {
            a_limbs,
            b_limbs,
            destination_len,
            product: Some(NttProductPlan {
                transform,
                active_left_len: left_width.limb_len,
                active_right_len: right_width.limb_len,
                left_capacity,
                right_capacity,
                left_digit_len,
                right_digit_len,
                convolution_len,
                transform_len,
                prime_workspace_len,
                twiddle_workspace_len,
                scratch_u32_len,
                scratch_u64_len,
            }),
        })
    }

    /// Exact destination width required by this product.
    #[must_use]
    pub const fn destination_len(&self) -> usize {
        self.destination_len
    }

    /// Exact reusable 32-bit workspace width required by this product.
    #[must_use]
    pub const fn scratch_u32_len(&self) -> usize {
        match self.product {
            Some(product) => product.scratch_u32_len,
            None => 0,
        }
    }

    /// Exact reusable Goldilocks workspace width required by this product.
    #[must_use]
    pub const fn scratch_u64_len(&self) -> usize {
        match self.product {
            Some(product) => product.scratch_u64_len,
            None => 0,
        }
    }

    /// Executes with fresh zeroed workspaces owned by this call.
    ///
    /// # Safety
    ///
    /// `dst` must contain at least [`Self::destination_len`] limbs.
    pub unsafe fn run_allocating<E: ParallelExecutor>(&self, dst: &mut [Limb], executor: &E) {
        let mut scratch_u32 = vec![0; self.scratch_u32_len()];
        let mut scratch_u64 = vec![0; self.scratch_u64_len()];
        // SAFETY: both freshly allocated buffers have the exact planned widths;
        // the caller supplies the remaining destination invariant.
        unsafe { self.run_with_scratch(dst, &mut scratch_u32, &mut scratch_u64, executor) }
    }

    /// Executes with caller-owned workspaces sized before timing.
    ///
    /// # Safety
    ///
    /// `dst` must contain at least [`Self::destination_len`] limbs, while both
    /// scratch spans must contain at least their corresponding planned widths.
    #[allow(
        clippy::too_many_lines,
        reason = "digit conversion, the two modulus families, and final packing form one prepared execution boundary"
    )]
    pub unsafe fn run_with_scratch<E: ParallelExecutor>(
        &self,
        dst: &mut [Limb],
        scratch_u32: &mut [u32],
        scratch_u64: &mut [u64],
        executor: &E,
    ) {
        let Some(product) = self.product else {
            dst.fill(0);
            return;
        };
        let NttProductPlan {
            transform,
            active_left_len,
            active_right_len,
            left_capacity,
            right_capacity,
            left_digit_len,
            right_digit_len,
            convolution_len,
            transform_len,
            prime_workspace_len,
            twiddle_workspace_len,
            ..
        } = product;

        // SAFETY: the caller supplies at least the exact prepared workspace.
        let planned_u32 = unsafe { scratch_u32.get_unchecked_mut(..product.scratch_u32_len) };
        // SAFETY: the identical contract applies to the Goldilocks workspace.
        let planned_u64 = unsafe { scratch_u64.get_unchecked_mut(..product.scratch_u64_len) };
        let (left_buffer, scratch_tail) = planned_u32.split_at_mut(left_capacity);
        let (right_buffer, transform_workspace) = scratch_tail.split_at_mut(right_capacity);
        // SAFETY: construction derived each active prefix from the borrowed
        // operand's highest nonzero limb.
        let active_left = unsafe { self.a_limbs.get_unchecked(..active_left_len) };
        // SAFETY: the same significant-width proof applies to the right operand.
        let active_right = unsafe { self.b_limbs.get_unchecked(..active_right_len) };
        // SAFETY: both conversion buffers cover every digit emitted by their
        // complete active limb prefixes.
        let _ =
            unsafe { Ntt::limbs_to_digits_into(left_buffer, active_left, transform.digit_bits) };
        // SAFETY: the right conversion buffer has the separately proved capacity.
        let _ =
            unsafe { Ntt::limbs_to_digits_into(right_buffer, active_right, transform.digit_bits) };
        // SAFETY: significant-width planning proves these exact nonzero prefixes.
        let left_digits = unsafe { left_buffer.get_unchecked(..left_digit_len) };
        // SAFETY: the corresponding right prefix is equally bounded.
        let right_digits = unsafe { right_buffer.get_unchecked(..right_digit_len) };

        if transform.modulus_count == 1 {
            // SAFETY: the one-prime layout reserves one complete output transform.
            let single_output = unsafe { transform_workspace.get_unchecked_mut(..transform_len) };
            let product_state = GoldilocksProduct::new(
                single_output,
                left_digits,
                right_digits,
                convolution_len,
                transform.digit_bits,
                planned_u64,
                executor,
            );
            let count = product_state.run(transform_len);
            // SAFETY: Goldilocks digit reconstruction writes at most its output span.
            let valid_digits = unsafe { single_output.get_unchecked(..count) };
            // SAFETY: exact product bits fit the prepared destination width.
            unsafe { Ntt::digits_to_limbs(dst, valid_digits, transform.digit_bits) }
            return;
        }

        let (outputs, worker_tail) = transform_workspace.split_at_mut(prime_workspace_len);
        let (workers, twiddle_tail) = worker_tail.split_at_mut(prime_workspace_len);
        let (twiddles, digits_out) = twiddle_tail.split_at_mut(twiddle_workspace_len);
        let mut workspace = PrimeWorkspace::new(
            outputs,
            workers,
            twiddles,
            digits_out,
            transform_len,
            transform.modulus_count,
        );
        workspace.multiply(left_digits, right_digits, executor);
        let valid_digits = if transform.modulus_count == 2 {
            workspace.reconstruct_two(convolution_len, transform.digit_bits)
        } else {
            workspace.reconstruct_three(convolution_len, transform.digit_bits)
        };
        // SAFETY: exact CRT reconstruction fits the prepared destination width.
        unsafe { Ntt::digits_to_limbs(dst, valid_digits, transform.digit_bits) }
    }
}

#[derive(Clone, Copy, Debug)]
struct SignificantWidth {
    limb_len: usize,
    bits: usize,
}

fn significant_width(limbs: &[Limb]) -> Option<SignificantWidth> {
    let top_index = limbs.iter().rposition(|&limb| limb != 0)?;
    let top_zero_bits = usize::try_from(limbs.get(top_index)?.leading_zeros()).ok()?;
    let top_bits = LIMB_BITS.checked_sub(top_zero_bits)?;
    let active_len = top_index.checked_add(1)?;
    let significant_bits = top_index.checked_mul(LIMB_BITS)?.checked_add(top_bits)?;
    Some(SignificantWidth {
        limb_len: active_len,
        bits: significant_bits,
    })
}
