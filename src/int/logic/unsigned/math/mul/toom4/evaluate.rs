//! Four-way Toom-Cook splitting and denominator-scaled evaluation.
//!
//! The tier driver admits four nonempty parts before evaluating. The first
//! three parts have the complete split width; the high part may be shorter.
//! Every evaluation destination retains one guard above that width.

#![expect(
    unsafe_code,
    reason = "Admitted four-part operands bound all evaluation spans; weights sum to fifteen and retain a bounded guard"
)]

use core::ptr::copy_nonoverlapping;

use super::{AddMulKernel, DoubleLimb, LIMB_BITS, Limb, SharedEval, Toom4};

impl Toom4 {
    /// Evaluate the denominator-scaled value `8*A(1/2)` in one limb pass.
    pub fn evaluate_half_scaled(
        dst: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        part3: &[Limb],
    ) {
        debug_assert!(dst.len() > 1, "evaluation includes a body and guard");
        // SAFETY: split_scratch supplies split_len + 1 initialized limbs.
        let (guard, body) = unsafe { dst.split_last_mut().unwrap_unchecked() };
        debug_assert!(
            part0.len() == body.len()
                && part1.len() == body.len()
                && part2.len() == body.len()
                && part3.len() <= body.len(),
            "the admitted four-part split has three full low blocks"
        );

        let prefix_len = part3.len();
        // SAFETY: four-part admission gives |part0|=|part1|=|part2|=|body|=m
        // and |part3|<=m, so the common prefix lies within all four slices.
        let (
            (body_prefix, body_suffix),
            (part0_prefix, part0_suffix),
            (part1_prefix, part1_suffix),
            (part2_prefix, part2_suffix),
        ) = unsafe {
            (
                body.split_at_mut_unchecked(prefix_len),
                part0.split_at_unchecked(prefix_len),
                part1.split_at_unchecked(prefix_len),
                part2.split_at_unchecked(prefix_len),
            )
        };
        let mut carry = 0;
        for ((((dst_limb, part0_limb), part1_limb), part2_limb), part3_limb) in body_prefix
            .iter_mut()
            .zip(part0_prefix)
            .zip(part1_prefix)
            .zip(part2_prefix)
            .zip(part3)
        {
            *dst_limb = Self::evaluate_at_two_limb(
                *part3_limb,
                *part2_limb,
                *part1_limb,
                *part0_limb,
                &mut carry,
            );
        }
        for (((dst_limb, part0_limb), part1_limb), part2_limb) in body_suffix
            .iter_mut()
            .zip(part0_suffix)
            .zip(part1_suffix)
            .zip(part2_suffix)
        {
            *dst_limb =
                Self::evaluate_at_two_limb(0, *part2_limb, *part1_limb, *part0_limb, &mut carry);
        }
        // 8*a0+4*a1+2*a2+a3 < 15*B^m, so the guard is at most fourteen.
        *guard = carry;
    }

    /// Evaluate `8*A(1/2)` through the selected scalar add-multiply backend.
    pub fn evaluate_half_scaled_with_kernel(
        dst: &mut [Limb],
        blocks: [&[Limb]; 4],
        kernel: AddMulKernel,
    ) {
        let [part0, part1, part2, part3] = blocks;
        debug_assert!(dst.len() > 1, "evaluation includes a body and guard");
        // SAFETY: split_scratch supplies split_len + 1 initialized limbs.
        let (guard, body) = unsafe { dst.split_last_mut().unwrap_unchecked() };
        debug_assert!(
            part0.len() == body.len()
                && part1.len() == body.len()
                && part2.len() == body.len()
                && part3.len() <= body.len(),
            "the admitted four-part split has three full low blocks"
        );
        // SAFETY: the admitted high block is at most the m-limb body width.
        let (prefix, suffix) = unsafe { body.split_at_mut_unchecked(part3.len()) };
        // SAFETY: prefix and part3 contain exactly |part3| initialized limbs;
        // the scratch evaluation and immutable operand are disjoint.
        unsafe {
            copy_nonoverlapping(part3.as_ptr(), prefix.as_mut_ptr(), part3.len());
        }
        suffix.fill(0);
        *guard = 0;
        SharedEval::add_mul_word_with_kernel_in_place(dst, part2, 2, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(dst, part1, 4, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(dst, part0, 8, kernel);
    }

    /// Split an admitted operand into three full low blocks and its high block.
    pub const fn split_four(
        values: &[Limb],
        split_len: usize,
    ) -> (&[Limb], &[Limb], &[Limb], &[Limb]) {
        debug_assert!(
            values.len() > split_len.saturating_mul(3),
            "the driver admits four nonempty parts before splitting"
        );
        // SAFETY: the driver proved values.len()>3m before calling this splitter.
        // Successive immutable splits preserve lifetimes and leave four nonempty parts.
        unsafe {
            let (part0, after_part0) = values.split_at_unchecked(split_len);
            let (part1, after_part1) = after_part0.split_at_unchecked(split_len);
            let (part2, part3) = after_part1.split_at_unchecked(split_len);
            (part0, part1, part2, part3)
        }
    }
    #[expect(
        clippy::as_conversions,
        reason = "The cast extracts the exact low and high limbs of the wide accumulator"
    )]
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::cast_possible_truncation,
            reason = "DoubleLimb narrows to its low and high native limbs on 32-bit and 64-bit targets"
        )
    )]
    const fn evaluate_at_two_limb(
        part0: Limb,
        part1: Limb,
        part2: Limb,
        part3: Limb,
        carry: &mut Limb,
    ) -> Limb {
        // SAFETY: weights sum to fifteen. Inductively carry<=14, hence the exact
        // weighted sum is <=15(B-1)+14=15B-1<B^2 even for B=2^16. Positive partial
        // sums are no larger; each DoubleLimb shift is at most three bits.
        let sum = unsafe {
            (part0 as DoubleLimb)
                .unchecked_add((part1 as DoubleLimb) << 1)
                .unchecked_add((part2 as DoubleLimb) << 2)
                .unchecked_add((part3 as DoubleLimb) << 3)
                .unchecked_add(*carry as DoubleLimb)
        };
        *carry = (sum >> LIMB_BITS) as Limb;
        sum as Limb
    }
}
