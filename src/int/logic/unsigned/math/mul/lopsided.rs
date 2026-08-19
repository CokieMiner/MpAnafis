//! Blocked multiplication for operands whose limb lengths differ substantially.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use core::cmp::{max, min};

use crate::parallel::ParallelExecutor;

use super::{Addition, Limb, Multiplication, TierCeiling, Widths};

/// Block width, as a multiple of the shorter operand, once a block product is
/// itself a transform. See [`transform_block_len`].
const WIDE_BLOCK_RATIO: usize = 16;

/// Namespace for blocked multiplication of highly unbalanced operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lopsided;

impl Lopsided {
    /// Return reusable workspace for a benchmark-forced lopsided block width.
    ///
    /// This is kept separate from production geometry selection so every proposed
    /// shape can be measured with the exact same accumulation and tail handling.
    pub fn mul_forced_scratch_len(len_a: usize, len_b: usize, block_len: usize) -> usize {
        let smaller_len = min(len_a, len_b);
        let larger_len = max(len_a, len_b);
        // SAFETY: block_len > 0 by construction (determined from nonzero operand lengths).
        let tail_len = unsafe { larger_len.checked_rem(block_len).unwrap_unchecked() };
        let block_scratch = Multiplication::required_scratch(block_len, smaller_len);
        let tail_scratch = Multiplication::required_scratch(smaller_len, tail_len);
        block_len
            .wrapping_add(smaller_len)
            .wrapping_add(max(block_scratch, tail_scratch))
    }

    /// Multiply a highly unbalanced pair as reusable balanced block products.
    ///
    /// Writing the first product directly initializes the destination prefix.
    /// Every later block overlaps that prefix by exactly `smaller.len()` limbs;
    /// its remaining high limbs extend the initialized frontier and are copied,
    /// not added, so dirty destination storage is never read.
    pub fn mul<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        executor: &E,
    ) {
        let smaller_len = min(a.len(), b.len());
        let larger_len = max(a.len(), b.len());
        Self::mul_forced(
            dst,
            a,
            b,
            scratch,
            Self::block_len(larger_len, smaller_len),
            executor,
        );
    }

    /// Multiply a highly unbalanced pair with a benchmark-forced full-block width.
    ///
    /// Production calls this with [`Self::block_len`]. The tuning facade may
    /// supply another nonzero width to compare complete algorithms rather than
    /// extrapolating isolated block timings.
    pub fn mul_forced<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        block_len: usize,
        executor: &E,
    ) {
        let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let smaller_len = smaller.len();
        if smaller_len == 0 {
            dst.fill(0);
            return;
        }
        debug_assert!(
            Widths::new(larger.len(), smaller_len).prefers_blocked_product(),
            "lopsided multiplication requires the blocked-product ratio the dispatcher selects on"
        );
        debug_assert!(
            dst.len() >= larger.len().wrapping_add(smaller_len),
            "lopsided multiplication destination is undersized"
        );
        debug_assert!(
            scratch.len() >= Self::mul_forced_scratch_len(larger.len(), smaller_len, block_len),
            "lopsided multiplication scratch is undersized"
        );

        let block_product_capacity = block_len.wrapping_add(smaller_len);
        let (block_product, recursive_scratch) = scratch.split_at_mut(block_product_capacity);
        let mut blocks = larger.chunks(block_len);
        let full_block_plan =
            Multiplication::select_plan(block_len, smaller_len, TierCeiling::Full);
        // SAFETY: larger is nonempty and block_len <= larger.len(), so chunks() yields ≥ 1 block.
        let first_block = unsafe { blocks.next().unwrap_unchecked() };
        debug_assert_eq!(
            first_block.len(),
            block_len,
            "the first lopsided block must have full width"
        );
        let (initialized_prefix, _) = dst.split_at_mut(block_product_capacity);
        Multiplication::execute_plan_with_executor(
            full_block_plan,
            initialized_prefix,
            first_block,
            smaller,
            recursive_scratch,
            executor,
        );

        let mut block_offset = block_len;
        for block in blocks {
            let active_product_len = smaller_len.wrapping_add(block.len());
            let (active_product, _) = block_product.split_at_mut(active_product_len);
            if block.len() == block_len {
                Multiplication::execute_plan_with_executor(
                    full_block_plan,
                    active_product,
                    block,
                    smaller,
                    recursive_scratch,
                    executor,
                );
            } else {
                let tail_plan =
                    Multiplication::select_plan(block.len(), smaller_len, TierCeiling::Full);
                Multiplication::execute_plan_with_executor(
                    tail_plan,
                    active_product,
                    block,
                    smaller,
                    recursive_scratch,
                    executor,
                );
            }

            let (product_low, product_high) = active_product.split_at(smaller_len);
            let (_, destination_from_block) = dst.split_at_mut(block_offset);
            let (overlap, destination_after_overlap) =
                destination_from_block.split_at_mut(smaller_len);
            let (new_tail, _) = destination_after_overlap.split_at_mut(block.len());
            let carry = Addition::add_slice_in_place(overlap, product_low);
            // SAFETY: product_high and new_tail have the same `block.len()` width
            // and are disjoint scratch/destination regions. `carry` is zero or one.
            let final_carry = unsafe {
                Addition::copy_tail_with_carry(
                    new_tail.as_mut_ptr(),
                    product_high.as_ptr(),
                    product_high.len(),
                    carry,
                )
            };
            // If P is the already processed prefix of the larger operand and C is
            // this block, then (P + C*B^offset)*smaller is strictly below
            // B^(offset + block.len() + smaller_len). That is precisely the new
            // initialized frontier, so no carry can exist beyond `new_tail`.
            debug_assert_eq!(
                final_carry, 0,
                "lopsided block accumulation exceeded the full product width"
            );
            block_offset = block_offset.wrapping_add(block.len());
        }
        debug_assert_eq!(
            block_offset,
            larger.len(),
            "lopsided block traversal did not consume the larger operand"
        );
    }

    /// Select a full-block width aligned with an implemented production tier.
    ///
    /// Toom-8.5 evaluates a degree-eight block against a degree-seven operand at
    /// the same split width, so a `9:8` block covers one eighth more input without
    /// enlarging its recursive point products. When an even partition remains
    /// within one sixteenth of that preferred width, eliminating the short tail
    /// retains Toom-8/8.5 while saving a separate recursive product. Below that
    /// tier, equal blocks retain the tuned balanced specializations.
    pub fn block_len(larger_len: usize, smaller_len: usize) -> usize {
        if let Some(wide_block_len) = transform_block_len(larger_len, smaller_len) {
            return wide_block_len;
        }
        let toom8_half_len = smaller_len.saturating_add(smaller_len.div_ceil(8));
        if !Multiplication::select_plan(toom8_half_len, smaller_len, TierCeiling::Full)
            .reaches_widest_tier()
        {
            return smaller_len;
        }
        let block_count = larger_len.div_ceil(toom8_half_len);
        let even_block_len = larger_len.div_ceil(block_count);
        if even_block_len.saturating_mul(16) >= toom8_half_len.saturating_mul(15)
            && Multiplication::select_plan(even_block_len, smaller_len, TierCeiling::Full)
                .reaches_widest_tier()
        {
            even_block_len
        } else {
            toom8_half_len
        }
    }
}

/// Block width for the case where each block product is itself a transform.
///
/// The rule above sizes blocks so each product is *balanced* enough for the
/// widest conventional split, which is right while the block product lands in
/// the Toom ladder. Once the shorter operand is long enough that a block product
/// reaches a transform instead, that reasoning inverts: a transform does not
/// care that its operands are unbalanced, and blocking's total cost is
/// `(larger / block) * M(block, smaller)`, so the overlap of `smaller` limbs per
/// block is pure repeated work that only a wider block amortizes.
///
/// The width cannot simply grow without bound — that is the single whole-product
/// transform, whose ring is sized by the longer operand while only `smaller`
/// limbs of content exist. [`WIDE_BLOCK_RATIO`] is where those two pressures
/// balance, measured across 32:1 to 256:1 at 131073 through 1048577 limbs:
///
/// | block width | 1x | 1.125x | 2x | 4x | 8x | 16x | 32x |
/// |---|---|---|---|---|---|---|---|
/// | mean against the reference | 1.21 | 1.19 | 1.06 | 0.92 | 0.95 | 0.89 | 0.93 |
///
/// The old narrow rule was the *worst* choice in that whole region, measuring
/// 1.13x to 1.34x behind where a sixteen-fold block measures 0.82x to 0.96x.
fn transform_block_len(larger_len: usize, smaller_len: usize) -> Option<usize> {
    let target = smaller_len.checked_mul(WIDE_BLOCK_RATIO)?;
    // A block this wide pays when the block product or balanced equivalent reaches a transform,
    // and only counts as blocking if at least two blocks remain.
    let is_transform = Multiplication::select_plan(target, smaller_len, TierCeiling::Full)
        .is_transform()
        || Multiplication::select_plan(smaller_len, smaller_len, TierCeiling::Full).is_transform();
    if !is_transform || larger_len < target.saturating_mul(2) {
        return None;
    }
    let block_count = larger_len.div_ceil(target);
    Some(larger_len.div_ceil(block_count))
}

#[cfg(test)]
#[path = "tests/tiers/lopsided.rs"]
mod tests;
