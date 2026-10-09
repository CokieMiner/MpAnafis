//! Blocked multiplication for operands whose limb lengths differ substantially.
//!
//! # References
//!
//! - Brent, R. P., & Zimmermann, P. (2011). *Modern Computer Arithmetic*,
//!   Section 1.3.3: Unequal Sizes. Cambridge University Press.
//! - Bodrato, M. (2007). Towards Optimal Toom-Cook
//!   Multiplication for Univariate and Multivariate Polynomials in
//!   Characteristic 2 and 0. *Proceedings of WAIFI '07*, LNCS 4547, 116–133.
//!   <https://doi.org/10.1007/978-3-540-73074-3_10>.

#![expect(
    unsafe_code,
    reason = "The checked block layout bounds every worker span and accumulation frontier; mutable partitions are disjoint"
)]

use core::{
    cmp::{max, min},
    num::NonZeroUsize,
    ops::{Div, Rem},
};

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{
    Addition, LOPSIDED_TRANSFORM_BLOCK_RATIO, Limb, LimbOutput, MulPlan, Multiplication,
    TierCeiling, Widths,
};

/// Namespace for blocked multiplication of highly unbalanced operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lopsided;

/// Shared block geometry and one worker's product-plus-recursion layout.
struct BlockJob<'operand> {
    smaller: &'operand [Limb],
    block_len: NonZeroUsize,
    plan: MulPlan,
    product_capacity: usize,
    worker_scratch_len: NonZeroUsize,
}

impl Lopsided {
    /// Returns the workspace for the selected block width and worker budget.
    pub fn mul_scratch_len(
        len_a: usize,
        len_b: usize,
        block_len: NonZeroUsize,
        parallelism: usize,
    ) -> usize {
        let smaller_len = min(len_a, len_b);
        let larger_len = max(len_a, len_b);
        if smaller_len == 0 {
            return 0;
        }
        let block_count = larger_len.div_ceil(block_len.get());
        let workers = Self::worker_count(parallelism, block_count, usize::MAX);
        let child_parallelism = if workers == 1 { parallelism } else { 1 };
        let (_, worker_scratch_len) =
            Self::worker_region(larger_len, smaller_len, block_len, child_parallelism);
        worker_scratch_len
            .get()
            .checked_mul(workers)
            .expect("lopsided worker workspace overflows usize")
    }

    /// Multiply a highly unbalanced pair as reusable balanced block products.
    ///
    /// Writing the first product directly initializes the destination prefix.
    /// Later blocks add their `smaller.len()`-limb overlap and copy their high
    /// extension, so accumulation reads only initialized storage. Each batch
    /// multiplies one block per worker, then reconstructs in block order.
    ///
    /// The dispatcher orders nonempty operands as `larger.len() >= smaller.len()`
    /// and selects a positive block width. Execution requires
    /// `block_len <= larger.len()`, a complete destination, and at least one
    /// region from [`Self::mul_scratch_len`].
    #[expect(
        clippy::too_many_lines,
        reason = "The batch driver keeps workspace validation, first-prefix initialization, and ordered reconstruction together without single-use stage wrappers"
    )]
    pub fn mul<E: ParallelExecutor>(
        dst: &mut [impl LimbOutput],
        larger: &[Limb],
        smaller: &[Limb],
        scratch: &mut [Limb],
        block_len: NonZeroUsize,
        executor: &E,
    ) {
        let smaller_len = smaller.len();
        let block_limbs = block_len.get();
        debug_assert!(
            larger.len() >= smaller_len,
            "block operands must be ordered"
        );
        debug_assert!(
            Widths::new(larger.len(), smaller_len).prefers_blocked_product(),
            "lopsided multiplication requires the blocked-product ratio the dispatcher selects on"
        );
        debug_assert!(
            // SAFETY: two native-limb slice byte bounds prove their summed
            // widths fit usize on all supported pointer widths.
            dst.len() >= unsafe { larger.len().unchecked_add(smaller_len) },
            "lopsided multiplication destination is undersized"
        );
        debug_assert!(block_limbs <= larger.len(), "invalid lopsided block width");
        // SAFETY: block_len <= larger.len(). Both valid limb slices have at
        // most isize::MAX/size_of::<Limb>() elements, with Limb >= 2 bytes,
        // so this combined limb count is at most isize::MAX < usize::MAX.
        let product_capacity = unsafe { block_limbs.unchecked_add(smaller_len) };
        let block_count = larger.len().div_ceil(block_limbs);
        let parallelism = executor.parallelism().get();
        let wanted_workers = Self::worker_count(parallelism, block_count, usize::MAX);
        let child_parallelism = if wanted_workers == 1 { parallelism } else { 1 };
        let (plan, worker_width) =
            Self::worker_region(larger.len(), smaller_len, block_len, child_parallelism);
        let worker_scratch_len = worker_width.get();
        debug_assert!(
            scratch.len() >= worker_scratch_len,
            "lopsided multiplication needs one complete worker region"
        );
        let affordable_workers = scratch.len().div(worker_width);
        let workers = Self::worker_count(parallelism, block_count, affordable_workers);
        // SAFETY: workers <= affordable_workers and block_len <= worker_scratch_len,
        // so the complete batch width is bounded by the valid scratch slice length.
        let batch_len = unsafe { block_limbs.unchecked_mul(workers) };
        let job = BlockJob {
            smaller,
            block_len,
            plan,
            product_capacity,
            worker_scratch_len: worker_width,
        };

        // The first block initializes the destination prefix. Multiple outer
        // workers use sequential recursion within their disjoint regions.
        // SAFETY: block_len is positive and no wider than larger. The complete
        // destination covers block_len+smaller_len. The caller retains at least
        // one validated worker region, whose checked width includes that product
        // and its recursive workspace. Contiguous splits keep mutable spans disjoint.
        let (first_block, after_first, first_prefix, first_scratch, other_regions) = unsafe {
            let (first_block, after_first) = larger.split_at_unchecked(block_limbs);
            let (first_prefix, _) = dst.split_at_mut_unchecked(product_capacity);
            let (first_region, other_regions) = scratch.split_at_mut_unchecked(worker_scratch_len);
            let (_, first_scratch) = first_region.split_at_mut_unchecked(product_capacity);
            (
                first_block,
                after_first,
                first_prefix,
                first_scratch,
                other_regions,
            )
        };
        let mut consumed = block_limbs;
        // SAFETY: the nonempty larger operand contains the complete first
        // block, so ceil(larger.len()/block_len) >= 1.
        let mut remaining_blocks = unsafe { block_count.unchecked_sub(1) };
        if workers == 1 && wanted_workers == 1 {
            Multiplication::execute_plan_with_executor(
                plan,
                first_prefix,
                first_block,
                smaller,
                first_scratch,
                executor,
            );
        } else if workers == 1 {
            // A reduced caller workspace can limit the outer batch to one
            // region. Its layout still belongs to a sequential block, so it
            // must not inherit the caller's wider transform executor.
            Multiplication::execute_plan_with_executor(
                plan,
                first_prefix,
                first_block,
                smaller,
                first_scratch,
                &SequentialExecutor,
            );
        } else {
            // SAFETY: workers > 1, and batch_len contains the first full block.
            let companion_len = min(after_first.len(), unsafe {
                batch_len.unchecked_sub(block_limbs)
            });
            // SAFETY: min(after_first.len(), ...) bounds this companion prefix.
            let (companions, _) = unsafe { after_first.split_at_unchecked(companion_len) };
            // SAFETY: workers > 1 and workers <= block_count. Removing the
            // first block leaves at least one companion; min retains exactly
            // the number of blocks in the companion prefix, including its tail.
            let companion_blocks = unsafe {
                NonZeroUsize::new_unchecked(min(remaining_blocks, workers.unchecked_sub(1)))
            };
            executor.join(
                || {
                    Multiplication::execute_plan_with_executor(
                        plan,
                        first_prefix,
                        first_block,
                        smaller,
                        first_scratch,
                        &SequentialExecutor,
                    );
                },
                || {
                    Self::multiply_batch(
                        executor,
                        other_regions,
                        companions,
                        companion_blocks,
                        &job,
                    );
                },
            );
            Self::accumulate_batch(dst, consumed, other_regions, companions, &job);
            // SAFETY: companions are a prefix of after_first, so this sum is
            // at most larger.len().
            consumed = unsafe { consumed.unchecked_add(companion_len) };
            // SAFETY: companion_blocks is at most remaining_blocks.
            remaining_blocks = unsafe { remaining_blocks.unchecked_sub(companion_blocks.get()) };
        }

        // SAFETY: consumed is the first block plus a prefix of after_first,
        // hence never exceeds larger.len().
        let (_, remaining) = unsafe { larger.split_at_unchecked(consumed) };
        for span in remaining.chunks(batch_len) {
            // SAFETY: chunks yields a nonempty span of at most workers blocks.
            // remaining_blocks counts the unconsumed blocks, so this minimum
            // is its exact positive block count, including a possible short tail.
            let batch_blocks =
                unsafe { NonZeroUsize::new_unchecked(min(remaining_blocks, workers)) };
            Self::multiply_batch(executor, scratch, span, batch_blocks, &job);
            Self::accumulate_batch(dst, consumed, scratch, span, &job);
            // SAFETY: disjoint chunks partition the remaining input; the
            // consumed frontier never exceeds larger.len().
            consumed = unsafe { consumed.unchecked_add(span.len()) };
            // SAFETY: batch_blocks is at most remaining_blocks.
            remaining_blocks = unsafe { remaining_blocks.unchecked_sub(batch_blocks.get()) };
        }
        debug_assert_eq!(remaining_blocks, 0, "lopsided block count was not consumed");
        debug_assert_eq!(
            consumed,
            larger.len(),
            "lopsided block traversal did not consume the larger operand"
        );
    }

    /// Multiplies `block_count` blocks into disjoint worker regions.
    ///
    /// Input and scratch splits use the same block index at each fork.
    fn multiply_batch<E: ParallelExecutor>(
        executor: &E,
        worker_scratch: &mut [Limb],
        span: &[Limb],
        block_count: NonZeroUsize,
        job: &BlockJob<'_>,
    ) {
        let block_len = job.block_len.get();
        debug_assert_eq!(
            block_count.get(),
            span.len().div_ceil(block_len),
            "lopsided batch count differs from its nonempty span"
        );
        if block_count.get() == 1 {
            // SAFETY: the positive block count proves this leaf is nonempty.
            // It owns a complete worker region with product_capacity limbs.
            let (product, scratch) =
                unsafe { worker_scratch.split_at_mut_unchecked(job.product_capacity) };
            // SAFETY: span is at most one block, and product_capacity
            // reserves block_len + smaller.len() initialized limbs.
            let active_product_len = unsafe { job.smaller.len().unchecked_add(span.len()) };
            // SAFETY: active_product_len <= job.product_capacity == product.len().
            let active_product = unsafe { product.get_unchecked_mut(..active_product_len) };
            let plan = if span.len() == block_len {
                job.plan
            } else {
                Multiplication::select_plan(span.len(), job.smaller.len(), TierCeiling::Full)
            };
            // Each outer worker executes its recursive product sequentially.
            Multiplication::execute_plan_with_executor(
                plan,
                active_product,
                span,
                job.smaller,
                scratch,
                &SequentialExecutor,
            );
            return;
        }
        // Split n >= 2 into floor(n/2) right blocks and the remaining left
        // blocks. This computes both child counts with one shift and subtraction.
        let right_count = block_count.get() >> 1;
        // SAFETY: n >= 2 proves 1 <= right_count < n and
        // 1 <= n-right_count < n. The left count is ceil(n/2), so those full
        // blocks lie strictly inside span. The batch owns one scratch region
        // per block, and contiguous partitions retain disjoint mutable borrows.
        let (left_span, right_span, left_scratch, right_scratch, left_blocks, right_blocks) = unsafe {
            let left_count = block_count.get().unchecked_sub(right_count);
            let span_split = left_count.unchecked_mul(block_len);
            let scratch_split = left_count.unchecked_mul(job.worker_scratch_len.get());
            let (left_span, right_span) = span.split_at_unchecked(span_split);
            let (left_scratch, right_scratch) =
                worker_scratch.split_at_mut_unchecked(scratch_split);
            (
                left_span,
                right_span,
                left_scratch,
                right_scratch,
                NonZeroUsize::new_unchecked(left_count),
                NonZeroUsize::new_unchecked(right_count),
            )
        };
        executor.join(
            || Self::multiply_batch(executor, left_scratch, left_span, left_blocks, job),
            || Self::multiply_batch(executor, right_scratch, right_span, right_blocks, job),
        );
    }

    /// Accumulates a completed batch from its initial radix position.
    fn accumulate_batch(
        dst: &mut [impl LimbOutput],
        block_offset: usize,
        worker_scratch: &[Limb],
        span: &[Limb],
        job: &BlockJob<'_>,
    ) {
        let mut offset = block_offset;
        for (block, region) in span
            .chunks(job.block_len.get())
            .zip(worker_scratch.chunks(job.worker_scratch_len.get()))
        {
            // SAFETY: the batch owns a full worker region per block. The zip
            // consumes only those complete regions, each retaining its entire
            // product_capacity prefix before the recursive workspace.
            let (product, _) = unsafe { region.split_at_unchecked(job.product_capacity) };
            let smaller_len = job.smaller.len();
            // SAFETY: each product region reserves block_len + smaller_len;
            // offset+block.len() is a frontier inside the larger operand.
            // The complete destination additionally reserves smaller_len limbs.
            let overlap_end = unsafe { offset.unchecked_add(smaller_len) };
            // SAFETY: the worker region contains smaller_len+job.block_len
            // initialized product limbs, and block.len() <= job.block_len.
            let (product_low, product_high) = unsafe {
                let (low, after_low) = product.split_at_unchecked(smaller_len);
                let (high, _) = after_low.split_at_unchecked(block.len());
                (low, high)
            };
            // SAFETY: offset+block.len() is inside the larger operand, so
            // overlap_end+block.len() <= larger.len()+smaller_len <= dst.len().
            // Consecutive splits give disjoint mutable overlap and tail spans.
            let (overlap, new_tail) = unsafe {
                let (initialized, after_overlap) = dst.split_at_mut_unchecked(overlap_end);
                let (_, overlap) = initialized.split_at_mut_unchecked(offset);
                let (tail, _) = after_overlap.split_at_mut_unchecked(block.len());
                (overlap, tail)
            };
            // SAFETY: the first product established the initial frontier;
            // each preceding block extends it by block.len(). The complete
            // smaller_len-limb overlap is below that initialized frontier.
            let carry = Addition::add_slice_in_place(
                unsafe { LimbOutput::assume_init_mut(overlap) },
                product_low,
            );
            // SAFETY: the product's block.len()-limb high part is disjoint from
            // the equally wide destination extension; carry is binary. Copying
            // initializes the new frontier before any later overlap reads it.
            let final_carry = unsafe {
                Addition::copy_tail_with_carry(
                    new_tail.as_mut_ptr().cast(),
                    product_high.as_ptr(),
                    block.len(),
                    carry,
                )
            };
            // The partial sum multiplies a prefix of the larger operand and
            // fits prefix.len()+smaller_len limbs, so no carry escapes.
            debug_assert_eq!(
                final_carry, 0,
                "block accumulation exceeded the product width"
            );
            // SAFETY: these chunks partition a span inside the larger input.
            offset = unsafe { offset.unchecked_add(block.len()) };
        }
    }

    /// Select a full-block width aligned with an implemented production tier.
    ///
    /// Toom-8.5 evaluates a degree-eight block against a degree-seven operand at
    /// the same split width, so a `9:8` block covers one eighth more input without
    /// enlarging its recursive point products. When an even partition remains
    /// within one sixteenth of that preferred width, eliminating the short tail
    /// retains Toom-8/8.5 while saving a separate recursive product. Below that
    /// tier, equal blocks retain the tuned balanced specializations.
    /// Sizing and execution establish nonempty, ordered widths before selection.
    pub fn block_len(larger_width: NonZeroUsize, smaller_width: NonZeroUsize) -> NonZeroUsize {
        let larger_len = larger_width.get();
        let smaller_len = smaller_width.get();
        debug_assert!(larger_len >= smaller_len, "block widths must be ordered");
        // Transform blocks amortize overlap accumulation over a larger span.
        // Rejection retains the conventional blocked-product fallback.
        if let Some(target) = smaller_len.checked_mul(LOPSIDED_TRANSFORM_BLOCK_RATIO) {
            // 2*target <= larger_len iff target <= floor(larger_len/2).
            // This comparison also rejects an overflowing doubled target for
            // arbitrary virtual widths without constructing that product.
            if target <= larger_len >> 1
                && (Multiplication::select_plan(target, smaller_len, TierCeiling::Full)
                    .is_transform()
                    || Multiplication::select_plan(smaller_len, smaller_len, TierCeiling::Full)
                        .is_transform())
            {
                // SAFETY: the tuning schema requires a positive transform
                // block ratio; the checked product of two positive widths
                // remains positive. NonZero division preserves positive counts.
                let target_width = unsafe { NonZeroUsize::new_unchecked(target) };
                let transform_blocks = larger_width.div_ceil(target_width);
                return larger_width.div_ceil(transform_blocks);
            }
        }
        // An unrepresentable 9:8 extension cannot name a legal block width;
        // retain the original positive width for virtual sizing queries.
        let Some(preferred_width) = smaller_width.checked_add(smaller_len.div_ceil(8)) else {
            return smaller_width;
        };
        let toom8_half_len = preferred_width.get();
        if !Multiplication::select_plan(toom8_half_len, smaller_len, TierCeiling::Full)
            .reaches_widest_tier()
        {
            return smaller_width;
        }
        let block_count = larger_width.div_ceil(preferred_width);
        let even_block_width = larger_width.div_ceil(block_count);
        // ceil(15*t/16) = t-floor(t/16), avoiding overflowing cross products.
        // SAFETY: floor(t/16) <= t for every usize width.
        let minimum_even_width =
            unsafe { toom8_half_len.unchecked_sub(toom8_half_len.div_euclid(16)) };
        if even_block_width.get() >= minimum_even_width
            && Multiplication::select_plan(even_block_width.get(), smaller_len, TierCeiling::Full)
                .reaches_widest_tier()
        {
            even_block_width
        } else {
            preferred_width
        }
    }
    /// Reserves a product and the maximum of full-block and tail scratch.
    ///
    /// Full blocks use `child_parallelism`; tails execute sequentially.
    fn worker_region(
        larger_len: usize,
        smaller_len: usize,
        block_width: NonZeroUsize,
        child_parallelism: usize,
    ) -> (MulPlan, NonZeroUsize) {
        let block_len = block_width.get();
        let tail_len = larger_len.rem(block_width);
        let plan = Multiplication::select_plan(block_len, smaller_len, TierCeiling::Full);
        let block_scratch = Multiplication::scratch_len_for_parallelism(
            plan,
            block_len,
            smaller_len,
            child_parallelism,
        );
        let tail_scratch = if tail_len == 0 {
            0
        } else {
            let tail_plan = Multiplication::select_plan(tail_len, smaller_len, TierCeiling::Full);
            Multiplication::scratch_len_for_parallelism(tail_plan, tail_len, smaller_len, 1)
        };
        let region = block_len
            .checked_add(smaller_len)
            .and_then(|product| product.checked_add(max(block_scratch, tail_scratch)))
            .expect("lopsided worker region overflows usize");
        // SAFETY: the checked sum contains block_len > 0 and only nonnegative
        // additions, so every successfully validated region is positive.
        (plan, unsafe { NonZeroUsize::new_unchecked(region) })
    }

    /// Selects one, two, or four workers within the budget and available regions.
    const fn worker_count(parallelism: usize, block_count: usize, affordable: usize) -> usize {
        #[cfg(target_pointer_width = "16")]
        {
            let _ = (parallelism, block_count, affordable);
            1
        }
        #[cfg(not(target_pointer_width = "16"))]
        {
            if parallelism >= 4 && block_count >= 4 && affordable >= 4 {
                4
            } else if parallelism >= 2 && block_count >= 2 && affordable >= 2 {
                2
            } else {
                1
            }
        }
    }
}
