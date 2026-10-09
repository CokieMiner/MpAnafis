//! Truncated low-product multiplication tier.
//!
//! References: T. Mulders, "On Short Multiplications and Divisions",
//! AAECC 11, 69-88, 2000. <https://doi.org/10.1007/s002000000037>.
//! G. Hanrot and P. Zimmermann, "A long note on Mulders' short product",
//! Journal of Symbolic Computation 37, 391-401, 2004.
//! <https://doi.org/10.1016/j.jsc.2003.03.001>.

#![expect(
    unsafe_code,
    reason = "Padded input spans and the checked Mulders layout bound recursive partitions; scaled split ratios cannot overflow"
)]

use core::{
    cmp::max,
    mem::MaybeUninit,
    ptr::{copy_nonoverlapping, eq},
    slice::from_raw_parts_mut,
};

use crate::parallel::{DefaultExecutor, ParallelExecutor, SequentialExecutor};

use super::{
    Addition, ArchKernels, LOW_PRODUCT_FULL_THRESHOLD, LOW_PRODUCT_RECURSIVE_THRESHOLD, Limb,
    LimbOutput, MulScratch, Multiplication, Schoolbook, ScratchBuffer, TierCeiling,
};

/// Namespace for truncated low-product multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LowProduct;

impl LowProduct {
    /// Truncated low product computation modulo $B^{\text{len}}$.
    ///
    /// Computes `dst = (a * b) mod B^len` into a preallocated slice.
    /// The destination has at least `len` writable limbs; both inputs contain
    /// at least `len` initialized limbs. Every active destination limb is
    /// initialized before accumulation; any remaining suffix is unchanged.
    pub fn mul(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        len: usize,
        scratch: &mut MulScratch,
    ) {
        if len == 0 {
            return;
        }
        debug_assert!(
            dst.len() >= len && a.len() >= len && b.len() >= len,
            "slice lengths too short for truncated product"
        );
        if len < LOW_PRODUCT_RECURSIVE_THRESHOLD {
            // SAFETY: the caller's padded-input contract supplies all three
            // len-limb spans; Rust's borrows make the destination disjoint.
            // LimbOutput preserves limb layout. The first row initializes
            // every destination limb before subsequent rows read it.
            unsafe {
                Schoolbook::mullo_basecase_unchecked(
                    dst.as_mut_ptr().cast(),
                    a.as_ptr(),
                    b.as_ptr(),
                    len,
                );
            }
            return;
        }
        let plan = Multiplication::select_plan(len, len, TierCeiling::Full);
        if (LOW_PRODUCT_FULL_THRESHOLD != 0 && len >= LOW_PRODUCT_FULL_THRESHOLD)
            || plan.is_transform()
        {
            // SAFETY: each factor occupies at most isize::MAX bytes; double len fits usize.
            let product_len = unsafe { len.unchecked_mul(2) };
            let square_plan = eq(a.as_ptr(), b.as_ptr())
                .then(|| Multiplication::select_square_plan(len, TierCeiling::Full));
            let full_inner = square_plan.map_or_else(
                || Multiplication::scratch_len(plan, len, len),
                |selected| Multiplication::square_scratch_len(selected, len),
            );
            let full_scratch = product_len
                .checked_add(full_inner)
                .expect("low product full scratch width overflows");
            scratch.prepare(full_scratch);
            // SAFETY: reservation guarantees product_len disjoint writable limbs
            // followed by full_inner scratch limbs.
            let (full_product, full_work) =
                unsafe { scratch.buf.split_at_mut_unchecked(product_len) };
            // SAFETY: caller pads inputs to at least len limbs.
            let (a_slice, b_slice) = unsafe { (a.get_unchecked(..len), b.get_unchecked(..len)) };
            if let Some(selected_square) = square_plan {
                Multiplication::execute_square_plan(
                    selected_square,
                    full_product,
                    a_slice,
                    full_work,
                );
            } else {
                Multiplication::execute_plan(plan, full_product, a_slice, b_slice, full_work);
            }
            // SAFETY: full_product holds at least 2*len >= len initialized limbs.
            // dst holds at least len writable slots with matching limb layout.
            unsafe {
                copy_nonoverlapping(full_product.as_ptr(), dst.as_mut_ptr().cast(), len);
            }
        } else {
            // A conventional product schedule uses one worker unless the
            // independent square tower selects a transform at this width.
            if Multiplication::select_square_plan(len, TierCeiling::Full).is_transform() {
                DefaultExecutor::with_resolved(|executor| {
                    Self::mulders_with_scratch(dst, a, b, len, scratch, executor);
                });
            } else {
                Self::mulders_with_scratch(dst, a, b, len, scratch, &SequentialExecutor);
            }
        }
    }

    /// Prepares one owned arena at the executor dispatch boundary, then runs the split.
    ///
    /// `len` reaches the recursive crossover; all three active spans cover len limbs.
    fn mulders_with_scratch(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        len: usize,
        scratch: &mut MulScratch,
        executor: &impl ParallelExecutor,
    ) {
        let small_len = Multiplication::mulders_small_len::<2>(len);
        let scratch_len = Self::mulders_scratch_len(len, small_len, executor.parallelism().get());
        scratch.prepare(scratch_len);
        // SAFETY: the root contract supplies at least len initialized left limbs.
        let left = unsafe { a.get_unchecked(..len) };
        // SAFETY: the root contract supplies at least len initialized right limbs.
        let right = unsafe { b.get_unchecked(..len) };
        // SAFETY: the active spans cover len limbs; sizing and execution share
        // one executor budget. prepare initialized the complete disjoint arena.
        unsafe {
            Self::mulders_at_split(dst, left, right, len, small_len, &mut scratch.buf, executor);
        }
    }

    /// Scratch limbs for one Mulders split and its recursive cross products.
    ///
    /// The full low-block product and either cross product never coexist: after
    /// copying the needed low `len` limbs to `dst`, the entire full-product region
    /// may be reused. The recurrence is therefore the maximum, not the sum, of
    /// `2*large + full_itch` and `small + low_itch(small)`.
    ///
    /// `0 < small_len <= len/2`. Root tuning uses this layout below the
    /// production crossover; recursive children retain that crossover.
    #[must_use]
    pub fn mulders_scratch_len(len: usize, small_len: usize, parallelism: usize) -> usize {
        debug_assert!(
            small_len > 0 && small_len <= len.div_euclid(2),
            "the low-product split has two nonempty blocks"
        );
        let mut current_width = len;
        let mut small = small_len;
        let mut prefix_len = 0_usize;
        let mut required = 0;
        loop {
            // SAFETY: each selected split satisfies 0<small<=current_width/2,
            // including virtual widths used only for checked workspace sizing.
            let large_len = unsafe { current_width.unchecked_sub(small) };
            let full_inner = max(
                Multiplication::scratch_len_for_parallelism(
                    Multiplication::select_plan(large_len, large_len, TierCeiling::Full),
                    large_len,
                    large_len,
                    parallelism,
                ),
                Multiplication::square_scratch_len_for_parallelism(
                    Multiplication::select_square_plan(large_len, TierCeiling::Full),
                    large_len,
                    parallelism,
                ),
            );
            let full_phase = large_len
                .checked_mul(2)
                .and_then(|width| width.checked_add(full_inner))
                .and_then(|width| width.checked_add(prefix_len))
                .expect("Mulders full-product scratch width overflows");
            required = max(required, full_phase);
            if small < LOW_PRODUCT_RECURSIVE_THRESHOLD {
                return required;
            }
            // SAFETY: prefix_len+current_width<=len holds initially and is
            // preserved by 2*small<=current_width. Thus prefix_len+small<=len.
            prefix_len = unsafe { prefix_len.unchecked_add(small) };
            current_width = small;
            small = Multiplication::mulders_small_len::<2>(current_width);
        }
    }

    /// Writes `multiplier * factor mod B^(n+1)` using the multiplier's width.
    ///
    /// With `k=multiplier.len()` and `s=n+1-k`, split `factor=D0+D1*B^s`.
    /// The full product `multiplier*D0` fills n+1 limbs; only the low k limbs
    /// of `multiplier*D1` survive its shift. Every output limb is initialized.
    ///
    /// # Safety
    /// The factor has n>0 initialized limbs, the multiplier has at most n,
    /// and output has exactly n+1 writable limbs. Both operands, output and
    /// the three workspaces are pairwise disjoint. An empty multiplier is valid.
    pub unsafe fn mul_with_guard<'output>(
        factor: &[Limb],
        multiplier: &[Limb],
        output: &'output mut [MaybeUninit<Limb>],
        padded: &mut ScratchBuffer,
        cross_product: &mut ScratchBuffer,
        mul_scratch: &mut MulScratch,
    ) -> &'output mut [Limb] {
        debug_assert!(
            !factor.is_empty() && multiplier.len() <= factor.len(),
            "the multiplier fits the nonempty factor width"
        );
        debug_assert_eq!(
            output.len().checked_sub(1),
            Some(factor.len()),
            "the low product retains one guard limb"
        );
        if multiplier.is_empty() {
            output.fill(MaybeUninit::new(0));
            // SAFETY: the fill initializes every output element as a Limb.
            // MaybeUninit preserves its layout, alignment and exclusive borrow.
            return unsafe { from_raw_parts_mut(output.as_mut_ptr().cast(), output.len()) };
        }
        if (LOW_PRODUCT_FULL_THRESHOLD != 0 && output.len() >= LOW_PRODUCT_FULL_THRESHOLD)
            || Multiplication::select_plan(factor.len(), multiplier.len(), TierCeiling::Full)
                .is_transform()
        {
            // SAFETY: each factor occupies at most isize::MAX bytes;
            // their summed widths fit usize on all supported targets.
            let full_len = unsafe { factor.len().unchecked_add(multiplier.len()) };
            cross_product.reset_with_capacity(full_len);
            // SAFETY: reservation supplies full_len disjoint spare limbs.
            // Full multiplication initializes all full_len limbs.
            // output.len() = factor.len() + 1 <= full_len because multiplier is nonempty.
            let full = unsafe {
                Multiplication::mul_nonempty_distinct_into_uninit(
                    multiplier,
                    factor,
                    cross_product
                        .spare_capacity_mut()
                        .get_unchecked_mut(..full_len),
                    mul_scratch,
                )
            };
            // SAFETY: full contains full_len >= output.len() initialized limbs.
            // output has output.len() writable MaybeUninit<Limb> elements.
            // Copying initializes all output limbs without aliasing.
            unsafe {
                let dest_ptr = output.as_mut_ptr().cast::<Limb>();
                copy_nonoverlapping(full.as_ptr(), dest_ptr, output.len());
                return from_raw_parts_mut(dest_ptr, output.len());
            }
        }
        // SAFETY: 1<=k<=n and output.len()=n+1 give 1<=s=n+1-k<=n.
        // Both factor partitions remain within its initialized span.
        let (split, lower, higher) = unsafe {
            let split = output.len().unchecked_sub(multiplier.len());
            let (lower, higher) = factor.split_at_unchecked(split);
            (split, lower, higher)
        };
        // SAFETY: the disjoint nonempty k- and s-limb operands have a complete
        // product of k+s=n+1 limbs, exactly the reserved output width.
        let initialized = unsafe {
            Multiplication::mul_nonempty_distinct_into_uninit(
                multiplier,
                lower,
                output,
                mul_scratch,
            )
        };
        if higher.is_empty() {
            return initialized;
        }
        // D1 has k-1 limbs; its zero guard supplies the k-limb low-product span.
        if padded.capacity() < multiplier.len() {
            padded.reset_with_capacity(multiplier.len());
        }
        // SAFETY: reservation supplies k disjoint writable limbs. Copying the
        // k-1 initialized source limbs and writing zero initializes the full span.
        unsafe {
            let pointer = padded.as_mut_ptr();
            pointer.copy_from_nonoverlapping(higher.as_ptr(), higher.len());
            pointer.add(higher.len()).write(0);
            padded.set_len(multiplier.len());
        }
        cross_product.reset_with_capacity(multiplier.len());
        // SAFETY: reservation supplies k writable limbs disjoint from both
        // initialized k-limb operands. The low-product writer initializes
        // all k limbs before set_len exposes them to the following addition.
        unsafe {
            Self::mul(
                cross_product
                    .spare_capacity_mut()
                    .get_unchecked_mut(..multiplier.len()),
                multiplier,
                padded,
                multiplier.len(),
                mul_scratch,
            );
            cross_product.set_len(multiplier.len());
        }
        // SAFETY: s+k=output.len() bounds this initialized k-limb suffix.
        // Its escaping carry is discarded modulo B^(n+1).
        let upper = unsafe { initialized.get_unchecked_mut(split..) };
        let _ = Addition::add_slice_in_place(upper, cross_product);
        initialized
    }

    /// Executes one Mulders split; cross products retain only the required prefix.
    ///
    /// For `a=a0+a1*B^m`, `b=b0+b1*B^m`, and `m+s=len`, reduction gives
    /// `a*b=a0*b0+B^m*(a1*b0+a0*b1) (mod B^len)`.
    /// The `a1*b1*B^(2m)` term vanishes because `2m>=len`. Only the low
    /// s limbs of each cross product survive the shift by m.
    ///
    /// # Safety
    ///
    /// `dst` contains at least `len` writable limbs; `a` and `b` contain at least
    /// `len` initialized limbs. `dst` is disjoint from both inputs,
    /// `0 < small_len <= len / 2`, and `scratch`
    /// contains the checked full-product and recursive cross-product workspace
    /// described by [`Self::mulders_scratch_len`] for this split and
    /// executor budget. The two inputs may alias one another.
    pub unsafe fn mulders_at_split<Output: LimbOutput>(
        dst: &mut [Output],
        a: &[Limb],
        b: &[Limb],
        len: usize,
        small_len: usize,
        scratch: &mut [Limb],
        executor: &impl ParallelExecutor,
    ) {
        debug_assert!(
            small_len > 0 && small_len <= len.div_euclid(2),
            "Mulders split must keep a nonempty smaller block no wider than the full-product block"
        );
        // SAFETY: 0 < small_len <= len/2 proves this subtraction exists and gives
        // large_len >= small_len with large_len+small_len == len.
        let large_len = unsafe { len.unchecked_sub(small_len) };
        // SAFETY: both inputs contain len initialized limbs and large_len <= len.
        // Their exact active spans split into large_len and small_len limbs.
        let (a0, a1, b0, b1) = unsafe {
            let (a0, a1) = a.get_unchecked(..len).split_at_unchecked(large_len);
            let (b0, b1) = b.get_unchecked(..len).split_at_unchecked(large_len);
            (a0, a1, b0, b1)
        };

        // SAFETY: the checked scratch layout reserves this doubled subspan.
        let full_product_len = unsafe { large_len.unchecked_mul(2) };
        // SAFETY: sizing reserves 2*large_len plus the maximum of product and
        // square workspaces. Either child can use the remaining initialized suffix;
        // recomputing its required width would repeat the sizing traversal.
        let (full_product, full_inner) =
            unsafe { scratch.split_at_mut_unchecked(full_product_len) };
        if eq(a0.as_ptr(), b0.as_ptr()) {
            Multiplication::execute_square_plan_with_executor(
                Multiplication::select_square_plan(large_len, TierCeiling::Full),
                full_product,
                a0,
                full_inner,
                executor,
            );
        } else {
            Multiplication::execute_plan_with_executor(
                Multiplication::select_plan(large_len, large_len, TierCeiling::Full),
                full_product,
                a0,
                b0,
                full_inner,
                executor,
            );
        }
        // `full_product` has `2*large_len >= large_len+small_len == len` limbs.
        // SAFETY: the root invariant also gives `dst.len() >= len`.
        let writable = unsafe { dst.get_unchecked_mut(..len) };
        // SAFETY: the inequality above proves the product prefix is in bounds.
        let full_low = unsafe { full_product.get_unchecked(..len) };
        // SAFETY: the product prefix contains len initialized limbs, the output
        // prefix has len writable elements with identical limb layout, and their
        // allocations are disjoint. The copy initializes every output element.
        unsafe {
            copy_nonoverlapping(full_low.as_ptr(), writable.as_mut_ptr().cast(), len);
        }
        // SAFETY: the preceding copy initialized the entire active output prefix.
        let dst_low = unsafe { Output::assume_init_mut(writable) };

        if small_len < LOW_PRODUCT_RECURSIVE_THRESHOLD {
            // Modulo B^small_len, the two cross products may be interleaved by
            // row. One backend selection and traversal accumulate both, sharing
            // the destination offset, suffix width, and loop index calculations.
            let kernel = ArchKernels::selected_add_mul_limbs_unchecked();
            let mut index = 0_usize;
            while index < small_len {
                // SAFETY: a1 and b1 have small_len initialized limbs; a0 and b0
                // have at least that many. index < small_len bounds both scalars
                // and the nonempty output suffix initialized by the full-product
                // copy. large_len+index < len <= dst.len(); all spans are disjoint
                // from dst. Each escaping carry is discarded modulo B^small_len.
                unsafe {
                    let inner_len = small_len.unchecked_sub(index);
                    let output = dst_low.as_mut_ptr().add(large_len).add(index);
                    let _ = kernel(output, b0.as_ptr(), inner_len, *a1.get_unchecked(index));
                    let _ = kernel(output, a0.as_ptr(), inner_len, *b1.get_unchecked(index));
                    index = index.unchecked_add(1);
                }
            }
            return;
        }

        // SAFETY: sizing reserves small_len+low_itch(small_len) for this phase.
        // The initialized suffix therefore covers either sequential cross product
        // without recomputing the checked scratch recurrence inside the kernel.
        let (cross_product, cross_inner) = unsafe { scratch.split_at_mut_unchecked(small_len) };
        let child_split = Multiplication::mulders_small_len::<2>(small_len);
        // SAFETY: dst contains len limbs; large_len+small_len == len.
        let dst_high = unsafe { dst_low.get_unchecked_mut(large_len..) };
        {
            // SAFETY: `b0.len() == large_len >= small_len` by the split invariant.
            let b0_low = unsafe { b0.get_unchecked(..small_len) };
            // SAFETY: both inputs and `cross_product` have exactly `small_len`
            // limbs, and `cross_inner` was split to the recursive scratch length.
            unsafe {
                Self::mulders_at_split(
                    cross_product,
                    a1,
                    b0_low,
                    small_len,
                    child_split,
                    cross_inner,
                    executor,
                );
            }
            let _ = Addition::add_slice_in_place(dst_high, cross_product);
        }
        {
            // SAFETY: `a0.len() == large_len >= small_len` by the split invariant.
            let a0_low = unsafe { a0.get_unchecked(..small_len) };
            // SAFETY: this is the symmetric exact-width recursive cross product.
            unsafe {
                Self::mulders_at_split(
                    cross_product,
                    a0_low,
                    b1,
                    small_len,
                    child_split,
                    cross_inner,
                    executor,
                );
            }
            let _ = Addition::add_slice_in_place(dst_high, cross_product);
        }
    }
}
