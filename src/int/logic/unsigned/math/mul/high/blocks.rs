//! Recursive high products with bounded error from omitted low blocks.

#![expect(
    unsafe_code,
    reason = "Materialized operand widths and bounded recursive splits establish disjoint initialized blocks, cross products and certification guards"
)]

use core::{mem::MaybeUninit, num::NonZeroUsize, slice::from_raw_parts_mut};

use super::{Addition, ArchKernels, HighProduct, Limb, MulScratch, Multiplication};

impl HighProduct {
    /// Work bound for a materialized operand; leaves need no arena.
    pub const fn scratch_len(operand: &[Limb]) -> usize {
        let smaller = operand.len();
        if smaller < Self::RECURSIVE_THRESHOLD {
            0
        } else {
            // SAFETY: the slice spans at most isize::MAX bytes and each limb
            // occupies at least two bytes. Twice its length fits usize.
            unsafe { smaller.unchecked_mul(2) }
        }
    }

    /// Forms a lower high approximation, retaining two certification digits.
    ///
    /// For radix R, put c=skip-2 and split A=A0+R^s*A1, B=B0+R^s*B1 with 2s<=c.
    /// Let Ah=floor(A1/R^(c-2s)), and define Bh symmetrically. The retained sum
    /// is floor(A1*B1/R^(c-2s))+floor(Ah*B0/R^s)+floor(A0*Bh/R^s).
    /// The first discarded fraction and A0*B0/R^c are each below one; each
    /// truncated cross contribution is below two. Thus the exact c-scaled
    /// product exceeds this sum by at most five. Recursive cross approximations
    /// each lose at most one, so the bound is seven at every level. Dividing
    /// by R^2 again loses at most one since seven<R^2 on all supported targets.
    ///
    /// # Safety
    /// Nonempty initialized inputs, output, work and multiplication scratch
    /// are pairwise disjoint; `min(a.len(),b.len())<=skip<a.len()+b.len()`.
    /// Output reserves the complete product width. Work covers the pending
    /// descendant spans under the root's checked bound. Returned start..end is initialized
    /// and within one below the exact high product; start-2..start also holds
    /// initialized certification digits when skip>=3.
    pub unsafe fn high_product_blocks(
        a: &[Limb],
        b: &[Limb],
        skip: usize,
        output: &mut [MaybeUninit<Limb>],
        work: &mut [MaybeUninit<Limb>],
        mul_scratch: &mut MulScratch,
    ) -> (usize, usize) {
        // SAFETY: two materialized limb counts fit usize on every target.
        let total = unsafe { a.len().unchecked_add(b.len()) };
        let smaller = a.len().min(b.len());
        if smaller < Self::RECURSIVE_THRESHOLD {
            if skip < 3 {
                // SAFETY: the caller reserves the complete disjoint output span.
                unsafe {
                    let _ = Multiplication::mul_nonempty_distinct_into_uninit(
                        a,
                        b,
                        output,
                        mul_scratch,
                    );
                }
                return (skip, total);
            }
            // SAFETY: the operands are nonempty and output covers total-cut.
            // Omitting the low diagonals gives an error below B^2 at cut,
            // since their maximum column width is strictly below B.
            let width = unsafe {
                let cut = skip.unchecked_sub(2);
                let width = total.unchecked_sub(cut);
                Self::high_product_diagonals(a, b, cut, output.get_unchecked_mut(..width));
                width
            };
            return (2, width);
        }
        // SAFETY: skip>=smaller>=6 makes the two low guards unconditional
        // at recursive widths; skip<total bounds the retained product width.
        let cut = unsafe { skip.unchecked_sub(2) };
        let small = Multiplication::mulders_small_len::<3>(smaller);
        // SAFETY: smaller>=6 and skip>=smaller give 0<small<=smaller/3
        // and 2*small<=skip-2. All input partitions are nonempty and bounded.
        let (a0, a1, b0, b1, offset, top_start, product_len) = unsafe {
            let (a0, a1) = a.split_at_unchecked(small);
            let (b0, b1) = b.split_at_unchecked(small);
            (
                a0,
                a1,
                b0,
                b1,
                cut.unchecked_sub(small.unchecked_mul(2)),
                cut.unchecked_sub(small),
                total.unchecked_sub(small.unchecked_mul(2)),
            )
        };
        // SAFETY: small<=smaller/3 and smaller>=6 leave at least four
        // initialized digits in each high block. These minimum widths also
        // describe the carry tails after their corresponding cross additions.
        let (a_width, b_width) = unsafe {
            let (a_lower, _) = a1.split_last_chunk::<4>().unwrap_unchecked();
            let (b_lower, _) = b1.split_last_chunk::<4>().unwrap_unchecked();
            (
                a_lower.len().unchecked_add(4),
                b_lower.len().unchecked_add(4),
            )
        };
        // SAFETY: A1 and B1 are nonempty, initialized and disjoint; the
        // caller's output covers their product_len=total-2*small limbs.
        let full = unsafe {
            Multiplication::mul_nonempty_distinct_into_uninit(
                a1,
                b1,
                output.get_unchecked_mut(..product_len),
                mul_scratch,
            )
        };
        // SAFETY: offset=cut-2*small<product_len leaves total-cut digits.
        let retained = unsafe { full.get_unchecked_mut(offset..) };
        for (outer, low, tail_width) in [(a, b0, b_width), (b, a0, a_width)] {
            if top_start >= outer.len() {
                continue;
            }
            // SAFETY: the comparison bounds the nonempty high prefix. Its
            // product width is covered by this level's checked arena prefix;
            // the disjoint suffix accommodates all recursive descendants.
            let (high, high_width, child, child_work) = unsafe {
                let high = outer.get_unchecked(top_start..);
                let high_width = NonZeroUsize::new_unchecked(high.len());
                let child_len = high_width.get().unchecked_add(small);
                let (child, child_work) = work.split_at_mut_unchecked(child_len);
                (high, high_width, child, child_work)
            };
            // SAFETY: low has small limbs, high is nonempty, and small is
            // below their total width. The input/output/work owners are disjoint.
            let (start, _) = unsafe {
                Self::high_product_blocks(high, low, small, child, child_work, mul_scratch)
            };
            // SAFETY: skipping small limbs from high.len()+small leaves
            // exactly high_width>0 initialized digits starting at start.
            // Their width is outer.len()-top_start<=total-cut.
            let cross = unsafe { child.as_ptr().add(start).cast::<Limb>() };
            let carry = if high_width.get() < 2 {
                // SAFETY: the nonzero width is one, within both initialized
                // disjoint spans. The scalar sum initializes the same output.
                unsafe {
                    let first_output = retained.first_mut().unwrap_unchecked();
                    let (sum, overflow) = first_output.overflowing_add(cross.read());
                    *first_output = sum;
                    Limb::from(overflow)
                }
            } else {
                // SAFETY: both disjoint initialized spans cover high_width
                // aligned limbs. This branch gives at least two addends.
                unsafe {
                    ArchKernels::add_limbs_unchecked(retained.as_mut_ptr(), cross, high_width.get())
                }
            };
            // SAFETY: retained.len()-high_width is exactly the other
            // operand's high-block width, established at least four above. This
            // initialized tail follows the addition prefix and stays within
            // output; nonnegative partial sums cannot overflow the product.
            let higher = unsafe {
                from_raw_parts_mut(retained.as_mut_ptr().add(high_width.get()), tail_width)
            };
            let overflow = Addition::propagate_carry(higher, carry);
            debug_assert_eq!(
                overflow, 0,
                "the lower high-product sum fits its retained width"
            );
        }
        // SAFETY: the full high block initialized product_len digits, and
        // offset+2 corresponds to skipping both low certification guards.
        unsafe { (offset.unchecked_add(2), product_len) }
    }
}
