//! Limb-slice products, squares, and reusable workspace preparation.
//!
//! Every entry point validates emptiness and aliasing once, hoists the
//! selector's basecase rule so quadratic products skip plan selection, and
//! then sizes and runs exactly the plan the dispatcher names.

#![expect(
    unsafe_code,
    reason = "Valid slice byte bounds prove product widths fit; reserved scratch capacity bounds write-only suffix initialization and length commits"
)]

use core::{
    mem::MaybeUninit,
    ptr::{eq, write_bytes},
};

#[cfg(not(target_pointer_width = "16"))]
use crate::parallel::{DefaultExecutor, ParallelExecutor};

#[cfg(any(test, not(target_pointer_width = "16")))]
use super::SquarePlan;
use super::{
    KARATSUBA_THRESHOLD, Limb, LimbOutput, MulPlan, Multiplication, SQR_KARATSUBA_THRESHOLD,
    Schoolbook, ScratchBuffer, TierCeiling,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, Ssa, SsaMultiplicationPlan, TransformChoice};

/// Caller-owned pooled workspace for repeated multiplication and squaring.
#[derive(Debug, Clone)]
pub struct MulScratch {
    pub buf: ScratchBuffer,
}

impl Multiplication {
    /// Scratch required by the selected product plan for these operand widths.
    #[cfg(any(test, not(target_pointer_width = "16")))]
    #[inline]
    pub fn required_scratch(a_len: usize, b_len: usize) -> usize {
        Self::scratch_len(
            Self::select_plan(a_len, b_len, TierCeiling::Full),
            a_len,
            b_len,
        )
    }

    /// Scratch required for these operand widths at one executor width.
    #[cfg(not(target_pointer_width = "16"))]
    #[inline]
    pub fn required_scratch_for_parallelism(
        a_len: usize,
        b_len: usize,
        parallelism: usize,
    ) -> usize {
        let plan = Self::select_plan(a_len, b_len, TierCeiling::Full);
        Self::scratch_len_for_parallelism(plan, a_len, b_len, parallelism)
    }

    /// Square scratch required for this operand width at one executor width.
    #[cfg(not(target_pointer_width = "16"))]
    #[inline]
    pub fn required_sqr_scratch_for_parallelism(len: usize, parallelism: usize) -> usize {
        let plan = Self::select_square_plan(len, TierCeiling::Full);
        Self::square_scratch_len_for_parallelism(plan, len, parallelism)
    }

    /// Multiplies `a_limbs` by `b_limbs` into `result` using caller-sized scratch.
    ///
    /// `result` must hold at least `a_limbs.len() + b_limbs.len()` limbs and
    /// `scratch` at least [`Self::required_scratch`] limbs; an empty operand
    /// zeroes `result`. Identical input slices use the square plan when the
    /// supplied scratch accommodates it; otherwise the product plan applies.
    #[cfg(any(test, not(target_pointer_width = "16")))]
    pub fn mul_limbs_with_slice_scratch(
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        result: &mut [Limb],
        scratch: &mut [Limb],
    ) {
        if a_limbs.is_empty() || b_limbs.is_empty() {
            result.fill(0);
            return;
        }
        debug_assert!(
            result.len()
                // SAFETY: each valid slice spans at most isize::MAX bytes and
                // Limb occupies at least two bytes on 16/32/64-bit targets.
                // Their combined limb count is therefore below usize::MAX.
                >= unsafe { a_limbs.len().unchecked_add(b_limbs.len()) },
            "the caller-sized multiplication destination is exact or wider"
        );
        if eq(a_limbs.as_ptr(), b_limbs.as_ptr()) && a_limbs.len() == b_limbs.len() {
            let square_plan = Self::select_square_plan(a_limbs.len(), TierCeiling::Full);
            if scratch.len() >= Self::square_scratch_len(square_plan, a_limbs.len()) {
                Self::execute_square_plan(square_plan, result, a_limbs, scratch);
                return;
            }
            // A tuning profile may put the square crossover below the product
            // crossover. The basecase square needs no workspace in that case.
            if a_limbs.len() < KARATSUBA_THRESHOLD {
                Self::execute_square_plan(SquarePlan::Schoolbook, result, a_limbs, scratch);
                return;
            }
        }

        // The selector excludes every higher tier below the Karatsuba width.
        if a_limbs.len() < KARATSUBA_THRESHOLD || b_limbs.len() < KARATSUBA_THRESHOLD {
            Schoolbook::mul_nonempty_distinct(result, a_limbs, b_limbs);
            return;
        }
        let plan = Self::select_plan(a_limbs.len(), b_limbs.len(), TierCeiling::Full);
        debug_assert!(
            scratch.len() >= Self::scratch_len(plan, a_limbs.len(), b_limbs.len()),
            "the caller-sized multiplication scratch matches the active executor"
        );
        Self::execute_plan(plan, result, a_limbs, b_limbs, scratch);
    }

    /// Multiplies `a_limbs` by `b_limbs` into `result`, growing a caller-owned
    /// scratch pool to whatever the selected tier needs.
    ///
    /// `result` must hold at least `a_limbs.len() + b_limbs.len()` limbs; an empty
    /// operand zeroes it. Reuses caller-provided scratch storage to avoid dynamic
    /// buffer reallocation across successive multiplications.
    #[inline]
    pub fn mul_limbs_with_scratch(
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        result: &mut [Limb],
        scratch: &mut MulScratch,
    ) {
        if a_limbs.is_empty() || b_limbs.is_empty() {
            result.fill(0);
            return;
        }
        if eq(a_limbs.as_ptr(), b_limbs.as_ptr()) && a_limbs.len() == b_limbs.len() {
            Self::sqr_limbs_with_scratch(a_limbs, result, scratch);
            return;
        }
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();

        // The selector excludes every higher tier below the Karatsuba width.
        if a_len < KARATSUBA_THRESHOLD || b_len < KARATSUBA_THRESHOLD {
            Schoolbook::mul_nonempty_distinct(result, a_limbs, b_limbs);
            return;
        }
        Self::mul_planned_with_scratch(a_limbs, b_limbs, result, scratch);
    }

    /// Initializes the exact product directly in newly reserved storage.
    ///
    /// Every tier writes its endpoint and point-product spans before reading
    /// them. Only reconstruction gaps receive their required initial zeros.
    /// Scratch remains caller-owned and is reused across calls.
    /// The returned borrow covers exactly `a.len()+b.len()` initialized limbs;
    /// any surplus destination capacity is untouched.
    ///
    /// # Safety
    /// Both operands are nonempty, initialized and disjoint. `output` is
    /// disjoint from them and contains their complete product width. These
    /// invariants are established by the enclosing arithmetic operation.
    pub unsafe fn mul_nonempty_distinct_into_uninit<'output>(
        a: &[Limb],
        b: &[Limb],
        output: &'output mut [MaybeUninit<Limb>],
        scratch: &mut MulScratch,
    ) -> &'output mut [Limb] {
        debug_assert!(
            !a.is_empty() && !b.is_empty(),
            "the product initializer receives two nonempty operands"
        );
        // SAFETY: each input occupies at most isize::MAX bytes, with at least
        // two bytes per limb. Their summed limb widths therefore fit usize on
        // 16/32/64-bit targets; the caller reserves that complete output span.
        let product_len = unsafe { a.len().unchecked_add(b.len()) };
        debug_assert!(
            output.len() >= product_len,
            "the complete product is reserved"
        );
        // SAFETY: the caller reserves product_len disjoint writable elements.
        let product = unsafe { output.get_unchecked_mut(..product_len) };
        if a.len() < KARATSUBA_THRESHOLD || b.len() < KARATSUBA_THRESHOLD {
            Schoolbook::mul_nonempty_distinct(product, a, b);
        } else {
            Self::mul_planned_with_scratch(a, b, product, scratch);
        }
        // SAFETY: the selected kernel established every exact-product limb:
        // direct first writes cover products, and interpolation gaps receive
        // their first zeros before any coefficient accumulation reads them.
        unsafe { LimbOutput::assume_init_mut(product) }
    }

    /// Plans a non-basecase product after root emptiness and alias validation.
    ///
    /// Keeping plan selection and workspace growth in their own frame leaves
    /// the quadratic entry independent of the transform planner's saved registers.
    #[inline(never)]
    fn mul_planned_with_scratch(
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        result: &mut [impl LimbOutput],
        scratch: &mut MulScratch,
    ) {
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();
        debug_assert!(
            a_len >= KARATSUBA_THRESHOLD && b_len >= KARATSUBA_THRESHOLD,
            "the root basecase gate admits only operands at the Karatsuba crossover"
        );
        let plan = Self::select_plan(a_len, b_len, TierCeiling::Full);
        debug_assert_ne!(
            plan,
            MulPlan::Schoolbook,
            "the hoisted basecase rule must exclude a schoolbook plan"
        );
        #[cfg(not(target_pointer_width = "16"))]
        let conventional_plan = if plan.is_transform() {
            let computed = DefaultExecutor::with_resolved(|executor| {
                let Some(prepared) = SsaMultiplicationPlan::try_new(
                    a_limbs,
                    b_limbs,
                    TransformChoice::PLANNED,
                    executor.parallelism(),
                ) else {
                    return false;
                };
                scratch.prepare(prepared.scratch_len);
                // SAFETY: the operand-bound plan sized this exact arena;
                // the root supplied a complete, disjoint product destination.
                // The synchronous executor retains its construction-time
                // worker budget, and every output limb receives its first write.
                unsafe { prepared.run_with_scratch(result, &mut scratch.buf, executor) }
                true
            });
            if computed {
                return;
            }
            // Unrepresentable transform workspace retains the conventional
            // fallback without reading or exposing the unwritten destination.
            Self::select_plan(a_len, b_len, TierCeiling::Toom6)
        } else {
            plan
        };
        #[cfg(target_pointer_width = "16")]
        let conventional_plan = plan;
        // Conventional tiers resolve the ambient pool only for lopsided
        // products. The transform arm above binds its plan once and uses that
        // same geometry for workspace sizing and execution.
        let scratch_len = Self::scratch_len(conventional_plan, a_len, b_len);
        scratch.prepare(scratch_len);
        Self::execute_plan(
            conventional_plan,
            result,
            a_limbs,
            b_limbs,
            &mut scratch.buf,
        );
    }

    /// Multiply `a_limbs` by `x_limbs` into `out_a`, and `b_limbs` by `x_limbs` into `out_b`.
    ///
    /// If both multiplications select `LargePlan::Ssa` with matching CRT geometry,
    /// this executes a fused SSA two-by-one product, sharing the forward FFT of `x_limbs`.
    /// Otherwise, it performs two normal multiplications with `scratch`.
    pub fn mul_two_by_one(
        a_limbs: &[Limb],
        b_limbs: &[Limb],
        x_limbs: &[Limb],
        out_a: &mut [Limb],
        out_b: &mut [Limb],
        scratch: &mut MulScratch,
    ) {
        #[cfg(not(target_pointer_width = "16"))]
        {
            let a_len = a_limbs.len();
            let b_len = b_limbs.len();
            let x_len = x_limbs.len();

            // Fusion requires two transform plans and compatible CRT layouts.
            // A zero scratch length rejects an unavailable shared layout.
            if a_len >= KARATSUBA_THRESHOLD
                && b_len >= KARATSUBA_THRESHOLD
                && x_len >= KARATSUBA_THRESHOLD
                && Self::select_plan(a_len, x_len, TierCeiling::Full)
                    == MulPlan::Large(LargePlan::Ssa)
                && Self::select_plan(b_len, x_len, TierCeiling::Full)
                    == MulPlan::Large(LargePlan::Ssa)
            {
                let mut success = false;
                DefaultExecutor::with_resolved(|executor| {
                    let needed_scratch = Ssa::mul_two_by_one_scratch_len_for_parallelism(
                        a_len,
                        b_len,
                        x_len,
                        executor.parallelism().get(),
                    );
                    if needed_scratch > 0 {
                        scratch.prepare(needed_scratch);
                        success = Ssa::try_mul_two_by_one_with_executor(
                            out_a,
                            out_b,
                            a_limbs,
                            b_limbs,
                            x_limbs,
                            TransformChoice::PLANNED,
                            &mut scratch.buf,
                            executor,
                        );
                    }
                });
                if success {
                    return;
                }
            }
        }

        Self::mul_limbs_with_scratch(a_limbs, x_limbs, out_a, scratch);
        Self::mul_limbs_with_scratch(b_limbs, x_limbs, out_b, scratch);
    }

    /// Squares `a_limbs` into `result`, reusing a caller-owned scratch pool.
    ///
    /// Selects and runs the configured squaring tier, growing `scratch` to whatever
    /// that tier needs. `result` must hold at least `2 * a_limbs.len()` limbs; an
    /// empty operand zeroes it. Reuses caller-provided scratch storage to avoid
    /// dynamic buffer reallocation across successive squaring operations.
    pub fn sqr_limbs_with_scratch(a_limbs: &[Limb], result: &mut [Limb], scratch: &mut MulScratch) {
        if a_limbs.is_empty() {
            result.fill(0);
            return;
        }
        // The selector excludes every higher square tier below this width.
        if a_limbs.len() < SQR_KARATSUBA_THRESHOLD {
            // The schoolbook square writes exactly `2 * len` limbs. A caller may
            // hand this entry point a destination wider than the exact square, so
            // the disjoint guard suffix above it is initialized here, exactly as
            // `execute_square_plan_with_executor` does for the schoolbook tier.
            // SAFETY: valid slices span at most isize::MAX bytes and a Limb
            // occupies at least two bytes on 16/32/64-bit targets. Therefore
            // 2*a_limbs.len() <= isize::MAX < usize::MAX.
            let square_len = unsafe { a_limbs.len().unchecked_mul(2) };
            if result.len() > square_len {
                let (_, guard) = result.split_at_mut(square_len);
                guard.fill(0);
            }
            Schoolbook::sqr(result, a_limbs);
            return;
        }
        let plan = Self::select_square_plan(a_limbs.len(), TierCeiling::Full);
        let scratch_len = Self::square_scratch_len(plan, a_limbs.len());
        scratch.prepare(scratch_len);
        Self::execute_square_plan(plan, result, a_limbs, &mut scratch.buf);
    }
}

impl MulScratch {
    /// Exposes at least `scratch_len` initialized limbs, preserving reusable storage.
    ///
    /// The exposed `[Limb]` workspace must be initialized even when a kernel's
    /// first write replaces it. An already initialized prefix needs no fill;
    /// callers can initialize staging directly before this boundary. Growth
    /// initializes only the remaining new suffix before committing its length.
    #[inline]
    pub fn prepare(&mut self, scratch_len: usize) {
        let old_len = self.buf.len();
        if old_len >= scratch_len {
            return;
        }
        let initialized_len = if self.buf.capacity() < scratch_len {
            self.buf = ScratchBuffer::acquire(scratch_len);
            0
        } else {
            old_len
        };
        // SAFETY: acquisition returns empty storage with capacity >= scratch_len;
        // reuse retains the initialized old_len-limb prefix and sufficient capacity.
        // In either branch initialized_len < scratch_len <= capacity. The exclusive
        // buffer owns the aligned allocation; zero initializes every new Limb.
        // The complete prefix is initialized before its length is committed.
        unsafe {
            write_bytes(
                self.buf.as_mut_ptr().add(initialized_len),
                0,
                scratch_len.unchecked_sub(initialized_len),
            );
            self.buf.set_len(scratch_len);
        }
    }
}

impl Default for MulScratch {
    fn default() -> Self {
        Self {
            buf: ScratchBuffer::acquire(0),
        }
    }
}
