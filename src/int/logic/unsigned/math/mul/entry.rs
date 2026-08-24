//! Multiplication and squaring entry points.
//!
//! Three layers, one file:
//!
//! - [`MulScratch`] and the owned-path scratch policy — how a product's workspace
//!   is sourced from the caller's stack or a pooled allocation.
//! - `impl InternalMpUint` — allocating, destination-reusing, and in-place
//!   products and squares on owned big integers.
//! - `impl Multiplication` — raw limb-slice products and squares for callers that
//!   already hold [`Limb`] spans and their own scratch.
//!
//! The owned `a * b` path establishes the complete execution contract here:
//! normalized nonzero operands, an exact `a.len() + b.len()` destination, and
//! scratch derived from the selected plan. Valid Rust slice widths prove that
//! sum cannot overflow on any supported limb width. SSA performs its remaining
//! fallible geometry arithmetic while building an operand-bound plan; execution
//! below that boundary contains only diagnostic assertions for proved invariants.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use core::ptr::eq;

use crate::parallel::{DefaultExecutor, ParallelExecutor};

use super::{
    INLINE_LIMBS, InternalMpUint, KARATSUBA_THRESHOLD, Karatsuba, Limb, MulPlan, Multiplication,
    Schoolbook, ScratchBuffer, SquarePlan, TierCeiling,
};

/// Maximum ephemeral Karatsuba workspace kept on the caller's stack.
///
/// Two hundred fifty-six limbs occupy at most two KiB on supported targets
/// (and proportionally less on narrower targets). This covers the complete
/// shallow Karatsuba region through 64-limb balanced products without a
/// second allocation while remaining modest for ordinary thread stacks.
const STACK_SCRATCH_LIMBS: usize = 256;

/// Pre-allocated scratch space for multiplication.
#[derive(Debug, Clone)]
pub struct MulScratch {
    pub buf: ScratchBuffer,
}

impl Default for MulScratch {
    fn default() -> Self {
        Self {
            buf: ScratchBuffer::acquire(0),
        }
    }
}

impl MulScratch {
    /// Exposes at least `scratch_len` limbs without rewriting reusable storage.
    ///
    /// Every multiplication plan writes or explicitly clears each scratch
    /// region before reading it. Preserving an already exposed buffer avoids an
    /// O(scratch) fill on every repeated product. Growth still initializes the
    /// newly exposed limbs once, as required by [`ScratchBuffer`]'s `Vec`-like
    /// element-validity contract.
    #[inline]
    pub fn prepare(&mut self, scratch_len: usize) {
        if self.buf.len() >= scratch_len {
            return;
        }
        self.buf.reset_with_capacity(scratch_len);
        self.buf.resize(scratch_len, Limb::MIN);
    }
}

impl InternalMpUint {
    /// Computes `self * other`.
    pub fn mul(&self, other: &Self) -> Self {
        if eq(self, other) {
            return self.square();
        }
        let a_len = self.limbs().len();
        let b_len = other.limbs().len();
        if a_len == 0 || b_len == 0 {
            return Self::zero();
        }
        if self.is_one() {
            return other.clone();
        }
        if other.is_one() {
            return self.clone();
        }
        let mut res = Self::with_capacity(a_len.wrapping_add(b_len));
        res.write_nonzero_product(self.limbs(), other.limbs());
        res
    }

    /// Computes the square `self * self`, avoiding the redundant work of a
    /// general product.
    pub fn square(&self) -> Self {
        let a_len = self.limbs().len();
        if a_len == 0 {
            return Self::zero();
        }
        let mut res = Self::with_capacity(a_len.wrapping_add(a_len));
        res.write_nonzero_square(self.limbs());
        res
    }

    /// Computes `self = a * b`, reusing `self`'s existing allocation.
    ///
    /// The destination-reusing form the `Mul` operator cannot offer: `&a * &b` has
    /// no buffer to write into and must allocate one per call. A caller
    /// multiplying in a loop supplies the same destination each time and pays that
    /// allocation once.
    ///
    /// `self` cannot alias either operand — the product is written before the
    /// operands are fully consumed — which the `&mut self` / `&Self` signature at
    /// the public boundary already enforces.
    pub fn assign_product(&mut self, a: &Self, b: &Self) {
        if eq(a, b) {
            self.assign_square(a);
            return;
        }
        let a_len = a.limbs().len();
        let b_len = b.limbs().len();
        if a_len == 0 || b_len == 0 {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }
        if a.is_one() {
            self.clone_from(b);
            return;
        }
        if b.is_one() {
            self.clone_from(a);
            return;
        }
        self.write_nonzero_product(a.limbs(), b.limbs());
    }

    /// Computes `self = a * a`, reusing `self`'s existing allocation.
    ///
    /// See [`Self::assign_product`] for why the destination-reusing form exists.
    pub fn assign_square(&mut self, a: &Self) {
        let a_limbs = a.limbs();
        if a_limbs.is_empty() {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }
        self.write_nonzero_square(a_limbs);
    }

    /// Computes `self = a * b` using a caller-owned scratch pool.
    pub fn assign_product_with_scratch(&mut self, a: &Self, b: &Self, scratch: &mut MulScratch) {
        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        if a_limbs.is_empty() || b_limbs.is_empty() {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }

        let res_len = a_limbs.len().wrapping_add(b_limbs.len());
        // SAFETY: the limb dispatcher fills the complete result slice before it is read.
        let result = unsafe { self.ensure_capacity_set_len_get_limbs(res_len) };
        Multiplication::mul_limbs_with_scratch(a_limbs, b_limbs, result, scratch);
        self.trim_product_guard(res_len);
    }

    /// Computes `self = a * a` using a caller-owned scratch pool.
    pub fn assign_square_with_scratch(&mut self, a: &Self, scratch: &mut MulScratch) {
        let a_limbs = a.limbs();
        if a_limbs.is_empty() {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }

        let res_len = a_limbs.len().wrapping_add(a_limbs.len());
        // SAFETY: the limb dispatcher fills the complete result slice before it is read.
        let result = unsafe { self.ensure_capacity_set_len_get_limbs(res_len) };
        Multiplication::sqr_limbs_with_scratch(a_limbs, result, scratch);
        self.trim_product_guard(res_len);
    }

    /// Multiplies in place by `other`.
    #[inline]
    pub fn mul_assign(&mut self, other: &Self) {
        let a_len = self.limbs().len();
        let b_len = other.limbs().len();
        if a_len == 0 || b_len == 0 {
            // SAFETY: setting len to 0 preserves every representation invariant.
            unsafe {
                self.set_len(0);
            }
            return;
        }
        if other.is_one() {
            return;
        }
        if self.is_one() {
            self.clone_from(other);
            return;
        }
        if b_len == 1 {
            // The one-limb multiplier is independent of the destination. Growing
            // by one preserves the initialized prefix; the scalar kernel may then
            // overwrite that prefix in place because it consumes limbs low to high.
            // SAFETY: this branch proves b_len == 1.
            let scalar = unsafe { *other.limbs().get_unchecked(0) };
            let res_len = a_len.wrapping_add(1);
            // SAFETY: the scalar kernel initializes all res_len limbs and permits
            // exact source/destination aliasing.
            let result = unsafe { self.ensure_capacity_set_len_get_limbs(res_len) };
            // SAFETY: result is writable for a_len+1 limbs, its initialized prefix
            // is readable for a_len limbs, and exact aliasing is supported.
            unsafe {
                Schoolbook::mul_limb_unchecked(result.as_mut_ptr(), result.as_ptr(), a_len, scalar);
            }
            self.normalize();
            return;
        }
        if a_len == 1 {
            // SAFETY: this branch proves a_len == 1.
            let scalar = unsafe { *self.limbs().get_unchecked(0) };
            let res_len = b_len.wrapping_add(1);
            // SAFETY: the separate source has b_len initialized limbs and the
            // scalar kernel writes the complete b_len+1-limb result.
            let result = unsafe { self.ensure_capacity_set_len_get_limbs(res_len) };
            // SAFETY: result and other are disjoint because other is independently
            // borrowed, and both spans satisfy the scalar-kernel contract.
            unsafe {
                Schoolbook::mul_limb_unchecked(
                    result.as_mut_ptr(),
                    other.limbs().as_ptr(),
                    b_len,
                    scalar,
                );
            }
            self.normalize();
            return;
        }
        // The destination is also an operand, so the multiplicand has to be
        // preserved before `result` overwrites it. Narrow values stage through an
        // inline array; wider ones through a pooled buffer.
        let res_len = a_len.wrapping_add(b_len);
        if a_len <= INLINE_LIMBS {
            let mut inline_a = [0; INLINE_LIMBS];
            // SAFETY: a_len is the current initialized limb length.
            let src_slice = unsafe { self.limbs().get_unchecked(..a_len) };
            // SAFETY: a_len <= INLINE_LIMBS by this branch.
            let dst_slice = unsafe { inline_a.get_unchecked_mut(..a_len) };
            dst_slice.copy_from_slice(src_slice);
            // SAFETY: a_len <= INLINE_LIMBS by this branch.
            let slice_a = unsafe { inline_a.get_unchecked(..a_len) };
            // SAFETY: the limb dispatcher fills the complete result slice before it is read.
            let result = unsafe { self.ensure_capacity_set_len_get_limbs(res_len) };
            multiply_nonzero_owned(slice_a, other.limbs(), result);
        } else {
            let mut saved_a = ScratchBuffer::acquire(a_len);
            saved_a.extend_from_slice(self.limbs());
            // SAFETY: the limb dispatcher fills the complete result slice before it is read.
            let result = unsafe { self.ensure_capacity_set_len_get_limbs(res_len) };
            multiply_nonzero_owned(&saved_a, other.limbs(), result);
        }
        self.normalize();
    }

    /// Consumes both operands, multiplying into whichever already owns the larger
    /// allocation.
    #[inline]
    pub fn mul_into(mut self, mut other: Self) -> Self {
        if self.capacity() >= other.capacity() {
            self.mul_assign(&other);
            self
        } else {
            other.mul_assign(&self);
            other
        }
    }

    /// Writes a nonzero product.
    ///
    /// Nonzero normalized `m`- and `n`-limb operands have a product of at least
    /// `m + n - 1` limbs, so only the highest allocated limb can be zero.
    fn write_nonzero_product(&mut self, a_limbs: &[Limb], b_limbs: &[Limb]) {
        let result_len = a_limbs.len().wrapping_add(b_limbs.len());
        // SAFETY: multiplication initializes every exposed result limb.
        let result = unsafe { self.ensure_capacity_set_len_get_limbs(result_len) };
        multiply_nonzero_owned(a_limbs, b_limbs, result);
        self.trim_product_guard(result_len);
    }

    /// Writes a nonzero square.
    ///
    /// A nonzero normalized `n`-limb value has a square of `2n` or `2n - 1`
    /// limbs, so at most one high guard limb can be zero.
    fn write_nonzero_square(&mut self, a_limbs: &[Limb]) {
        let result_len = a_limbs.len().wrapping_add(a_limbs.len());
        // SAFETY: squaring initializes every exposed result limb.
        let result = unsafe { self.ensure_capacity_set_len_get_limbs(result_len) };
        square_nonzero_owned(a_limbs, result);
        self.trim_product_guard(result_len);
    }

    /// Drops the single guard limb a product or square may leave zero.
    ///
    /// Cheaper than [`Self::normalize`], which scans for an arbitrary run of high
    /// zero limbs; the product lower bound proves at most one can be zero here.
    #[inline]
    fn trim_product_guard(&mut self, result_len: usize) {
        let top_is_zero = self.limbs().last().is_some_and(|top_limb| *top_limb == 0);
        // SAFETY: the algorithm initialized every limb below result_len, and the
        // product lower bound proves the trimmed length is result_len or
        // result_len - 1.
        unsafe {
            self.set_len(result_len.wrapping_sub(usize::from(top_is_zero)));
        }
    }
}

impl Multiplication {
    /// Scratch one [`Self::mul_limbs_with_slice_scratch`] call needs for these
    /// operand widths.
    #[inline]
    pub fn required_scratch(a_len: usize, b_len: usize) -> usize {
        DefaultExecutor::with_resolved(|executor| {
            Self::required_scratch_for_parallelism(a_len, b_len, executor.parallelism().get())
        })
    }

    /// Scratch required for these operand widths at one executor width.
    #[inline]
    pub fn required_scratch_for_parallelism(
        a_len: usize,
        b_len: usize,
        parallelism: usize,
    ) -> usize {
        let plan = Self::select_plan(a_len, b_len, TierCeiling::Full);
        Self::scratch_len_for_parallelism(plan, a_len, b_len, parallelism)
    }

    /// Scratch one [`Self::sqr_limbs_with_slice_scratch`] call needs for this
    /// operand width.
    #[inline]
    pub fn required_sqr_scratch(len: usize) -> usize {
        DefaultExecutor::with_resolved(|executor| {
            Self::required_sqr_scratch_for_parallelism(len, executor.parallelism().get())
        })
    }

    /// Square scratch required for this operand width at one executor width.
    #[inline]
    pub fn required_sqr_scratch_for_parallelism(len: usize, parallelism: usize) -> usize {
        let plan = Self::select_square_plan(len, TierCeiling::Full);
        Self::square_scratch_len_for_parallelism(plan, len, parallelism)
    }

    /// Multiplies `a_limbs` by `b_limbs` into `result` using caller-sized scratch.
    ///
    /// `result` must hold at least `a_limbs.len() + b_limbs.len()` limbs and
    /// `scratch` at least [`Self::required_scratch`] limbs; an empty operand
    /// zeroes `result`.
    ///
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
        // Each valid slice spans at most `isize::MAX` bytes and a limb occupies
        // at least two bytes on every supported target, so the sum of two limb
        // counts is strictly below `usize::MAX`.
        debug_assert!(
            result.len() >= a_limbs.len().wrapping_add(b_limbs.len()),
            "the caller-sized multiplication destination is exact or wider"
        );
        if eq(a_limbs.as_ptr(), b_limbs.as_ptr()) && a_limbs.len() == b_limbs.len() {
            Self::sqr_limbs_with_slice_scratch(a_limbs, result, scratch);
            return;
        }

        // The selector's first rule, hoisted for the same reason as in
        // `mul_limbs_with_scratch`: this is the entry point the SSA pointwise stage
        // and the lopsided tail reach, on small operands, in a loop. No later
        // predicate can override a sub-Karatsuba operand.
        if a_limbs.len() < KARATSUBA_THRESHOLD || b_limbs.len() < KARATSUBA_THRESHOLD {
            Schoolbook::mul_nonempty_distinct(result, a_limbs, b_limbs);
            return;
        }
        let plan = Self::select_plan(a_limbs.len(), b_limbs.len(), TierCeiling::Full);
        DefaultExecutor::with_resolved(|executor| {
            debug_assert!(
                scratch.len()
                    >= Self::scratch_len_for_parallelism(
                        plan,
                        a_limbs.len(),
                        b_limbs.len(),
                        executor.parallelism().get(),
                    ),
                "the caller-sized multiplication scratch matches the active executor"
            );
            Self::execute_plan_with_executor(plan, result, a_limbs, b_limbs, scratch, executor);
        });
    }

    /// Multiplies `a_limbs` by `b_limbs` into `result`, growing a caller-owned
    /// scratch pool to whatever the selected tier needs.
    ///
    /// `result` must hold at least `a_limbs.len() + b_limbs.len()` limbs; an empty
    /// operand zeroes it. Public so the benchmark facade can drive the real
    /// dispatcher without paying a pool acquisition inside its timed region.
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

        // This is the selector's first rule. Hoisting it avoids all shape and enum
        // dispatch overhead in the quadratic tier without changing policy: no
        // later predicate can override a sub-Karatsuba operand.
        if a_len < KARATSUBA_THRESHOLD || b_len < KARATSUBA_THRESHOLD {
            Schoolbook::mul_nonempty_distinct(result, a_limbs, b_limbs);
            return;
        }
        let plan = Self::select_plan(a_len, b_len, TierCeiling::Full);
        debug_assert_ne!(
            plan,
            MulPlan::Schoolbook,
            "the hoisted basecase rule must exclude a schoolbook plan"
        );
        DefaultExecutor::with_resolved(|executor| {
            let scratch_len =
                Self::scratch_len_for_parallelism(plan, a_len, b_len, executor.parallelism().get());
            if scratch_len > 0 {
                scratch.prepare(scratch_len);
            }
            Self::execute_plan_with_executor(
                plan,
                result,
                a_limbs,
                b_limbs,
                &mut scratch.buf,
                executor,
            );
        });
    }

    /// Squares `a_limbs` into `result` using caller-sized scratch.
    ///
    /// `result` must hold at least `2 * a_limbs.len()` limbs and `scratch` at least
    /// [`Self::required_sqr_scratch`] limbs; an empty operand zeroes `result`.
    ///
    pub fn sqr_limbs_with_slice_scratch(
        a_limbs: &[Limb],
        result: &mut [Limb],
        scratch: &mut [Limb],
    ) {
        if a_limbs.is_empty() {
            result.fill(0);
            return;
        }
        // A valid limb slice spans at most `isize::MAX` bytes and a limb occupies
        // at least two bytes, so doubling its element count cannot overflow.
        debug_assert!(
            result.len() >= a_limbs.len().wrapping_mul(2),
            "the caller-sized square destination is exact or wider"
        );

        let plan = Self::select_square_plan(a_limbs.len(), TierCeiling::Full);
        DefaultExecutor::with_resolved(|executor| {
            debug_assert!(
                scratch.len()
                    >= Self::square_scratch_len_for_parallelism(
                        plan,
                        a_limbs.len(),
                        executor.parallelism().get(),
                    ),
                "the caller-sized square scratch matches the active executor"
            );
            Self::execute_square_plan_with_executor(plan, result, a_limbs, scratch, executor);
        });
    }

    /// Squares `a_limbs` into `result`, reusing a caller-owned scratch pool.
    ///
    /// Selects and runs the configured squaring tier, growing `scratch` to whatever
    /// that tier needs. `result` must hold at least `2 * a_limbs.len()` limbs; an
    /// empty operand zeroes it. Public so the benchmark facade can drive the real
    /// dispatcher without paying a pool acquisition inside its timed region.
    pub fn sqr_limbs_with_scratch(a_limbs: &[Limb], result: &mut [Limb], scratch: &mut MulScratch) {
        if a_limbs.is_empty() {
            result.fill(0);
            return;
        }
        let plan = Self::select_square_plan(a_limbs.len(), TierCeiling::Full);
        DefaultExecutor::with_resolved(|executor| {
            let scratch_len = Self::square_scratch_len_for_parallelism(
                plan,
                a_limbs.len(),
                executor.parallelism().get(),
            );
            if scratch_len > 0 {
                scratch.prepare(scratch_len);
            }
            Self::execute_square_plan_with_executor(
                plan,
                result,
                a_limbs,
                &mut scratch.buf,
                executor,
            );
        });
    }
}

/// Runs a nonzero product.
fn multiply_nonzero_owned(a_limbs: &[Limb], b_limbs: &[Limb], result: &mut [Limb]) {
    let a_len = a_limbs.len();
    let b_len = b_limbs.len();
    let plan = Multiplication::select_plan(a_len, b_len, TierCeiling::Full);
    if plan == MulPlan::Schoolbook {
        Multiplication::execute_plan(plan, result, a_limbs, b_limbs, &mut []);
        return;
    }
    // The balanced Karatsuba widths the owned path sees most often have a scratch
    // requirement known at compile time, so each takes an exactly sized stack
    // frame instead of the conservative `STACK_SCRATCH_LIMBS` one.
    if plan == MulPlan::Karatsuba && a_len == b_len {
        match a_len {
            20 => {
                karatsuba_on_stack::<{ Karatsuba::BALANCED_20_SCRATCH_LIMBS }>(
                    result, a_limbs, b_limbs,
                );
                return;
            }
            32 => {
                karatsuba_on_stack::<{ Karatsuba::BALANCED_32_SCRATCH_LIMBS }>(
                    result, a_limbs, b_limbs,
                );
                return;
            }
            48 => {
                karatsuba_on_stack::<{ Karatsuba::BALANCED_48_SCRATCH_LIMBS }>(
                    result, a_limbs, b_limbs,
                );
                return;
            }
            _ => {}
        }
    }

    DefaultExecutor::with_resolved(|executor| {
        let scratch_len = Multiplication::scratch_len_for_parallelism(
            plan,
            a_len,
            b_len,
            executor.parallelism().get(),
        );
        // A transform tier's scratch is far past any stack budget, and asking for it
        // by value would reserve the frame whether or not the branch is taken.
        let uses_transform = plan.is_transform();
        if scratch_len <= STACK_SCRATCH_LIMBS && !uses_transform {
            with_stack_scratch(scratch_len, |active_scratch| {
                Multiplication::execute_plan_with_executor(
                    plan,
                    result,
                    a_limbs,
                    b_limbs,
                    active_scratch,
                    executor,
                );
            });
        } else {
            let mut scratch = MulScratch::default();
            scratch.prepare(scratch_len);
            Multiplication::execute_plan_with_executor(
                plan,
                result,
                a_limbs,
                b_limbs,
                &mut scratch.buf,
                executor,
            );
        }
    });
}

/// Squares a nonzero operand into `result`, selecting and running the plan.
fn square_nonzero_owned(a_limbs: &[Limb], result: &mut [Limb]) {
    let a_len = a_limbs.len();
    let plan = Multiplication::select_square_plan(a_len, TierCeiling::Full);
    if plan == SquarePlan::Schoolbook {
        Multiplication::execute_square_plan(plan, result, a_limbs, &mut []);
        return;
    }
    DefaultExecutor::with_resolved(|executor| {
        let scratch_len = Multiplication::square_scratch_len_for_parallelism(
            plan,
            a_len,
            executor.parallelism().get(),
        );
        let uses_transform = plan.is_transform();
        if scratch_len <= STACK_SCRATCH_LIMBS && !uses_transform {
            with_stack_scratch(scratch_len, |active_scratch| {
                Multiplication::execute_square_plan_with_executor(
                    plan,
                    result,
                    a_limbs,
                    active_scratch,
                    executor,
                );
            });
        } else {
            let mut scratch = MulScratch::default();
            scratch.prepare(scratch_len);
            Multiplication::execute_square_plan_with_executor(
                plan,
                result,
                a_limbs,
                &mut scratch.buf,
                executor,
            );
        }
    });
}

/// Runs `body` against `scratch_len` limbs of initialized stack scratch.
///
/// `scratch_len` must not exceed [`STACK_SCRATCH_LIMBS`], which both callers
/// establish by comparison immediately before calling.
#[inline]
fn with_stack_scratch(scratch_len: usize, body: impl FnOnce(&mut [Limb])) {
    debug_assert!(
        scratch_len <= STACK_SCRATCH_LIMBS,
        "stack scratch is bounded by STACK_SCRATCH_LIMBS"
    );
    let mut stack_scratch = [Limb::MIN; STACK_SCRATCH_LIMBS];
    // SAFETY: scratch_len <= STACK_SCRATCH_LIMBS is asserted above.
    let active_scratch = unsafe { stack_scratch.get_unchecked_mut(..scratch_len) };
    body(active_scratch);
}

/// Runs balanced Karatsuba against a stack frame sized exactly for `N` limbs.
#[inline]
fn karatsuba_on_stack<const N: usize>(result: &mut [Limb], a_limbs: &[Limb], b_limbs: &[Limb]) {
    let mut stack_scratch = [Limb::MIN; N];
    Karatsuba::mul(result, a_limbs, b_limbs, &mut stack_scratch);
}

#[cfg(test)]
#[path = "tests/entry.rs"]
mod tests;
