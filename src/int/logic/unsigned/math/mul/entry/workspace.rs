//! Stack and pooled workspace execution for nonzero owned products.

#![expect(
    unsafe_code,
    reason = "Selected stack frames bound initialized workspace prefixes and preserve exclusive borrows"
)]

use core::{mem::MaybeUninit, ptr::write_bytes, slice::from_raw_parts_mut};

use super::{
    KARATSUBA_THRESHOLD, Karatsuba, Limb, MulPlan, MulScratch, Multiplication, Schoolbook,
    SquarePlan, TierCeiling,
};

/// Maximum stack workspace: two KiB on 64-bit targets.
const STACK_SCRATCH_LIMBS: usize = 256;
/// Smallest stack workspace tier.
const SMALL_STACK_SCRATCH_LIMBS: usize = STACK_SCRATCH_LIMBS.div_euclid(4);
/// Intermediate stack workspace tier.
const MEDIUM_STACK_SCRATCH_LIMBS: usize = STACK_SCRATCH_LIMBS.div_euclid(2);

impl Multiplication {
    /// Runs a nonzero product into its complete destination.
    #[inline]
    pub fn multiply_nonzero_owned(a_limbs: &[Limb], b_limbs: &[Limb], result: &mut [Limb]) {
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();
        // The selector's first rule excludes every higher tier below this width.
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
        // These balanced Karatsuba widths have compile-time scratch requirements.
        if plan == MulPlan::Karatsuba && a_len == b_len {
            let balanced = |scratch: &mut [Limb]| Karatsuba::mul(result, a_limbs, b_limbs, scratch);
            match a_len {
                20 => {
                    const FRAME: usize = Karatsuba::BALANCED_20_SCRATCH_LIMBS;
                    Self::stack_frame::<FRAME>(FRAME, balanced);
                    return;
                }
                24 => {
                    const FRAME: usize = Karatsuba::BALANCED_24_SCRATCH_LIMBS;
                    Self::stack_frame::<FRAME>(FRAME, balanced);
                    return;
                }
                32 => {
                    const FRAME: usize = Karatsuba::BALANCED_32_SCRATCH_LIMBS;
                    Self::stack_frame::<FRAME>(FRAME, balanced);
                    return;
                }
                48 => {
                    const FRAME: usize = Karatsuba::BALANCED_48_SCRATCH_LIMBS;
                    Self::stack_frame::<FRAME>(FRAME, balanced);
                    return;
                }
                _ => {}
            }
        }
        // Only lopsided and transform plans depend on the ambient worker budget.
        let scratch_len = Self::scratch_len(plan, a_len, b_len);
        let uses_transform = plan.is_transform();
        if scratch_len <= STACK_SCRATCH_LIMBS && !uses_transform {
            Self::with_stack_scratch(scratch_len, |active_scratch| {
                Self::execute_plan(plan, result, a_limbs, b_limbs, active_scratch);
            });
        } else {
            let mut scratch = MulScratch::default();
            scratch.prepare(scratch_len);
            Self::execute_plan(plan, result, a_limbs, b_limbs, &mut scratch.buf);
        }
    }

    /// Squares a nonzero operand into its complete destination.
    pub fn square_nonzero_owned(a_limbs: &[Limb], result: &mut [Limb]) {
        let a_len = a_limbs.len();
        let plan = Self::select_square_plan(a_len, TierCeiling::Full);
        if plan == SquarePlan::Schoolbook {
            Self::execute_square_plan(plan, result, a_limbs, &mut []);
            return;
        }
        // Only transform squares depend on the ambient worker budget.
        let scratch_len = Self::square_scratch_len(plan, a_len);
        let uses_transform = plan.is_transform();
        if scratch_len <= STACK_SCRATCH_LIMBS && !uses_transform {
            Self::with_stack_scratch(scratch_len, |active_scratch| {
                Self::execute_square_plan(plan, result, a_limbs, active_scratch);
            });
        } else {
            let mut scratch = MulScratch::default();
            scratch.prepare(scratch_len);
            Self::execute_square_plan(plan, result, a_limbs, &mut scratch.buf);
        }
    }

    /// Supplies the smallest initialized frame holding the requested workspace.
    #[inline]
    fn with_stack_scratch(scratch_len: usize, body: impl FnOnce(&mut [Limb])) {
        debug_assert!(
            scratch_len <= STACK_SCRATCH_LIMBS,
            "stack scratch exceeds its budget"
        );
        if scratch_len <= SMALL_STACK_SCRATCH_LIMBS {
            Self::stack_frame::<SMALL_STACK_SCRATCH_LIMBS>(scratch_len, body);
        } else if scratch_len <= MEDIUM_STACK_SCRATCH_LIMBS {
            Self::stack_frame::<MEDIUM_STACK_SCRATCH_LIMBS>(scratch_len, body);
        } else {
            Self::stack_frame::<STACK_SCRATCH_LIMBS>(scratch_len, body);
        }
    }

    /// Initializes only the active prefix of a compile-time bounded frame.
    #[inline]
    fn stack_frame<const FRAME: usize>(scratch_len: usize, body: impl FnOnce(&mut [Limb])) {
        debug_assert!(
            scratch_len <= FRAME,
            "stack frame tier does not hold scratch"
        );
        let mut stack_scratch = [MaybeUninit::<Limb>::uninit(); FRAME];
        // SAFETY: each caller selects FRAME >= scratch_len. MaybeUninit<Limb>
        // preserves Limb's alignment and layout. Zero writes initialize exactly
        // the active prefix before its exclusive typed borrow; the suffix is unread.
        let active_scratch = unsafe {
            let limbs = stack_scratch.as_mut_ptr().cast::<Limb>();
            write_bytes(limbs, 0, scratch_len);
            from_raw_parts_mut(limbs, scratch_len)
        };
        body(active_scratch);
    }
}
