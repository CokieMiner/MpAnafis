//! The Karatsuba driver: general splits and the fixed-width specializations.
//!
//! Reference: A. Karatsuba and Yu. Ofman, "Multiplication of many-digital
//! numbers by automatic computers", Doklady Akademii Nauk SSSR 145(2),
//! 293-294, 1962. <https://www.mathnet.ru/eng/dan26729>

#![expect(
    unsafe_code,
    reason = "Validated Karatsuba layouts reserve disjoint endpoints, guarded middle products, and complete recursive workspaces"
)]

use core::cmp::{max, min};

use super::{
    KARATSUBA_THRESHOLD, Limb, LimbOutput, Multiplication, SQR_KARATSUBA_THRESHOLD, Schoolbook,
    SharedEval,
};

/// Namespace for the Karatsuba multiplication and squaring tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Karatsuba;

impl Karatsuba {
    /// Scratch limbs required by the exact balanced 20-limb specialization.
    pub const BALANCED_20_SCRATCH_LIMBS: usize = 2 * 20 + 1;

    /// Scratch limbs required by the exact balanced 24-limb specialization.
    pub const BALANCED_24_SCRATCH_LIMBS: usize = 2 * 24 + 1;

    /// Scratch limbs required by the exact balanced 32-limb specialization.
    pub const BALANCED_32_SCRATCH_LIMBS: usize = 2 * 32 + 1;

    /// Scratch limbs required by the exact balanced 48-limb specialization.
    pub const BALANCED_48_SCRATCH_LIMBS: usize = 2 * 24 + 2 * Self::BALANCED_24_SCRATCH_LIMBS;

    /// Selects schoolbook or Karatsuba multiplication at the configured crossover.
    ///
    /// `dst` must have at least `a.len() + b.len()` elements. When Karatsuba is
    /// selected, `scratch` must cover [`Multiplication::karatsuba_mul_scratch_len`].
    /// The schoolbook path requires no scratch.
    pub fn dispatch_mul(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.is_empty() || b.is_empty() {
            return;
        }
        if a.len() < KARATSUBA_THRESHOLD
            || b.len() < KARATSUBA_THRESHOLD
            || a.len() < 2
            || b.len() < 2
        {
            Schoolbook::mul(dst, a, b);
            return;
        }
        Self::mul(dst, a, b, scratch);
    }

    /// Executes one Karatsuba level regardless of the configured crossover.
    ///
    /// Recursive subproducts still use normal tier dispatch, so this isolates the
    /// cost of selecting Karatsuba at the root. A shape that cannot produce two
    /// nonempty halves remains a rectangular schoolbook multiplication.
    pub fn mul(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.is_empty() || b.is_empty() {
            return;
        }
        if a.len() < 2 || b.len() < 2 {
            Schoolbook::mul(dst, a, b);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::karatsuba_mul_scratch_len(a.len(), b.len()),
            "Karatsuba multiplication scratch is undersized"
        );
        if a.len() == b.len() {
            match a.len() {
                20 => Self::mul_balanced_basecase::<10>(dst, a, b, scratch),
                23 => Self::mul_balanced_basecase_odd::<12>(dst, a, b, scratch),
                24 => Self::mul_balanced_basecase::<12>(dst, a, b, scratch),
                32 => Self::mul_balanced_32(dst, a, b, scratch),
                48 => Self::mul_balanced_48(dst, a, b, scratch),
                len if len.is_multiple_of(2)
                    && Self::balanced_split_len(len) == len.div_ceil(2) =>
                {
                    Self::mul_balanced::<false>(dst, a, b, scratch);
                }
                _ => Self::mul_balanced::<true>(dst, a, b, scratch),
            }
            return;
        }
        let smaller_len = min(a.len(), b.len());
        let larger_len = max(a.len(), b.len());
        let split_len = larger_len.div_ceil(2);
        if smaller_len <= split_len {
            // The smaller operand would have an empty high half. This split
            // provides no three-product reduction, so use the rectangular leaf.
            Schoolbook::mul(dst, a, b);
            return;
        }
        // SAFETY: smaller_len > split_len proves both operands retain
        // nonempty high halves and both split positions are inside the inputs.
        let ((a0, a1), (b0, b1)) = unsafe {
            (
                a.split_at_unchecked(split_len),
                b.split_at_unchecked(split_len),
            )
        };

        // SAFETY: split_len < both input lengths; the sum buffers retain one
        // guard, and the validated scratch and destination contain these spans.
        let (sum_space, low_product_len, high_product_len) = unsafe {
            (
                split_len.unchecked_add(1),
                split_len.unchecked_mul(2),
                a1.len().unchecked_add(b1.len()),
            )
        };
        // SAFETY: the validated general layout reserves two sum_space-limb
        // inputs, a 2*sum_space-limb middle product, and recursive workspace.
        // Consecutive partitions give disjoint mutable evaluation buffers.
        let (a_sum_buffer, b_sum_buffer, recursive_scratch) = unsafe {
            let (a_sum, after_a) = scratch.split_at_mut_unchecked(sum_space);
            let (b_sum, tail) = after_a.split_at_mut_unchecked(sum_space);
            (a_sum, b_sum, tail)
        };
        let a_sum_len = Self::add_blocks(a_sum_buffer, a0, a1);
        let b_sum_len = Self::add_blocks(b_sum_buffer, b0, b1);
        // SAFETY: add_blocks initializes the complete input width and guard;
        // its returned active length is at most the sum_space-limb allocation.
        let ((a_sum, _), (b_sum, _)) = unsafe {
            (
                a_sum_buffer.split_at_unchecked(a_sum_len),
                b_sum_buffer.split_at_unchecked(b_sum_len),
            )
        };

        // SAFETY: the two endpoint widths sum to a.len()+b.len(), which the
        // caller reserves in dst. Consecutive partitions keep them disjoint.
        let (low_product, high_product) = unsafe {
            let (low, after_low) = dst.split_at_mut_unchecked(low_product_len);
            let (high, _) = after_low.split_at_mut_unchecked(high_product_len);
            (low, high)
        };
        Self::dispatch_mul(low_product, a0, b0, recursive_scratch);
        Self::dispatch_mul(high_product, a1, b1, recursive_scratch);
        // SAFETY: both recursive endpoint writers initialized their complete
        // disjoint spans before middle reconstruction reads either product.
        let (low_value, high_value) = unsafe {
            (
                LimbOutput::assume_init_mut(low_product),
                LimbOutput::assume_init_mut(high_product),
            )
        };

        // SAFETY: the sized local workspace reserves two sum_space-limb inputs
        // followed by their complete product. Both sums fit sum_space limbs.
        let (middle_space, middle_product_len) = unsafe {
            (
                sum_space.unchecked_mul(2),
                a_sum_len.unchecked_add(b_sum_len),
            )
        };
        // SAFETY: the validated layout retains middle_space limbs followed by
        // the maximum child workspace. The two initialized sums have lengths
        // at most sum_space, so their exact product fits middle_space limbs.
        let (middle_product, next_scratch) = unsafe {
            let (storage, tail) = recursive_scratch.split_at_mut_unchecked(middle_space);
            let (product, _) = storage.split_at_mut_unchecked(middle_product_len);
            (product, tail)
        };
        // M = (a0+a1)(b0+b1) >= a0*b0+a1*b1 bounds both differences by
        // the initialized middle-product span.
        Self::dispatch_mul(middle_product, a_sum, b_sum, next_scratch);

        // (a0+a1)(b0+b1)-a0b0-a1b1 = a0b1+a1b0 is nonnegative.
        // Both endpoint spans include their possible high zero limb.
        SharedEval::sub_two_full_slices_in_place(middle_product, low_value, high_value);
        let middle_len = SharedEval::active_len(middle_product);
        // SAFETY: active_len returns a prefix length at most middle_product.len().
        let (active_middle, _) = unsafe { middle_product.split_at_unchecked(middle_len) };
        debug_assert!(
            split_len <= dst.len() && active_middle.len() <= dst.len().saturating_sub(split_len),
            "Karatsuba middle exceeds the complete product destination"
        );
        // SAFETY: endpoint writers initialized all a.len()+b.len() output limbs.
        // Their slice byte bounds prove the sum fits usize on 16/32/64 bits.
        // Subtraction leaves the exact cross coefficient a0*b1+a1*b0, whose
        // split_len-limb radix shift fits the complete product destination.
        let _ = unsafe {
            let initialized = LimbOutput::assume_init_mut(
                dst.get_unchecked_mut(..a.len().unchecked_add(b.len())),
            );
            SharedEval::fused_add_shifted_in_place(initialized, active_middle, split_len)
        };
    }

    /// Selects schoolbook or Karatsuba squaring at the configured crossover.
    pub fn dispatch_sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.is_empty() {
            return;
        }
        if a.len() < SQR_KARATSUBA_THRESHOLD || a.len() < 2 {
            Schoolbook::sqr(dst, a);
            return;
        }
        Self::sqr(dst, a, scratch);
    }

    /// Executes one Karatsuba square level regardless of the configured crossover.
    ///
    /// Recursive squares retain normal dispatch, so only the root decision is
    /// forced. Inputs shorter than two limbs use the schoolbook square because no
    /// nontrivial two-way split exists. The active destination is overwritten
    /// completely by the endpoint squares and middle reconstruction.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.is_empty() {
            return;
        }
        if a.len() < 2 {
            Schoolbook::sqr(dst, a);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::karatsuba_sqr_scratch_len(a.len()),
            "Karatsuba squaring scratch is undersized: have {}, need {} for {} limbs",
            scratch.len(),
            Multiplication::karatsuba_sqr_scratch_len(a.len()),
            a.len()
        );

        if a.len().is_multiple_of(2) {
            Self::sqr_balanced::<false>(dst, a, scratch);
        } else {
            Self::sqr_balanced::<true>(dst, a, scratch);
        }
    }

    /// Exact 48-limb difference-form Karatsuba with three 24-limb leaves.
    fn mul_balanced_48(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        const SPLIT: usize = 24;
        const PRODUCT_LIMBS: usize = 2 * SPLIT;
        const MIDDLE_LIMBS: usize = PRODUCT_LIMBS + 1;

        // SAFETY: the parent selects two 48-limb inputs. Two 24-limb halves
        // produce disjoint 48-limb endpoints inside the complete 96-limb output.
        let ((a0, a1), (b0, b1), (low_product, high_product)) = unsafe {
            let (low, after_low) = dst.split_at_mut_unchecked(PRODUCT_LIMBS);
            let (high, _) = after_low.split_at_mut_unchecked(PRODUCT_LIMBS);
            (
                a.split_at_unchecked(SPLIT),
                b.split_at_unchecked(SPLIT),
                (low, high),
            )
        };
        {
            // SAFETY: the 146-limb root workspace includes the 49 limbs the
            // sequential 24-by-24 endpoint leaves each require.
            let (leaf_scratch, _) =
                unsafe { scratch.split_at_mut_unchecked(Self::BALANCED_24_SCRATCH_LIMBS) };
            Self::mul_balanced_basecase::<12>(low_product, a0, b0, leaf_scratch);
            Self::mul_balanced_basecase::<12>(high_product, a1, b1, leaf_scratch);
        }
        // SAFETY: the two 24-by-24 writers initialized both 48-limb endpoints.
        let (low_value, high_value) = unsafe {
            (
                LimbOutput::assume_init_mut(low_product),
                LimbOutput::assume_init_mut(high_product),
            )
        };

        // SAFETY: the validated 146-limb layout is 24+24+49+49 limbs:
        // two differences, one guarded middle product, and its leaf workspace.
        let (a_difference, b_difference, middle_product, leaf_scratch) = unsafe {
            let (a_diff, after_a) = scratch.split_at_mut_unchecked(SPLIT);
            let (b_diff, after_b) = after_a.split_at_mut_unchecked(SPLIT);
            let (middle, tail) = after_b.split_at_mut_unchecked(MIDDLE_LIMBS);
            (a_diff, b_diff, middle, tail)
        };
        let a_negative = Self::abs_difference_equal_width(a_difference, a0, a1);
        let b_negative = Self::abs_difference_equal_width(b_difference, b0, b1);
        // SAFETY: the 49-limb middle retains one guard above the 48-limb product.
        let (middle_value, middle_guard) =
            unsafe { middle_product.split_at_mut_unchecked(PRODUCT_LIMBS) };
        Self::mul_balanced_basecase::<12>(middle_value, a_difference, b_difference, leaf_scratch);
        if a_negative == b_negative {
            // Reverse subtraction initializes the guard from the borrow.
            Self::reverse_subtract_product_from_middle(middle_product, low_value);
        } else {
            // Addition needs a zero guard above the exact 48-limb product.
            // SAFETY: the 48-limb layout reserves 49 limbs for the middle.
            unsafe {
                *middle_guard.get_unchecked_mut(0) = 0;
            }
            Self::add_product_to_middle(middle_product, low_value);
        }
        Self::add_product_to_middle(middle_product, high_value);
        let middle_len = SharedEval::active_len(middle_product);
        // SAFETY: active_len returns a prefix length inside middle_product.
        let (active_middle, _) = unsafe { middle_product.split_at_unchecked(middle_len) };
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "48-limb Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: the normalized middle value is the exact 24-limb-shifted
        // cross coefficient of this complete 48-by-48 product.
        let _ = unsafe {
            let initialized = LimbOutput::assume_init_mut(dst.get_unchecked_mut(..96));
            SharedEval::fused_add_shifted_in_place(initialized, active_middle, SPLIT)
        };
    }

    /// One-level balanced difference-form Karatsuba with basecase subproducts.
    ///
    /// For `a = a0 + a1*B^S` and `b = b0 + b1*B^S`, let
    /// `d = (a0-a1)(b0-b1)`. The middle coefficient is `z0 + z2 - d`.
    /// Absolute differences keep all three products exactly `S` limbs wide,
    /// avoiding the guard-limb multiplication required by sum-form Karatsuba.
    fn mul_balanced_basecase<const SPLIT: usize>(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        // SAFETY: this private specialization is instantiated only with
        // SPLIT = 10 or 12; both 2*SPLIT and 2*SPLIT+1 fit even usize16.
        let (product_limbs, middle_limbs) = unsafe {
            let product = SPLIT.unchecked_mul(2);
            (product, product.unchecked_add(1))
        };
        // SAFETY: the parent selects this specialization only for two
        // 2*SPLIT-limb inputs and a complete 4*SPLIT-limb destination.
        let ((a0, a1), (b0, b1), (low_product, high_product)) = unsafe {
            let (low, after_low) = dst.split_at_mut_unchecked(product_limbs);
            let (high, _) = after_low.split_at_mut_unchecked(product_limbs);
            (
                a.split_at_unchecked(SPLIT),
                b.split_at_unchecked(SPLIT),
                (low, high),
            )
        };
        Schoolbook::mul_equal_nonempty(low_product, a0, b0);
        Schoolbook::mul_equal_nonempty(high_product, a1, b1);
        // SAFETY: both basecases wrote their complete disjoint endpoint spans.
        let (low_value, high_value) = unsafe {
            (
                LimbOutput::assume_init_mut(low_product),
                LimbOutput::assume_init_mut(high_product),
            )
        };

        // SAFETY: the selected fixed layout reserves two SPLIT-limb
        // differences and a 2*SPLIT+1-limb middle product (41 or 49 limbs).
        let (a_difference, b_difference, middle_product) = unsafe {
            let (a_diff, after_a) = scratch.split_at_mut_unchecked(SPLIT);
            let (b_diff, after_b) = after_a.split_at_mut_unchecked(SPLIT);
            let (middle, _) = after_b.split_at_mut_unchecked(middle_limbs);
            (a_diff, b_diff, middle)
        };
        let a_negative = Self::abs_difference_equal_width(a_difference, a0, a1);
        let b_negative = Self::abs_difference_equal_width(b_difference, b0, b1);
        // SAFETY: middle_product retains one initialized guard above product_limbs.
        let (middle_value, middle_guard) =
            unsafe { middle_product.split_at_mut_unchecked(product_limbs) };
        Schoolbook::mul_fixed_equal_distinct::<SPLIT>(middle_value, a_difference, b_difference);
        if a_negative == b_negative {
            // Reverse subtraction initializes the guard from the borrow.
            Self::reverse_subtract_product_from_middle(middle_product, low_value);
        } else {
            // Addition needs a zero guard above the exact 2*S-limb product.
            // SAFETY: middle_limbs is product_limbs + 1, so the guard exists.
            unsafe {
                *middle_guard.get_unchecked_mut(0) = 0;
            }
            Self::add_product_to_middle(middle_product, low_value);
        }
        Self::add_product_to_middle(middle_product, high_value);
        let middle_len = SharedEval::active_len(middle_product);
        // SAFETY: active_len returns a prefix length inside middle_product.
        let (active_middle, _) = unsafe { middle_product.split_at_unchecked(middle_len) };
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "fixed Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: `active_middle` is the exact cross coefficient and therefore
        // fits after its `SPLIT`-limb radix shift in the complete product.
        let _ = unsafe {
            let initialized = LimbOutput::assume_init_mut(
                dst.get_unchecked_mut(..a.len().unchecked_add(b.len())),
            );
            SharedEval::fused_add_shifted_in_place(initialized, active_middle, SPLIT)
        };
    }

    /// One-level odd-width Karatsuba with three fixed basecase leaves.
    ///
    /// For width `2*S-1`, the low blocks and absolute differences have `S` limbs,
    /// while the high blocks have `S-1`. Zero extension is used only to form the
    /// exact `S`-limb differences; the high endpoint product retains its natural
    /// `2*(S-1)`-limb span.
    fn mul_balanced_basecase_odd<const SPLIT: usize>(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        // SAFETY: this private specialization is instantiated only with
        // SPLIT = 12; high_len = 11 and the largest derived width is 25.
        let (product_limbs, high_product_limbs, middle_limbs) = unsafe {
            let high = SPLIT.unchecked_sub(1);
            let product = SPLIT.unchecked_mul(2);
            (product, high.unchecked_mul(2), product.unchecked_add(1))
        };
        debug_assert_eq!(
            a.len(),
            23,
            "left operand does not match the odd Karatsuba width"
        );
        debug_assert_eq!(
            b.len(),
            23,
            "right operand does not match the odd Karatsuba width"
        );

        // SAFETY: the parent selects only the 23-by-23 shape, with SPLIT=12.
        // Its endpoints occupy 24+22=46 limbs in the complete destination.
        let ((a0, a1), (b0, b1), (low_product, high_product)) = unsafe {
            let (low, after_low) = dst.split_at_mut_unchecked(product_limbs);
            let (high, _) = after_low.split_at_mut_unchecked(high_product_limbs);
            (
                a.split_at_unchecked(SPLIT),
                b.split_at_unchecked(SPLIT),
                (low, high),
            )
        };
        Schoolbook::mul_equal_nonempty(low_product, a0, b0);
        Schoolbook::mul_equal_nonempty(high_product, a1, b1);
        // SAFETY: both basecases initialized the exact 24- and 22-limb endpoints.
        let (low_value, high_value) = unsafe {
            (
                LimbOutput::assume_init_mut(low_product),
                LimbOutput::assume_init_mut(high_product),
            )
        };

        // SAFETY: the validated odd layout reserves 12+12+25=49 limbs for
        // the disjoint differences and guarded middle product.
        let (a_difference, b_difference, middle_product) = unsafe {
            let (a_diff, after_a) = scratch.split_at_mut_unchecked(SPLIT);
            let (b_diff, after_b) = after_a.split_at_mut_unchecked(SPLIT);
            let (middle, _) = after_b.split_at_mut_unchecked(middle_limbs);
            (a_diff, b_diff, middle)
        };
        let a_negative = Self::abs_difference_zero_extended(a_difference, a0, a1);
        let b_negative = Self::abs_difference_zero_extended(b_difference, b0, b1);
        // SAFETY: middle_product retains one guard above product_limbs.
        let (middle_value, middle_guard) =
            unsafe { middle_product.split_at_mut_unchecked(product_limbs) };
        Schoolbook::mul_fixed_equal_distinct::<SPLIT>(middle_value, a_difference, b_difference);
        // The signed difference identity is identical to the equal-half case:
        // z1 = z0 + z2 - (a0-a1)(b0-b1). Only z2 has the shorter exact span.
        if a_negative == b_negative {
            // Reverse subtraction initializes the guard from the borrow.
            Self::reverse_subtract_product_from_middle(middle_product, low_value);
        } else {
            // Addition needs a zero guard above the exact 2*S-limb product.
            // SAFETY: the odd-width layout retains one guard above product_limbs.
            unsafe {
                *middle_guard.get_unchecked_mut(0) = 0;
            }
            Self::add_product_to_middle(middle_product, low_value);
        }
        Self::add_product_to_middle(middle_product, high_value);
        let middle_len = SharedEval::active_len(middle_product);
        // SAFETY: active_len returns a prefix length inside middle_product.
        let (active_middle, _) = unsafe { middle_product.split_at_unchecked(middle_len) };
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "odd Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: zero-extending the shorter high blocks does not change the
        // exact cross coefficient, whose radix shift fits the full product.
        let _ = unsafe {
            let initialized = LimbOutput::assume_init_mut(
                dst.get_unchecked_mut(..a.len().unchecked_add(b.len())),
            );
            SharedEval::fused_add_shifted_in_place(initialized, active_middle, SPLIT)
        };
    }
}
