//! The Karatsuba driver: general splits and the fixed-width specializations.
use core::cmp::{max, min};

use super::{
    KARATSUBA_THRESHOLD, Limb, Multiplication, SQR_KARATSUBA_THRESHOLD, Schoolbook, SharedEval,
};

/// Namespace for the Karatsuba multiplication and squaring tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Karatsuba;

impl Karatsuba {
    /// Scratch limbs required by the exact balanced 20-limb specialization.
    pub const BALANCED_20_SCRATCH_LIMBS: usize = 41;

    /// Scratch limbs required by the exact balanced 24-limb specialization.
    pub const BALANCED_24_SCRATCH_LIMBS: usize = 49;

    /// Scratch limbs required by the exact balanced 32-limb specialization.
    pub const BALANCED_32_SCRATCH_LIMBS: usize = 65;

    /// Scratch limbs required by the exact balanced 48-limb specialization.
    pub const BALANCED_48_SCRATCH_LIMBS: usize = 146;

    /// Computes the product of two limb slices using Karatsuba multiplication.
    ///
    /// `dst` must have at least `a.len() + b.len()` elements. `scratch` must have
    /// at least [`Multiplication::karatsuba_mul_scratch_len`] elements for the operand lengths.
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
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
        Self::mul_forced(dst, a, b, scratch);
    }

    /// Executes one Karatsuba level regardless of the configured crossover.
    ///
    /// Recursive subproducts still use normal tier dispatch, so this isolates the
    /// cost of selecting Karatsuba at the root. A shape that cannot produce two
    /// nonempty halves remains a rectangular schoolbook multiplication.
    #[allow(
        unsafe_code,
        reason = "The Karatsuba identity bounds the shifted active middle coefficient by the exact product width"
    )]
    pub fn mul_forced(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.is_empty() || b.is_empty() {
            return;
        }
        if a.len() < 2 || b.len() < 2 {
            Schoolbook::mul(dst, a, b);
            return;
        }
        let smaller_len = min(a.len(), b.len());
        let larger_len = max(a.len(), b.len());
        if smaller_len <= larger_len.div_ceil(2) {
            // Splitting at ceil(larger/2) would leave the smaller operand's high
            // half empty. The nominal three-product identity would then retain a
            // stale high destination range and perform no useful reduction on the
            // larger operand, so this shape remains a single rectangular basecase.
            Schoolbook::mul(dst, a, b);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::karatsuba_mul_forced_scratch_len(a.len(), b.len()),
            "Karatsuba multiplication scratch is undersized: have {}, need {} for {}x{} limbs",
            scratch.len(),
            Multiplication::karatsuba_mul_forced_scratch_len(a.len(), b.len()),
            a.len(),
            b.len()
        );
        if a.len() == 20 && b.len() == 20 {
            Self::karatsuba_balanced_basecase_difference::<10>(dst, a, b, scratch);
            return;
        }
        if a.len() == 23 && b.len() == 23 {
            Self::karatsuba_balanced_basecase_difference_odd::<12>(dst, a, b, scratch);
            return;
        }
        if a.len() == 24 && b.len() == 24 {
            Self::karatsuba_balanced_basecase_difference::<12>(dst, a, b, scratch);
            return;
        }
        if a.len() == 32 && b.len() == 32 {
            Self::karatsuba_balanced_32_difference(dst, a, b, scratch);
            return;
        }
        if a.len() == 48 && b.len() == 48 {
            Self::karatsuba_balanced_48_difference(dst, a, b, scratch);
            return;
        }
        if a.len() == b.len() {
            if Self::balanced_split_len(a.len()) != a.len().div_ceil(2) {
                Self::balanced_difference::<true>(dst, a, b, scratch);
                return;
            }
            if a.len().is_multiple_of(2) {
                Self::balanced_difference::<false>(dst, a, b, scratch);
                return;
            }
            Self::balanced_difference::<true>(dst, a, b, scratch);
            return;
        }
        let split_len = larger_len.div_ceil(2);
        let (a0, a1) = if a.len() > split_len {
            a.split_at(split_len)
        } else {
            (a, [].as_slice())
        };
        let (b0, b1) = if b.len() > split_len {
            b.split_at(split_len)
        } else {
            (b, [].as_slice())
        };

        let sum_space = split_len.wrapping_add(1);
        let (a_sum_buffer, after_a_sum) = scratch.split_at_mut(sum_space);
        let (b_sum_buffer, recursive_scratch) = after_a_sum.split_at_mut(sum_space);
        let a_sum_len = Self::add_slices_in_place(a_sum_buffer, a0, a1);
        let b_sum_len = Self::add_slices_in_place(b_sum_buffer, b0, b1);
        let (a_sum, _) = a_sum_buffer.split_at(a_sum_len);
        let (b_sum, _) = b_sum_buffer.split_at(b_sum_len);

        let low_product_len = split_len.wrapping_mul(2);
        let (low_product, high_product) = dst.split_at_mut(low_product_len);
        Self::mul(low_product, a0, b0, recursive_scratch);
        Self::mul(high_product, a1, b1, recursive_scratch);

        let middle_space = sum_space.wrapping_mul(2);
        let (middle_product, next_scratch) = recursive_scratch.split_at_mut(middle_space);
        let middle_product_len = a_sum_len.wrapping_add(b_sum_len);
        let (middle_value, middle_guard) = middle_product.split_at_mut(middle_product_len);
        // Each sum has either split_len or split_len+1 limbs, so at most two
        // reserved guard limbs lie above the exact recursive product. Zero only
        // those guards; the child multiplication overwrites every value limb.
        middle_guard.fill(0);
        Self::mul(middle_value, a_sum, b_sum, next_scratch);

        let mut middle_len = SharedEval::active_len(middle_product);
        let low_len = SharedEval::active_len(low_product);
        let high_len = SharedEval::active_len(high_product);
        let (active_low, _) = low_product.split_at(low_len);
        let (active_high, _) = high_product.split_at(high_len);

        // (a0+a1)(b0+b1)-a0b0-a1b1 = a0b1+a1b0 is nonnegative.
        Self::sub_slices_in_place(middle_product, &mut middle_len, active_low);
        Self::sub_slices_in_place(middle_product, &mut middle_len, active_high);
        let (active_middle, _) = middle_product.split_at(middle_len);
        debug_assert!(
            split_len <= dst.len() && active_middle.len() <= dst.len().saturating_sub(split_len),
            "Karatsuba middle exceeds destination: {} limbs at shift {} for {}x{} into {} limbs",
            active_middle.len(),
            split_len,
            a.len(),
            b.len(),
            dst.len()
        );
        // SAFETY: subtraction above leaves `a0*b1 + a1*b0`, the exact cross
        // coefficient. Its `split_len`-limb radix shift is therefore contained
        // by the complete product destination required by this tier's contract.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, split_len) };
    }

    /// Computes the square of a limb slice using Karatsuba decomposition.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.is_empty() {
            return;
        }
        if a.len() < SQR_KARATSUBA_THRESHOLD || a.len() < 2 {
            Schoolbook::sqr(dst, a);
            return;
        }
        Self::sqr_forced(dst, a, scratch);
    }

    /// Executes one Karatsuba square level regardless of the configured crossover.
    ///
    /// Recursive squares retain normal dispatch, so only the root decision is
    /// forced. Inputs shorter than two limbs use the schoolbook square because no
    /// nontrivial two-way split exists. The active destination is overwritten
    /// completely by the endpoint squares and middle reconstruction.
    pub fn sqr_forced(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.is_empty() {
            return;
        }
        if a.len() < 2 {
            Schoolbook::sqr(dst, a);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::karatsuba_sqr_forced_scratch_len(a.len()),
            "Karatsuba squaring scratch is undersized: have {}, need {} for {} limbs",
            scratch.len(),
            Multiplication::karatsuba_sqr_forced_scratch_len(a.len()),
            a.len()
        );

        if a.len().is_multiple_of(2) {
            Self::balanced_square::<false>(dst, a, scratch);
        } else {
            Self::balanced_square::<true>(dst, a, scratch);
        }
    }

    /// One-level balanced difference-form Karatsuba with basecase subproducts.
    ///
    /// For `a = a0 + a1*B^S` and `b = b0 + b1*B^S`, let
    /// `d = (a0-a1)(b0-b1)`. The middle coefficient is `z0 + z2 - d`.
    /// Absolute differences keep all three products exactly `S` limbs wide,
    /// avoiding the guard-limb multiplication required by sum-form Karatsuba.
    #[allow(
        unsafe_code,
        reason = "The fixed Karatsuba reconstruction bounds its shifted cross coefficient by the destination"
    )]
    fn karatsuba_balanced_basecase_difference<const SPLIT: usize>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        let product_limbs = SPLIT.wrapping_mul(2);
        let middle_limbs = product_limbs.wrapping_add(1);
        let (a0, a1) = a.split_at(SPLIT);
        let (b0, b1) = b.split_at(SPLIT);
        let (low_product, after_low_product) = dst.split_at_mut(product_limbs);
        let (high_product, _) = after_low_product.split_at_mut(product_limbs);
        Schoolbook::mul(low_product, a0, b0);
        Schoolbook::mul(high_product, a1, b1);

        let (a_difference, scratch_after_a) = scratch.split_at_mut(SPLIT);
        let (b_difference, middle_storage) = scratch_after_a.split_at_mut(SPLIT);
        let (middle_product, _) = middle_storage.split_at_mut(middle_limbs);
        let a_negative = Self::abs_difference_equal_width(a_difference, a0, a1);
        let b_negative = Self::abs_difference_equal_width(b_difference, b0, b1);
        let (middle_value, middle_guard) = middle_product.split_at_mut(product_limbs);
        Schoolbook::mul(middle_value, a_difference, b_difference);
        // The basecase writes exactly 2*S limbs. The guard is required because
        // z0+z2-d may transiently use one additional limb modulo B^(2*S+1).
        if let Some(guard) = middle_guard.first_mut() {
            *guard = 0;
        }

        if a_negative == b_negative {
            Self::reverse_subtract_product_from_middle(middle_product, low_product);
        } else {
            Self::add_product_to_middle(middle_product, low_product);
        }
        Self::add_product_to_middle(middle_product, high_product);
        let middle_len = SharedEval::active_len(middle_product);
        let (active_middle, _) = middle_product.split_at(middle_len);
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "fixed Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: `active_middle` is the exact cross coefficient and therefore
        // fits after its `SPLIT`-limb radix shift in the complete product.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, SPLIT) };
    }

    /// One-level odd-width Karatsuba with three fixed basecase leaves.
    ///
    /// For width `2*S-1`, the low blocks and absolute differences have `S` limbs,
    /// while the high blocks have `S-1`. Zero extension is used only to form the
    /// exact `S`-limb differences; the high endpoint product retains its natural
    /// `2*(S-1)`-limb span.
    #[allow(
        unsafe_code,
        reason = "The odd-width Karatsuba reconstruction bounds its shifted cross coefficient by the destination"
    )]
    fn karatsuba_balanced_basecase_difference_odd<const SPLIT: usize>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        let high_len = SPLIT.wrapping_sub(1);
        let product_limbs = SPLIT.wrapping_mul(2);
        let high_product_limbs = high_len.wrapping_mul(2);
        let middle_limbs = product_limbs.wrapping_add(1);
        debug_assert_eq!(
            a.len(),
            SPLIT.wrapping_add(high_len),
            "left operand does not match the odd Karatsuba width"
        );
        debug_assert_eq!(
            b.len(),
            SPLIT.wrapping_add(high_len),
            "right operand does not match the odd Karatsuba width"
        );

        let (a0, a1) = a.split_at(SPLIT);
        let (b0, b1) = b.split_at(SPLIT);
        let (low_product, after_low_product) = dst.split_at_mut(product_limbs);
        let (high_product, _) = after_low_product.split_at_mut(high_product_limbs);
        Schoolbook::mul(low_product, a0, b0);
        Schoolbook::mul(high_product, a1, b1);

        let (a_difference, scratch_after_a) = scratch.split_at_mut(SPLIT);
        let (b_difference, middle_storage) = scratch_after_a.split_at_mut(SPLIT);
        let (middle_product, _) = middle_storage.split_at_mut(middle_limbs);
        let a_negative = Self::abs_difference_zero_extended(a_difference, a0, a1);
        let b_negative = Self::abs_difference_zero_extended(b_difference, b0, b1);
        let (middle_value, middle_guard) = middle_product.split_at_mut(product_limbs);
        Schoolbook::mul(middle_value, a_difference, b_difference);
        if let Some(guard) = middle_guard.first_mut() {
            *guard = 0;
        }

        // The signed difference identity is identical to the equal-half case:
        // z1 = z0 + z2 - (a0-a1)(b0-b1). Only z2 has the shorter exact span.
        if a_negative == b_negative {
            Self::reverse_subtract_product_from_middle(middle_product, low_product);
        } else {
            Self::add_product_to_middle(middle_product, low_product);
        }
        Self::add_product_to_middle(middle_product, high_product);
        let middle_len = SharedEval::active_len(middle_product);
        let (active_middle, _) = middle_product.split_at(middle_len);
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "odd Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: zero-extending the shorter high blocks does not change the
        // exact cross coefficient, whose radix shift fits the full product.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, SPLIT) };
    }

    /// Exact 32-limb Karatsuba using fixed-width absolute differences.
    ///
    /// With `z0 = a0*b0`, `z2 = a1*b1`, and
    /// `d = (a0-a1)(b0-b1)`, the middle coefficient is
    /// `z0 + z2 - d`.  Absolute differences keep the third multiplication at
    /// exactly 16 limbs; the sign determines whether its magnitude is added or
    /// subtracted.  Arithmetic in the 33-limb middle buffer is modulo `B^33`,
    /// which is exact because the cross product is strictly below `B^33`.
    #[allow(
        unsafe_code,
        reason = "The exact 32-limb Karatsuba reconstruction bounds its shifted cross coefficient"
    )]
    fn karatsuba_balanced_32_difference(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        const SPLIT: usize = 16;
        const PRODUCT_LIMBS: usize = SPLIT.wrapping_mul(2);
        const MIDDLE_LIMBS: usize = PRODUCT_LIMBS.wrapping_add(1);

        let (a0, a1) = a.split_at(SPLIT);
        let (b0, b1) = b.split_at(SPLIT);
        let (low_product, after_low_product) = dst.split_at_mut(PRODUCT_LIMBS);
        let (high_product, _) = after_low_product.split_at_mut(PRODUCT_LIMBS);
        Schoolbook::mul_fixed_equal_distinct::<16>(low_product, a0, b0);
        Schoolbook::mul_fixed_equal_distinct::<16>(high_product, a1, b1);

        let (a_difference, after_a_difference) = scratch.split_at_mut(SPLIT);
        let (b_difference, middle_storage) = after_a_difference.split_at_mut(SPLIT);
        let a_negative = Self::abs_difference_equal_width(a_difference, a0, a1);
        let b_negative = Self::abs_difference_equal_width(b_difference, b0, b1);
        let (middle_product, _) = middle_storage.split_at_mut(MIDDLE_LIMBS);
        let (middle_value, middle_guard) = middle_product.split_at_mut(PRODUCT_LIMBS);
        Schoolbook::mul_fixed_equal_distinct::<16>(middle_value, a_difference, b_difference);
        if let Some(guard) = middle_guard.first_mut() {
            *guard = 0;
        }

        if a_negative == b_negative {
            Self::reverse_subtract_product_from_middle(middle_product, low_product);
        } else {
            Self::add_product_to_middle(middle_product, low_product);
        }
        Self::add_product_to_middle(middle_product, high_product);

        let middle_len = SharedEval::active_len(middle_product);
        let (active_middle, _) = middle_product.split_at(middle_len);
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "32-limb Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: the normalized middle value is the exact 16-limb-shifted
        // cross coefficient of this complete 32-by-32 product.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, SPLIT) };
    }

    /// Exact 48-limb difference-form Karatsuba with three 24-limb leaves.
    #[allow(
        unsafe_code,
        reason = "The exact 48-limb Karatsuba reconstruction bounds its shifted cross coefficient"
    )]
    fn karatsuba_balanced_48_difference(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        const SPLIT: usize = 24;
        const PRODUCT_LIMBS: usize = SPLIT.wrapping_mul(2);
        const MIDDLE_LIMBS: usize = PRODUCT_LIMBS.wrapping_add(1);

        let (a0, a1) = a.split_at(SPLIT);
        let (b0, b1) = b.split_at(SPLIT);
        let (low_product, after_low_product) = dst.split_at_mut(PRODUCT_LIMBS);
        let (high_product, _) = after_low_product.split_at_mut(PRODUCT_LIMBS);
        {
            let (leaf_scratch, _) = scratch.split_at_mut(Self::BALANCED_24_SCRATCH_LIMBS);
            Self::karatsuba_balanced_basecase_difference::<12>(low_product, a0, b0, leaf_scratch);
            Self::karatsuba_balanced_basecase_difference::<12>(high_product, a1, b1, leaf_scratch);
        }

        let (a_difference, scratch_after_a) = scratch.split_at_mut(SPLIT);
        let (b_difference, scratch_after_b) = scratch_after_a.split_at_mut(SPLIT);
        let (middle_product, leaf_scratch) = scratch_after_b.split_at_mut(MIDDLE_LIMBS);
        let a_negative = Self::abs_difference_equal_width(a_difference, a0, a1);
        let b_negative = Self::abs_difference_equal_width(b_difference, b0, b1);
        let (middle_value, middle_guard) = middle_product.split_at_mut(PRODUCT_LIMBS);
        Self::karatsuba_balanced_basecase_difference::<12>(
            middle_value,
            a_difference,
            b_difference,
            leaf_scratch,
        );
        // The exact leaf writes 48 limbs; the difference coefficient needs one
        // guard limb for the sum of the two half products.
        if let Some(guard) = middle_guard.first_mut() {
            *guard = 0;
        }

        if a_negative == b_negative {
            Self::reverse_subtract_product_from_middle(middle_product, low_product);
        } else {
            Self::add_product_to_middle(middle_product, low_product);
        }
        Self::add_product_to_middle(middle_product, high_product);
        let middle_len = SharedEval::active_len(middle_product);
        let (active_middle, _) = middle_product.split_at(middle_len);
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "48-limb Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: the normalized middle value is the exact 24-limb-shifted
        // cross coefficient of this complete 48-by-48 product.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, SPLIT) };
    }
}
