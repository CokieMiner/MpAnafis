//! Generic balanced difference-form Karatsuba decomposition.

use core::cmp::Ordering;

use super::{
    ArchKernels, KARATSUBA_THRESHOLD, Karatsuba, Limb, SQR_KARATSUBA_THRESHOLD, Schoolbook,
    SharedEval,
};

impl Karatsuba {
    /// Multiply equal-width operands with recursively multiplied differences.
    ///
    /// For `a = a0 + a1*B^S` and `b = b0 + b1*B^S`, the middle coefficient is
    /// `a0*b0 + a1*b1 - (a0-a1)(b0-b1)`. Unlike sum-form Karatsuba, the third
    /// product never grows to `S+1` limbs. Odd-width operands zero-extend only the
    /// high blocks used by the absolute differences; their endpoint product keeps
    /// its exact shorter width.
    #[allow(
        unsafe_code,
        reason = "The Karatsuba reconstruction identity proves the active middle coefficient fits at its radix offset"
    )]
    pub fn balanced_difference<const ZERO_EXTENDED_HIGH: bool>(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        debug_assert_eq!(a.len(), b.len(), "balanced operands must have equal widths");
        let split = if ZERO_EXTENDED_HIGH {
            Self::balanced_split_len(a.len())
        } else {
            a.len() >> 1
        };
        let high_len = a.len().wrapping_sub(split);
        let product_limbs = split.wrapping_mul(2);
        let middle_limbs = product_limbs.wrapping_add(1);
        let (a0, a1) = a.split_at(split);
        let (b0, b1) = b.split_at(split);
        let (low_product, after_low_product) = dst.split_at_mut(product_limbs);
        let high_product_limbs = if ZERO_EXTENDED_HIGH {
            high_len.wrapping_mul(2)
        } else {
            product_limbs
        };
        let (high_product, _) = after_low_product.split_at_mut(high_product_limbs);

        // The three subproducts are sequential. Low and high may therefore borrow
        // all scratch before the difference operands occupy its prefix. Below the
        // crossover, call the basecase directly and skip three dispatcher frames.
        let subproducts_are_basecase = split < KARATSUBA_THRESHOLD;
        if subproducts_are_basecase {
            Schoolbook::mul(low_product, a0, b0);
            Schoolbook::mul(high_product, a1, b1);
        } else {
            Self::mul(low_product, a0, b0, scratch);
            Self::mul(high_product, a1, b1, scratch);
        }

        let (a_difference, scratch_after_a) = scratch.split_at_mut(split);
        let (b_difference, scratch_after_b) = scratch_after_a.split_at_mut(split);
        let (middle_product, recursive_scratch) = scratch_after_b.split_at_mut(middle_limbs);
        let (a_negative, b_negative) = if ZERO_EXTENDED_HIGH {
            (
                Self::abs_difference_zero_extended(a_difference, a0, a1),
                Self::abs_difference_zero_extended(b_difference, b0, b1),
            )
        } else {
            (
                Self::abs_difference_equal_width(a_difference, a0, a1),
                Self::abs_difference_equal_width(b_difference, b0, b1),
            )
        };
        let (middle_value, middle_guard) = middle_product.split_at_mut(product_limbs);
        if subproducts_are_basecase {
            Schoolbook::mul(middle_value, a_difference, b_difference);
        } else {
            Self::mul(middle_value, a_difference, b_difference, recursive_scratch);
        }
        // The recursive product writes 2*S limbs; reconstruction needs one guard
        // for the fixed-width modular sum of the three coefficients.
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
            split <= dst.len() && active_middle.len() <= dst.len().saturating_sub(split),
            "balanced Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: `active_middle` is the normalized cross coefficient
        // `a0*b1 + a1*b0`. After multiplication by `B^split` it is a term of
        // the exact product held by `dst`; the earlier endpoint partitions also
        // establish that `dst` contains the complete product width.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, split) };
    }
}

impl Karatsuba {
    /// Square an equal-width operand with a recursively squared absolute difference.
    ///
    /// For `a = a0 + a1*B^S`, let `d = (a0-a1)^2`. Then the middle
    /// coefficient is `a0^2 + a1^2 - d = 2*a0*a1`. The difference is exactly
    /// `S` limbs, whereas sum-form Karatsuba may grow to `S+1`; this removes a
    /// guard limb from the third recursive square and its complete subtree.
    #[allow(
        unsafe_code,
        reason = "The Karatsuba square identity proves the active middle coefficient fits at its radix offset"
    )]
    pub fn balanced_square<const ZERO_EXTENDED_HIGH: bool>(
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [Limb],
    ) {
        let split = a.len().div_ceil(2);
        let high_len = a.len().wrapping_sub(split);
        let product_limbs = split.wrapping_mul(2);
        let middle_limbs = product_limbs.wrapping_add(1);
        let (a0, a1) = a.split_at(split);
        let (low_product, after_low_product) = dst.split_at_mut(product_limbs);
        let high_product_limbs = high_len.wrapping_mul(2);
        let (high_product, _) = after_low_product.split_at_mut(high_product_limbs);

        // Endpoint squares run before the local difference occupies scratch, so
        // both may borrow the complete buffer. Below the square crossover, direct
        // basecase calls avoid recursive dispatcher frames.
        let subproducts_are_basecase = split < SQR_KARATSUBA_THRESHOLD;
        if subproducts_are_basecase {
            Schoolbook::sqr(low_product, a0);
            Schoolbook::sqr(high_product, a1);
        } else {
            Self::sqr(low_product, a0, scratch);
            Self::sqr(high_product, a1, scratch);
        }

        let (difference, after_difference) = scratch.split_at_mut(split);
        if ZERO_EXTENDED_HIGH {
            let _ = Self::abs_difference_zero_extended(difference, a0, a1);
        } else {
            let _ = Self::abs_difference_equal_width(difference, a0, a1);
        }
        let (middle_product, recursive_scratch) = after_difference.split_at_mut(middle_limbs);
        let (middle_value, middle_guard) = middle_product.split_at_mut(product_limbs);
        if subproducts_are_basecase {
            Schoolbook::sqr(middle_value, difference);
        } else {
            Self::sqr(middle_value, difference, recursive_scratch);
        }
        // The difference square writes exactly 2*S limbs. Reconstruction needs
        // one sign-extension guard for the potentially negative intermediate z0-d.
        if let Some(guard) = middle_guard.first_mut() {
            *guard = 0;
        }

        Self::reverse_subtract_product_from_middle(middle_product, low_product);
        Self::add_product_to_middle(middle_product, high_product);
        let middle_len = SharedEval::active_len(middle_product);
        let (active_middle, _) = middle_product.split_at(middle_len);
        debug_assert!(
            split <= dst.len() && active_middle.len() <= dst.len().saturating_sub(split),
            "balanced Karatsuba square middle coefficient exceeds the destination"
        );
        // SAFETY: `active_middle = 2*a0*a1` is the exact cross coefficient of
        // the square. Its shift by `split` therefore lies wholly within the
        // complete square destination established by the endpoint partitions.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, split) };
    }
}

impl Karatsuba {
    /// Choose the low-block width for balanced difference-form recursion.
    ///
    /// Even operands use equal halves. Odd operands place the extra limb in the
    /// low block and zero-extend the shorter high block for the difference product.
    /// Once children recurse, a half one limb below a power of two is rounded up:
    /// that exposes the balanced power-of-two specialization without padding a
    /// basecase leaf, where the extra work cannot be recovered recursively.
    pub const fn balanced_split_len(len: usize) -> usize {
        let half = len.div_ceil(2);
        let rounded_half = half.wrapping_add(1);
        if half < KARATSUBA_THRESHOLD || half.is_multiple_of(2) || !rounded_half.is_power_of_two() {
            return half;
        }
        rounded_half
    }
}

impl Karatsuba {
    pub fn abs_difference_zero_extended(dst: &mut [Limb], low: &[Limb], high: &[Limb]) -> bool {
        debug_assert_eq!(
            dst.len(),
            low.len(),
            "difference must use the low-block width"
        );
        debug_assert!(high.len() <= low.len(), "high block exceeds the low block");
        if high.is_empty() {
            dst.copy_from_slice(low);
            return false;
        }
        let (shared_low, extension) = low.split_at(high.len());
        let ordering = if extension.iter().any(|limb| *limb != 0) {
            Ordering::Greater
        } else {
            shared_low.iter().rev().cmp(high.iter().rev())
        };

        if ordering == Ordering::Less {
            let (prefix, suffix) = dst.split_at_mut(high.len());
            // SAFETY: `prefix`, `high`, and the low shared prefix all have exactly
            // `high.len()` limbs and occupy disjoint buffers. The ordering proof
            // establishes high >= low, so no borrow escapes this shared width.
            let borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    prefix.as_mut_ptr(),
                    high.as_ptr(),
                    low.as_ptr(),
                    high.len(),
                )
            };
            debug_assert_eq!(borrow, 0, "absolute difference underflowed");
            suffix.fill(0);
            true
        } else {
            let (prefix, destination_extension) = dst.split_at_mut(high.len());
            let (low_prefix, low_extension) = low.split_at(high.len());
            // SAFETY: the three shared prefixes have the same width and are
            // disjoint. A borrow may leave this prefix, but the ordering proof
            // guarantees the copied low extension absorbs it.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    prefix.as_mut_ptr(),
                    low_prefix.as_ptr(),
                    high.as_ptr(),
                    high.len(),
                )
            };
            for (dst_limb, low_limb) in destination_extension.iter_mut().zip(low_extension) {
                let (difference, underflow) = low_limb.overflowing_sub(borrow);
                *dst_limb = difference;
                borrow = Limb::from(underflow);
            }
            debug_assert_eq!(borrow, 0, "absolute difference underflowed");
            false
        }
    }
}
