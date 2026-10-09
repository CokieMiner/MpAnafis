//! Generic balanced difference-form Karatsuba decomposition.
//!
//! # References
//!
//! - Karatsuba, A., & Ofman, Y. (1962). Multiplication of Many-Digital
//!   Numbers by Automata. *Proceedings of the USSR Academy of Sciences*,
//!   145(2), 293–294.

#![expect(
    unsafe_code,
    reason = "The balanced split bounds equal-width differences, disjoint endpoint products, and the retained middle guard"
)]

use core::cmp::Ordering;

use super::{
    ArchKernels, KARATSUBA_THRESHOLD, Karatsuba, Limb, LimbOutput, SQR_KARATSUBA_THRESHOLD,
    Schoolbook, SharedEval,
};

impl Karatsuba {
    /// Multiply equal-width operands with recursively multiplied differences.
    ///
    /// For `a = a0 + a1*B^S` and `b = b0 + b1*B^S`, the middle coefficient is
    /// `a0*b0 + a1*b1 - (a0-a1)(b0-b1)`. Unlike sum-form Karatsuba, the third
    /// product never grows to `S+1` limbs. Odd-width operands zero-extend only the
    /// high blocks used by the absolute differences; their endpoint product keeps
    /// its exact shorter width.
    pub fn mul_balanced<const ZERO_EXTENDED_HIGH: bool>(
        dst: &mut [impl LimbOutput],
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
        // SAFETY: this tier retains 0 < split < a.len(). A valid limb slice
        // has at most isize::MAX/size_of::<Limb>() elements with Limb >= 2
        // bytes, so 2*split+1 <= 2*a.len()-1 < usize::MAX on every target.
        let (product_limbs, middle_limbs) = unsafe {
            let product = split.unchecked_mul(2);
            (product, product.unchecked_add(1))
        };
        // SAFETY: the direct tier admits equal widths of at least two limbs.
        // balanced_split_len retains two nonempty blocks; the even split is
        // exactly half the width. Both split positions are inside the inputs.
        let ((a0, a1), (b0, b1)) =
            unsafe { (a.split_at_unchecked(split), b.split_at_unchecked(split)) };
        let high_product_limbs = if ZERO_EXTENDED_HIGH {
            // SAFETY: a1 is a valid initialized limb slice, whose byte bound
            // proves twice its limb count fits on 16/32/64-bit targets.
            unsafe { a1.len().unchecked_mul(2) }
        } else {
            product_limbs
        };
        // SAFETY: the caller supplies the complete 2*a.len()-limb product.
        // The two endpoint widths sum to that length, and splitting preserves
        // disjoint mutable borrows of the initialized destination.
        let (low_product, high_product) = unsafe {
            let (low, after_low) = dst.split_at_mut_unchecked(product_limbs);
            let (high, _) = after_low.split_at_mut_unchecked(high_product_limbs);
            (low, high)
        };

        // The three subproducts are sequential. Low and high may therefore borrow
        // all scratch before the difference operands occupy its prefix. Below the
        // crossover, call the basecase directly and skip three dispatcher frames.
        let subproducts_are_basecase = split < KARATSUBA_THRESHOLD;
        if subproducts_are_basecase {
            Schoolbook::mul_equal_nonempty(low_product, a0, b0);
            Schoolbook::mul_equal_nonempty(high_product, a1, b1);
        } else {
            // The full low block clears the crossover. Equal high blocks have
            // the same width; only a zero-extended high block can fall below it.
            Self::mul(low_product, a0, b0, scratch);
            if ZERO_EXTENDED_HIGH {
                Self::dispatch_mul(high_product, a1, b1, scratch);
            } else {
                Self::mul(high_product, a1, b1, scratch);
            }
        }
        // SAFETY: endpoint writers initialized their entire disjoint spans;
        // together they cover the exact 2*a.len()-limb destination prefix.
        let (low_value, high_value) = unsafe {
            (
                LimbOutput::assume_init_mut(low_product),
                LimbOutput::assume_init_mut(high_product),
            )
        };

        // SAFETY: karatsuba_mul_scratch_len reserves two split-limb differences,
        // a 2*split+1-limb middle product, and the recursive tail for this exact
        // split. Consecutive partitions retain disjoint initialized spans.
        let (a_difference, b_difference, middle_product, recursive_scratch) = unsafe {
            let (a_diff, after_a) = scratch.split_at_mut_unchecked(split);
            let (b_diff, after_b) = after_a.split_at_mut_unchecked(split);
            let (middle, tail) = after_b.split_at_mut_unchecked(middle_limbs);
            (a_diff, b_diff, middle, tail)
        };
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
        // SAFETY: middle_product has product_limbs + 1 initialized limbs.
        let (middle_value, middle_guard) =
            unsafe { middle_product.split_at_mut_unchecked(product_limbs) };
        if subproducts_are_basecase {
            // The two nonempty differences occupy disjoint scratch partitions,
            // so the same-slice square and empty-operand cases cannot apply.
            Schoolbook::mul_equal_nonempty_distinct(middle_value, a_difference, b_difference);
        } else {
            Self::mul(middle_value, a_difference, b_difference, recursive_scratch);
        }
        if a_negative == b_negative {
            // Reverse subtraction writes the guard from its escaping borrow;
            // the guard left by the scratch buffer is never read in this branch.
            Self::reverse_subtract_product_from_middle(middle_product, low_value);
        } else {
            // Addition reads the guard above the exact 2*S-limb difference
            // product, so this branch must extend that product with zero.
            // SAFETY: middle_limbs is product_limbs + 1, retaining one guard.
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
            split <= dst.len() && active_middle.len() <= dst.len().saturating_sub(split),
            "balanced Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: `active_middle` is the normalized cross coefficient
        // `a0*b1 + a1*b0`. After multiplication by `B^split` it is a term of
        // the exact product held by `dst`; the earlier endpoint partitions also
        // establish that `dst` contains the complete product width.
        let _ = unsafe {
            let initialized =
                LimbOutput::assume_init_mut(dst.get_unchecked_mut(..a.len().unchecked_mul(2)));
            SharedEval::fused_add_shifted_in_place(initialized, active_middle, split)
        };
    }

    /// Square an equal-width operand with a recursively squared absolute difference.
    ///
    /// For `a = a0 + a1*B^S`, let `d = (a0-a1)^2`. Then the middle
    /// coefficient is `a0^2 + a1^2 - d = 2*a0*a1`. The difference is exactly
    /// `S` limbs, whereas sum-form Karatsuba may grow to `S+1`; this removes a
    /// guard limb from the third recursive square and its complete subtree.
    pub fn sqr_balanced<const ZERO_EXTENDED_HIGH: bool>(
        dst: &mut [Limb],
        a: &[Limb],
        scratch: &mut [Limb],
    ) {
        let split = a.len().div_ceil(2);
        // SAFETY: a.len() >= 2 gives 0 < split < a.len(). The valid limb
        // slice byte bound with Limb >= 2 bytes proves 2*split+1 fits usize.
        let (product_limbs, middle_limbs) = unsafe {
            let product = split.unchecked_mul(2);
            (product, product.unchecked_add(1))
        };
        // SAFETY: the direct square admits at least two limbs and splits at
        // ceil(a.len()/2), retaining nonempty low and high blocks.
        let (a0, a1) = unsafe { a.split_at_unchecked(split) };
        // SAFETY: a1's valid byte span proves twice its initialized width fits.
        let high_product_limbs = unsafe { a1.len().unchecked_mul(2) };
        // SAFETY: the two endpoint widths sum to the complete 2*a.len()-limb
        // square supplied by the caller. Consecutive partitions do not overlap.
        let (low_product, high_product) = unsafe {
            let (low, after_low) = dst.split_at_mut_unchecked(product_limbs);
            let (high, _) = after_low.split_at_mut_unchecked(high_product_limbs);
            (low, high)
        };

        // Endpoint squares run before the local difference occupies scratch, so
        // both may borrow the complete buffer. Below the square crossover, direct
        // basecase calls avoid recursive dispatcher frames.
        let subproducts_are_basecase = split < SQR_KARATSUBA_THRESHOLD;
        if subproducts_are_basecase {
            Schoolbook::sqr_nonempty(low_product, a0);
            Schoolbook::sqr_nonempty(high_product, a1);
        } else {
            // The low block clears the square crossover. The high block needs
            // dispatch only when the odd-width split makes it one limb shorter.
            Self::sqr(low_product, a0, scratch);
            if ZERO_EXTENDED_HIGH {
                Self::dispatch_sqr(high_product, a1, scratch);
            } else {
                Self::sqr(high_product, a1, scratch);
            }
        }

        // SAFETY: karatsuba_sqr_scratch_len reserves a split-limb difference,
        // its guarded 2*split+1-limb square, and the complete recursive tail.
        let (difference, middle_product, recursive_scratch) = unsafe {
            let (difference, after_difference) = scratch.split_at_mut_unchecked(split);
            let (middle, tail) = after_difference.split_at_mut_unchecked(middle_limbs);
            (difference, middle, tail)
        };
        if ZERO_EXTENDED_HIGH {
            let _ = Self::abs_difference_zero_extended(difference, a0, a1);
        } else {
            let _ = Self::abs_difference_equal_width(difference, a0, a1);
        }
        // SAFETY: middle_product retains exactly one guard above product_limbs.
        let (middle_value, _) = unsafe { middle_product.split_at_mut_unchecked(product_limbs) };
        if subproducts_are_basecase {
            Schoolbook::sqr_nonempty(middle_value, difference);
        } else {
            Self::sqr(middle_value, difference, recursive_scratch);
        }
        // The difference square writes exactly 2*S limbs. Reverse subtraction
        // initializes the remaining guard directly from the borrow of z0-d,
        // without reading the scratch buffer's previous guard value.
        Self::reverse_subtract_product_from_middle(middle_product, low_product);
        Self::add_product_to_middle(middle_product, high_product);
        let middle_len = SharedEval::active_len(middle_product);
        // SAFETY: active_len returns a prefix length inside middle_product.
        let (active_middle, _) = unsafe { middle_product.split_at_unchecked(middle_len) };
        debug_assert!(
            split <= dst.len() && active_middle.len() <= dst.len().saturating_sub(split),
            "balanced Karatsuba square middle coefficient exceeds the destination"
        );
        // SAFETY: `active_middle = 2*a0*a1` is the exact cross coefficient of
        // the square. Its shift by `split` therefore lies wholly within the
        // complete square destination established by the endpoint partitions.
        let _ = unsafe { SharedEval::fused_add_shifted_in_place(dst, active_middle, split) };
    }

    /// Choose the low-block width for balanced difference-form recursion.
    ///
    /// Even operands use equal halves. Odd operands place the extra limb in the
    /// low block and zero-extend the shorter high block for the difference product.
    /// Once children recurse, a half one limb below a power of two is rounded up:
    /// that exposes the balanced power-of-two specialization without padding a
    /// basecase leaf, where the extra work cannot be recovered recursively.
    pub const fn balanced_split_len(len: usize) -> usize {
        let half = len.div_ceil(2);
        // SAFETY: ceil(len/2) <= ceil(usize::MAX/2), so half+1 fits on all
        // supported pointer widths, even when sizing receives an arbitrary len.
        let rounded_half = unsafe { half.unchecked_add(1) };
        // A recursive split needs at least two limbs before alignment can grow
        // its low half; a two-limb operand must retain its one-limb high half.
        let recursive_min = if KARATSUBA_THRESHOLD < 2 {
            2
        } else {
            KARATSUBA_THRESHOLD
        };
        if half < recursive_min || half.is_multiple_of(2) || !rounded_half.is_power_of_two() {
            return half;
        }
        rounded_half
    }

    pub fn abs_difference_zero_extended(dst: &mut [Limb], low: &[Limb], high: &[Limb]) -> bool {
        debug_assert_eq!(
            dst.len(),
            low.len(),
            "difference must use the low-block width"
        );
        debug_assert!(high.len() <= low.len(), "high block exceeds the low block");
        debug_assert!(
            !high.is_empty(),
            "two-way splits retain a nonempty high block"
        );
        // SAFETY: two-way splits retain high.len() <= low.len(), and the
        // difference destination has exactly the low-block width.
        let ((shared_low, extension), (prefix, destination_extension)) = unsafe {
            (
                low.split_at_unchecked(high.len()),
                dst.split_at_mut_unchecked(high.len()),
            )
        };
        let ordering = if extension.iter().any(|limb| *limb != 0) {
            Ordering::Greater
        } else {
            shared_low.iter().rev().cmp(high.iter().rev())
        };

        if ordering == Ordering::Less {
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
            destination_extension.fill(0);
            true
        } else {
            // SAFETY: the three shared prefixes have the same width and are
            // disjoint. A borrow may leave this prefix, but the ordering proof
            // guarantees the copied low extension absorbs it.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    prefix.as_mut_ptr(),
                    shared_low.as_ptr(),
                    high.as_ptr(),
                    high.len(),
                )
            };
            for (dst_limb, low_limb) in destination_extension.iter_mut().zip(extension) {
                let (difference, underflow) = low_limb.overflowing_sub(borrow);
                *dst_limb = difference;
                borrow = Limb::from(underflow);
            }
            debug_assert_eq!(borrow, 0, "absolute difference underflowed");
            false
        }
    }
}
