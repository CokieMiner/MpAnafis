//! Negacyclic execution through residues modulo X+1 and its odd-factor quotient.

#![expect(
    unsafe_code,
    reason = "the hot factorization kernels index exact block partitions proven by NegacyclicPlan"
)]

use core::{cmp::Ordering, num::NonZeroUsize};

use crate::parallel::SequentialExecutor;

use super::{
    DoubleLimb, LIMB_BITS, Limb, Multiplication, NegacyclicPlan, SharedEval, SsaCarry,
    SsaPointwise, SsaRing,
};

impl NegacyclicPlan {
    /// Multiplies two canonical residues, overwriting `left`.
    ///
    /// # Safety
    /// Operands and result have exactly `modulus_limbs+1` initialized limbs;
    /// scratch has at least [`Self::scratch_len`] initialized limbs. All spans
    /// are disjoint. Operands are canonical and differ from the special
    /// residue `2^N`; zero is valid. `right` remains immutable.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear quotient/small-product/CRT pass keeps the arena's phase lifetimes explicit"
    )]
    pub unsafe fn mul_assign_left(
        &self,
        left: &mut [Limb],
        right: &[Limb],
        result: &mut [Limb],
        scratch: &mut [Limb],
    ) {
        let factor = self.factor.get();
        // SAFETY: for_factor checked modulus_bits=modulus_limbs*LIMB_BITS;
        // with LIMB_BITS>=16 the retained data width plus one guard fits usize.
        let modulus_coeff_len = unsafe { self.modulus_limbs.unchecked_add(1) };
        debug_assert_eq!(left.len(), modulus_coeff_len, "left coefficient width");
        debug_assert_eq!(right.len(), modulus_coeff_len, "right coefficient width");
        debug_assert!(
            scratch.len() >= self.scratch_len,
            "negacyclic scratch is undersized"
        );

        // SAFETY: for_factor checked this exact linear layout, and the caller
        // supplies scratch_len initialized limbs. All six spans are disjoint.
        let (
            [
                left_small,
                right_small,
                fold_scratch,
                left_quotient,
                right_quotient,
                quotient_product,
            ],
            work,
        ) = unsafe {
            let (left_small, after_left_small) =
                scratch.split_at_mut_unchecked(self.small_coeff_len);
            let (right_small, after_right_small) =
                after_left_small.split_at_mut_unchecked(self.small_coeff_len);
            let (fold, after_fold) = after_right_small.split_at_mut_unchecked(self.small_coeff_len);
            let (left_quotient, after_left_quotient) =
                after_fold.split_at_mut_unchecked(self.quotient_coeff_len);
            let (right_quotient, after_right_quotient) =
                after_left_quotient.split_at_mut_unchecked(self.quotient_coeff_len);
            let (product, work) =
                after_right_quotient.split_at_mut_unchecked(self.quotient_product_len);
            (
                [
                    left_small,
                    right_small,
                    fold,
                    left_quotient,
                    right_quotient,
                    product,
                ],
                work,
            )
        };

        fold_mod_x_plus_one(left_small, fold_scratch, left, self.block_len.get(), factor);
        fold_mod_x_plus_one(
            right_small,
            fold_scratch,
            right,
            self.block_len.get(),
            factor,
        );

        quotient_residue(left_quotient, left, self);
        quotient_residue(right_quotient, right, self);

        // SAFETY: each quotient coefficient has exactly `quotient_len + 1`
        // initialized limbs, so these prefixes contain the complete data
        // widths. The two coefficients and the product buffer are disjoint
        // partitions of `scratch`.
        let left_quotient_input = unsafe { left_quotient.get_unchecked(..self.quotient_len) };
        // SAFETY: the right coefficient has the same complete initialized prefix.
        let right_quotient_input = unsafe { right_quotient.get_unchecked(..self.quotient_len) };
        Multiplication::execute_plan_with_executor(
            self.quotient_plan,
            quotient_product,
            left_quotient_input,
            right_quotient_input,
            work,
            &SequentialExecutor,
        );
        // SAFETY: the quotient product has `2 * quotient_len` limbs and `left`
        // is a complete, disjoint coefficient modulo `B^modulus_limbs + 1`.
        unsafe {
            reduce_product_mod_fermat(left, quotient_product, self.modulus_limbs);
        }

        // Both operand folds are complete, so their temporary becomes the
        // small result. Reduction consumed the quotient product in full; its
        // prefix now holds the small full product, disjoint from tower scratch.
        let small_product = fold_scratch;
        // SAFETY: k>=3 makes 2*block_len <= 2*(k-1)*block_len; the quotient
        // product's last read ended before reusing this initialized prefix.
        let (small_full_product, _) =
            unsafe { quotient_product.split_at_mut_unchecked(self.small_product_len) };
        // SAFETY: each small coefficient has exactly `block_len + 1`
        // initialized limbs, so these prefixes contain the complete data
        // widths. Both prefixes and the product buffer are disjoint scratch
        // partitions.
        let left_small_input = unsafe { left_small.get_unchecked(..self.block_len.get()) };
        // SAFETY: the right coefficient has the same complete initialized prefix.
        let right_small_input = unsafe { right_small.get_unchecked(..self.block_len.get()) };
        // Folding can create the canonical residue X = -1 even when both
        // outer guards are zero. Its guard participates in this small product.
        // SAFETY: all small coefficients are canonical, complete, and disjoint.
        unsafe {
            if !SsaPointwise::write_special_residue_product(
                small_product,
                left_small,
                right_small,
                self.small_bits,
            ) {
                Multiplication::execute_plan_with_executor(
                    self.small_plan,
                    small_full_product,
                    left_small_input,
                    right_small_input,
                    work,
                    &SequentialExecutor,
                );
                SsaPointwise::reduce_full_product(
                    small_product,
                    small_full_product,
                    self.block_len.get(),
                );
            }
        }

        // t*k = small_product - left (mod X+1).  Compatibility of the two
        // residues guarantees a solution because Q(-1) = k.
        // The small multiplication consumed right_small. Reuse that complete
        // coefficient for the final fold's negative accumulator.
        fold_mod_x_plus_one(left_small, right_small, left, self.block_len.get(), factor);
        // SAFETY: both buffers are complete canonical X+1 residues.
        unsafe {
            SsaRing::sub_in_place(small_product, left_small, self.small_bits);
            let _ = SsaRing::normalize(small_product, self.small_bits);
        }

        make_exactly_divisible_by_factor(small_product, self.factor);
        SharedEval::exact_div_odd_in_place(small_product, factor, self.factor_inverse);

        // The caller's result coefficient holds t*Q; the right operand remains
        // available for another product with the same transformed coefficient.
        build_times_quotient(result, small_product, self.block_len.get(), factor);
        let escaped = SsaCarry::add_full_in_place(left, result);
        debug_assert_eq!(
            escaped, 0,
            "the CRT sum is strictly below twice the modulus"
        );
        // SAFETY: left is a complete coefficient for this Fermat ring.
        unsafe {
            let _ = SsaRing::normalize(left, self.modulus_bits);
        }
    }
}

/// Reduces one operand modulo `Q`.
fn quotient_residue(dst: &mut [Limb], src: &[Limb], plan: &NegacyclicPlan) {
    let block_len = plan.block_len.get();
    let factor = plan.factor.get();
    // SAFETY: for_factor admits only k=3 or 5.
    let factor_minus_one = unsafe { factor.unchecked_sub(1) };
    // The driver partitions complete canonical and quotient coefficients.
    let quotient_len = plan.quotient_len;
    let modulus_len = plan.modulus_limbs;
    // SAFETY: the exact destination width retains its guard at quotient_len;
    // the complete data prefix is overwritten by the following copy.
    unsafe {
        *dst.get_unchecked_mut(quotient_len) = 0;
    }
    // SAFETY: the driver's exact partitions leave one guard above this prefix
    // in dst and a complete top block after it in src.
    unsafe { dst.get_unchecked_mut(..quotient_len) }
        .copy_from_slice(unsafe { src.get_unchecked(..quotient_len) });
    // SAFETY: `modulus_len == quotient_len + block_len < src.len()`.
    let top = unsafe { src.get_unchecked(quotient_len..modulus_len) };

    // X^(k-1) = X^(k-2) - X^(k-3) + ... + X - 1 (mod Q).
    // Add all positive terms first; their sum dominates the later negative
    // terms, so no borrow can escape the retained guard limb.
    for exponent in (1..factor_minus_one).step_by(2) {
        // SAFETY: exponent<factor-1 bounds this offset by the checked quotient width.
        let shift = unsafe { exponent.unchecked_mul(block_len) };
        // SAFETY: `exponent <= factor - 2` makes the suffix at least
        // `block_len + 1` limbs, while `top.len() == block_len`.
        let escaped = SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, top);
        debug_assert_eq!(escaped, 0, "Q residue retained its guard carry");
    }
    for exponent in (0..factor_minus_one).step_by(2) {
        // SAFETY: exponent<factor-1 bounds this offset by the checked quotient width.
        let shift = unsafe { exponent.unchecked_mul(block_len) };
        // SAFETY: the same block-partition bound leaves room for `top`.
        let escaped = SsaCarry::sub_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, top);
        debug_assert_eq!(escaped, 0, "positive Q residue cannot underflow");
    }

    // SAFETY: the driver supplies exactly quotient_coeff_len initialized limbs;
    // the copy and guarded accumulations retain that complete coefficient.
    if unsafe { plan.compare_with_quotient_modulus(dst) } != Ordering::Less {
        subtract_quotient_modulus(dst, block_len, factor);
    }
    debug_assert_eq!(
        // SAFETY: `dst.len() == quotient_len + 1`, so this is its guard limb.
        unsafe { *dst.get_unchecked(quotient_len) },
        0,
        "canonical Q residue fits its data limbs"
    );
}

/// Evaluates a base-X operand at X=-1, producing a residue modulo X+1.
fn fold_mod_x_plus_one(
    dst: &mut [Limb],
    negative: &mut [Limb],
    src: &[Limb],
    block_len: usize,
    factor: usize,
) {
    debug_assert!(block_len > 0, "fold blocks must be nonempty");
    debug_assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    // SAFETY: the owning plan admitted k=3 or 5 and a representable
    // k*block_len*LIMB_BITS, so both data widths and their guards fit usize.
    let (coefficient_len, modulus_len, modulus_coeff_len) = unsafe {
        let modulus = block_len.unchecked_mul(factor);
        (
            block_len.unchecked_add(1),
            modulus,
            modulus.unchecked_add(1),
        )
    };
    debug_assert_eq!(dst.len(), coefficient_len, "fold destination width differs");
    debug_assert_eq!(
        negative.len(),
        coefficient_len,
        "fold scratch width differs"
    );
    debug_assert_eq!(
        src.len(),
        modulus_coeff_len,
        "fold source coefficient width differs"
    );
    // Each initialized accumulator starts from its first block and a zero guard.
    // SAFETY: both destinations hold exactly `block_len` data limbs plus a
    // guard, and the source partitions below are exact `block_len` blocks.
    unsafe { dst.get_unchecked_mut(..block_len) }
        .copy_from_slice(unsafe { src.get_unchecked(..block_len) });
    // SAFETY: `block_len` is the guard index of both complete accumulators.
    unsafe {
        *dst.get_unchecked_mut(block_len) = 0;
    }
    // SAFETY: the second source block is exact and disjoint from the first.
    // `factor >= 3`, so twice the block width stays below the validated
    // `modulus_len` and remains representable on every pointer width.
    unsafe { negative.get_unchecked_mut(..block_len) }
        .copy_from_slice(unsafe { src.get_unchecked(block_len..block_len.unchecked_mul(2)) });
    // SAFETY: same guard-width proof as the even accumulator.
    unsafe {
        *negative.get_unchecked_mut(block_len) = 0;
    }
    for exponent in (2..factor).step_by(2) {
        // SAFETY: exponent<factor bounds this exact offset by modulus_len.
        let start = unsafe { exponent.unchecked_mul(block_len) };
        // SAFETY: `exponent < factor` partitions the first `modulus_len`
        // source limbs into exact `block_len`-limb blocks.
        let escaped = SsaCarry::add_full_in_place(dst, unsafe {
            src.get_unchecked(start..start.unchecked_add(block_len))
        });
        debug_assert_eq!(escaped, 0, "even block sum fits its guard limb");
    }
    for exponent in (3..factor).step_by(2) {
        // SAFETY: exponent<factor bounds this exact offset by modulus_len.
        let start = unsafe { exponent.unchecked_mul(block_len) };
        // SAFETY: this is the same exact source-block partition.
        let escaped = SsaCarry::add_full_in_place(negative, unsafe {
            src.get_unchecked(start..start.unchecked_add(block_len))
        });
        debug_assert_eq!(escaped, 0, "odd block sum fits its guard limb");
    }
    // A guard at X^k contributes -guard because k is odd.
    // SAFETY: `src.len() == modulus_len + 1`, so this is the source guard.
    let source_guard = unsafe { *src.get_unchecked(modulus_len) };
    let guard_escape = SsaCarry::add_full_in_place(negative, &[source_guard]);
    debug_assert_eq!(guard_escape, 0, "single source guard fits the fold");

    // SAFETY: block_len*LIMB_BITS is the checked small ring width of the plan.
    let small_bits = unsafe { block_len.unchecked_mul(LIMB_BITS) };
    // SAFETY: both buffers contain one complete coefficient modulo X+1.
    unsafe {
        let _ = SsaRing::normalize(dst, small_bits);
        let _ = SsaRing::normalize(negative, small_bits);
        SsaRing::sub_in_place(dst, negative, small_bits);
        let _ = SsaRing::normalize(dst, small_bits);
    }
}

/// Adds the unique small multiple of `X+1` that makes `value` divisible by k.
fn make_exactly_divisible_by_factor(value: &mut [Limb], factor: NonZeroUsize) {
    debug_assert!(!value.is_empty(), "the X+1 coefficient must retain a guard");
    debug_assert!(
        matches!(factor.get(), 3 | 5),
        "unsupported negacyclic factor"
    );
    // Each accepted factor instantiates a constant-divisor remainder.
    let remainder = match factor.get() {
        3 => limb_sum_mod::<3>(value),
        _ => limb_sum_mod::<5>(value),
    };
    // B = 1 modulo 3 and 5 on all supported limb widths, hence X+1 = 2.
    // The inverses of -2 are 1 modulo 3 and 2 modulo 5.
    let multiple = if factor.get() == 3 {
        remainder
    } else {
        // SAFETY: the factor-five remainder is at most four, so doubling fits.
        let doubled = unsafe { remainder.unchecked_mul(2) };
        if doubled >= 5 {
            // SAFETY: 5<=doubled<=8, so one exact subtraction reduces modulo five.
            unsafe { doubled.unchecked_sub(5) }
        } else {
            doubled
        }
    };

    let escaped = SsaCarry::add_full_in_place(value, &[multiple]);
    debug_assert_eq!(escaped, 0, "low X+1 adjustment retains its carry");
    // SAFETY: the planned small coefficient retains at least its guard limb.
    let guard_index = unsafe { value.len().unchecked_sub(1) };
    // SAFETY: the planned X+1 coefficient always retains one guard limb.
    let guard = unsafe { value.get_unchecked_mut(guard_index) };
    // SAFETY: the canonical input guard was at most one; the low addition
    // raises it by at most one, and multiple<=4. Thus the result is at most six.
    *guard = unsafe { guard.unchecked_add(multiple) };
}

/// Sums `value` modulo a compile-time odd factor with one final reduction.
fn limb_sum_mod<const FACTOR: usize>(value: &[Limb]) -> usize {
    const { assert!(FACTOR == 3 || FACTOR == 5, "unsupported negacyclic factor") }
    // SAFETY: the compile-time assertion admits only nonzero divisors three and five.
    let factor = unsafe { NonZeroUsize::new_unchecked(FACTOR) };
    let sum = value.iter().fold(DoubleLimb::MIN, |acc, &limb| {
        #[expect(
            clippy::as_conversions,
            reason = "DoubleLimb has twice Limb::BITS on 16/32/64-bit targets, so this native unsigned digit widens exactly"
        )]
        let wide_digit = limb as DoubleLimb;
        // SAFETY: a native-limb slice has n<=isize::MAX/size_of::<Limb>()
        // elements. With B=2^Limb::BITS, every nonnegative partial sum is
        // <=n*(B-1)<B^2 and fits DoubleLimb on 16/32/64-bit targets.
        unsafe { acc.unchecked_add(wide_digit) }
    });
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "The low cast selects the radix-B digit; shifting the double-width sum by Limb::BITS leaves its complete high digit on every supported pointer width"
    )]
    let (low, high) = (sum as Limb, (sum >> Limb::BITS) as Limb);
    // B == 1 modulo three and five. Fold the two digits and retain the
    // radix-B carry as one before the single constant-factor remainder.
    let (folded, carry) = low.overflowing_add(high);
    let remainder = folded % factor;
    // SAFETY: remainder<FACTOR<=5 and carry is binary, so corrected<=5.
    let corrected = unsafe { remainder.unchecked_add(usize::from(carry)) };
    if corrected == FACTOR { 0 } else { corrected }
}

/// Builds `value * Q` exactly in a full `X^k+1` coefficient.
fn build_times_quotient(dst: &mut [Limb], value: &[Limb], block_len: usize, factor: usize) {
    debug_assert!(block_len > 0, "quotient blocks must be nonempty");
    debug_assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    // SAFETY: the plan checked factor*block_len*LIMB_BITS, and LIMB_BITS>=16
    // leaves room for one guard above the full and small data widths.
    let (modulus_coeff_len, small_coeff_len) = unsafe {
        let modulus = block_len.unchecked_mul(factor);
        (modulus.unchecked_add(1), block_len.unchecked_add(1))
    };
    debug_assert_eq!(
        dst.len(),
        modulus_coeff_len,
        "CRT destination width differs"
    );
    debug_assert_eq!(
        value.len(),
        small_coeff_len,
        "small CRT value width differs"
    );
    // The first positive term initializes the low prefix; higher digits start at zero.
    // SAFETY: the small value holds exactly `block_len` data limbs plus a
    // guard, and the destination spans the complete `modulus_len + 1` width,
    // which covers that prefix because `factor >= 3`.
    unsafe { dst.get_unchecked_mut(..small_coeff_len) }
        .copy_from_slice(unsafe { value.get_unchecked(..small_coeff_len) });
    // SAFETY: the copied prefix ends below the complete destination width.
    unsafe { dst.get_unchecked_mut(small_coeff_len..) }.fill(0);
    for exponent in (2..factor).step_by(2) {
        // SAFETY: exponent<factor bounds this offset by the validated modulus width.
        let shift = unsafe { exponent.unchecked_mul(block_len) };
        // SAFETY: `exponent < factor` leaves at least `block_len + 1` limbs in
        // the exact `modulus_len + 1` destination suffix.
        let escaped = SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, value);
        debug_assert_eq!(escaped, 0, "t*Q positive terms fit the coefficient");
    }
    for exponent in (1..factor).step_by(2) {
        // SAFETY: exponent<factor bounds this offset by the validated modulus width.
        let shift = unsafe { exponent.unchecked_mul(block_len) };
        // SAFETY: the same exact block partition leaves room for `value`.
        let escaped = SsaCarry::sub_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, value);
        debug_assert_eq!(escaped, 0, "t*Q is nonnegative");
    }
}

fn subtract_quotient_modulus(dst: &mut [Limb], block_len: usize, factor: usize) {
    debug_assert!(block_len > 0, "quotient blocks must be nonempty");
    debug_assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    // SAFETY: factor is 3 or 5, and the plan checked the larger full ring bit
    // width. The quotient's data width and retained guard are representable.
    let quotient_coeff_len = unsafe {
        let quotient = block_len.unchecked_mul(factor.unchecked_sub(1));
        quotient.unchecked_add(1)
    };
    debug_assert_eq!(
        dst.len(),
        quotient_coeff_len,
        "quotient modulus destination width differs"
    );
    let low_borrow = SsaCarry::sub_full_in_place(dst, &[1]);
    debug_assert_eq!(low_borrow, 0, "a value at least Q is nonzero");
    for exponent in 1..factor {
        // SAFETY: exponent<=factor-1 places this offset at most at quotient_len.
        let shift = unsafe { exponent.unchecked_mul(block_len) };
        let escaped = if exponent.is_multiple_of(2) {
            // SAFETY: the final exponent starts at `quotient_len`, leaving the
            // one-limb guard suffix required by this subtraction.
            SsaCarry::sub_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, &[1])
        } else {
            // SAFETY: the same exact block offset leaves a one-limb suffix.
            SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, &[1])
        };
        debug_assert_eq!(escaped, 0, "exact Q subtraction stays in range");
    }
}

/// Reduces a product shorter than two full modulus widths using B^n=-1.
///
/// # Safety
/// `dst` has exactly `modulus_limbs + 1` limbs and `product` has between
/// `modulus_limbs` and `2 * modulus_limbs` limbs. The buffers are initialized
/// and disjoint.
unsafe fn reduce_product_mod_fermat(dst: &mut [Limb], product: &[Limb], modulus_limbs: usize) {
    // SAFETY: the contract gives both complete low prefixes.
    unsafe { dst.get_unchecked_mut(..modulus_limbs) }
        .copy_from_slice(unsafe { product.get_unchecked(..modulus_limbs) });
    // SAFETY: the contract gives a product at least this wide.
    let high = unsafe { product.get_unchecked(modulus_limbs..) };
    // SAFETY: the high product has at most `modulus_limbs` limbs, while the
    // selected destination prefix has exactly that width.
    let borrow =
        SsaCarry::sub_full_in_place(unsafe { dst.get_unchecked_mut(..modulus_limbs) }, high);
    if borrow != 0 {
        // SAFETY: the destination data prefix has exactly `modulus_limbs` limbs.
        let carry =
            SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(..modulus_limbs) }, &[1]);
        // SAFETY: the contract gives the guard at this exact index.
        *unsafe { dst.get_unchecked_mut(modulus_limbs) } = carry;
    } else {
        // SAFETY: the complete destination coefficient retains its guard here.
        *unsafe { dst.get_unchecked_mut(modulus_limbs) } = 0;
    }
}
