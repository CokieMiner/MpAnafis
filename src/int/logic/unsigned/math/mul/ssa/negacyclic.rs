//! Odd-factor decomposition for medium Fermat-ring point products.
//!
//! For an odd `k` and `X = B^n`,
//!
//! ```text
//! X^k + 1 = (X + 1) Q,  Q = X^(k-1) - X^(k-2) + ... - X + 1.
//! ```
//!
//! A product modulo `X^k + 1` can therefore be recovered from one product
//! modulo `Q` and one modulo `X + 1`.  This replaces one `k*n`-limb product by
//! products of `(k-1)*n` and `n` limbs.  The implementation below is derived
//! directly from that factorization and uses only the portable multiplication
//! tower and limb add/subtract primitives.

#![allow(
    unsafe_code,
    reason = "the hot factorization kernels index exact block partitions proven by NegacyclicPlan"
)]

use core::{cmp::Ordering, num::NonZeroUsize};

use super::{
    LIMB_BITS, Limb, MulPlan, Multiplication, SSA_NEGACYCLIC_FACTOR3_THRESHOLD,
    SSA_NEGACYCLIC_FACTOR5_THRESHOLD, SharedEval, SsaCarry, SsaPointwise, SsaRing, TierCeiling,
};

const FACTOR_THREE: NonZeroUsize = NonZeroUsize::new(3).unwrap();
const FACTOR_FIVE: NonZeroUsize = NonZeroUsize::new(5).unwrap();

/// A preselected odd-factor decomposition for one fixed coefficient width.
#[derive(Clone, Copy)]
pub struct NegacyclicPlan {
    factor: NonZeroUsize,
    block_len: usize,
    quotient_len: usize,
    quotient_plan: MulPlan,
    small_plan: MulPlan,
}

impl NegacyclicPlan {
    /// Selects a decomposition only where its two smaller products are expected
    /// to repay the folding and CRT passes.
    pub fn new(modulus_limbs: usize) -> Option<Self> {
        let _modulus_bits = modulus_limbs.checked_mul(LIMB_BITS)?;
        let factor = if modulus_limbs >= SSA_NEGACYCLIC_FACTOR5_THRESHOLD
            && modulus_limbs.is_multiple_of(5)
        {
            FACTOR_FIVE
        } else if modulus_limbs >= SSA_NEGACYCLIC_FACTOR3_THRESHOLD
            && modulus_limbs.is_multiple_of(3)
        {
            FACTOR_THREE
        } else {
            return None;
        };
        let block_len = modulus_limbs.div_euclid(factor.get());
        if block_len == 0 {
            return None;
        }
        let quotient_len = modulus_limbs.checked_sub(block_len)?;
        Some(Self {
            factor,
            block_len,
            quotient_len,
            quotient_plan: Multiplication::select_plan(
                quotient_len,
                quotient_len,
                TierCeiling::Full,
            ),
            small_plan: Multiplication::select_plan(block_len, block_len, TierCeiling::Full),
        })
    }

    /// Scratch required by [`Self::mul_assign_left`].
    pub fn scratch_len(self) -> usize {
        let small_coeff_len = self
            .block_len
            .checked_add(1)
            .expect("small negacyclic coefficient width overflowed");
        let quotient_coeff_len = self
            .quotient_len
            .checked_add(1)
            .expect("quotient coefficient width overflowed");
        let quotient_work = Multiplication::required_scratch(self.quotient_len, self.quotient_len);
        let small_work = self
            .block_len
            .checked_mul(2)
            .and_then(|product_len| {
                product_len.checked_add(Multiplication::required_scratch(
                    self.block_len,
                    self.block_len,
                ))
            })
            .expect("small negacyclic product scratch overflowed");

        small_coeff_len
            .checked_mul(4)
            .and_then(|total| {
                quotient_coeff_len
                    .checked_mul(2)
                    .and_then(|quotients| total.checked_add(quotients))
            })
            .and_then(|total| {
                self.quotient_len
                    .checked_mul(2)
                    .and_then(|product| total.checked_add(product))
            })
            .and_then(|total| total.checked_add(quotient_work.max(small_work)))
            .expect("negacyclic scratch layout overflowed")
    }

    /// Multiplies two canonical residues, overwriting `left`.
    ///
    /// # Safety
    /// - `left` and `right` are disjoint `modulus_limbs + 1`-limb coefficients.
    /// - Neither operand is zero nor the special residue `2^N`.
    /// - `scratch` has at least [`Self::scratch_len`] limbs.
    pub unsafe fn mul_assign_left(
        self,
        left: &mut [Limb],
        right: &mut [Limb],
        scratch: &mut [Limb],
    ) {
        let factor = self.factor.get();
        let modulus_limbs = self
            .block_len
            .checked_mul(factor)
            .expect("planned negacyclic modulus width overflowed");
        let modulus_bits = modulus_limbs
            .checked_mul(LIMB_BITS)
            .expect("planned negacyclic modulus bit width overflowed");
        let small_bits = self
            .block_len
            .checked_mul(LIMB_BITS)
            .expect("planned small modulus bit width overflowed");
        let modulus_coeff_len = modulus_limbs
            .checked_add(1)
            .expect("planned negacyclic coefficient width overflowed");
        let small_coeff_len = self
            .block_len
            .checked_add(1)
            .expect("planned small coefficient width overflowed");
        let quotient_coeff_len = self
            .quotient_len
            .checked_add(1)
            .expect("planned quotient coefficient width overflowed");
        debug_assert_eq!(
            left.len(),
            modulus_coeff_len,
            "left negacyclic coefficient width differs"
        );
        debug_assert_eq!(
            right.len(),
            modulus_coeff_len,
            "right negacyclic coefficient width differs"
        );
        debug_assert!(
            scratch.len() >= self.scratch_len(),
            "negacyclic scratch is undersized"
        );

        let (left_small, after_left_small) = scratch.split_at_mut(small_coeff_len);
        let (right_small, after_right_small) = after_left_small.split_at_mut(small_coeff_len);
        let (fold_scratch, after_fold) = after_right_small.split_at_mut(small_coeff_len);
        let (small_product, after_small_product) = after_fold.split_at_mut(small_coeff_len);
        let (left_quotient, after_left_quotient) =
            after_small_product.split_at_mut(quotient_coeff_len);
        let (right_quotient, after_right_quotient) =
            after_left_quotient.split_at_mut(quotient_coeff_len);
        let quotient_product_len = self
            .quotient_len
            .checked_mul(2)
            .expect("planned quotient product width overflowed");
        let (quotient_product, work) = after_right_quotient.split_at_mut(quotient_product_len);

        fold_mod_x_plus_one(left_small, fold_scratch, left, self.block_len, factor);
        fold_mod_x_plus_one(right_small, fold_scratch, right, self.block_len, factor);

        quotient_residue(left_quotient, left, self.block_len, factor);
        quotient_residue(right_quotient, right, self.block_len, factor);

        // SAFETY: each quotient coefficient has exactly `quotient_len + 1`
        // initialized limbs, so these prefixes contain the complete data
        // widths. The two coefficients and the product buffer are disjoint
        // partitions of `scratch`.
        Multiplication::execute_plan(
            self.quotient_plan,
            quotient_product,
            unsafe { left_quotient.get_unchecked(..self.quotient_len) },
            unsafe { right_quotient.get_unchecked(..self.quotient_len) },
            work,
        );
        // SAFETY: the quotient product has `2 * quotient_len` limbs and `left`
        // is a complete, disjoint coefficient modulo `B^modulus_limbs + 1`.
        unsafe {
            reduce_product_mod_fermat(left, quotient_product, modulus_limbs);
        }

        let small_product_len = self
            .block_len
            .checked_mul(2)
            .expect("planned small product width overflowed");
        let (small_full_product, small_tower_scratch) = work.split_at_mut(small_product_len);
        // SAFETY: each small coefficient has exactly `block_len + 1`
        // initialized limbs, so these prefixes contain the complete data
        // widths. Both prefixes and the product buffer are disjoint scratch
        // partitions.
        Multiplication::execute_plan(
            self.small_plan,
            small_full_product,
            unsafe { left_small.get_unchecked(..self.block_len) },
            unsafe { right_small.get_unchecked(..self.block_len) },
            small_tower_scratch,
        );
        // SAFETY: `small_product` is one complete coefficient and the exact
        // product buffer contains two complete block-width halves.
        unsafe {
            SsaPointwise::reduce_full_product(small_product, small_full_product, self.block_len);
        }

        // t*k = small_product - left (mod X+1).  Compatibility of the two
        // residues guarantees a solution because Q(-1) = k.
        fold_mod_x_plus_one(left_small, fold_scratch, left, self.block_len, factor);
        // SAFETY: both buffers are complete canonical X+1 residues.
        unsafe {
            SsaRing::sub_in_place(small_product, left_small, small_bits);
            SsaRing::normalize(small_product, small_bits);
        }

        make_exactly_divisible_by_factor(small_product, self.factor);
        SharedEval::exact_div_odd_in_place(small_product, factor, SharedEval::invert_odd(factor));

        // `right` is dead after the point product, so use it to construct t*Q.
        build_times_quotient(right, small_product, self.block_len, factor);
        let escaped = SsaCarry::add_full_in_place(left, right);
        debug_assert_eq!(
            escaped, 0,
            "the CRT sum is strictly below twice the modulus"
        );
        // SAFETY: left is a complete coefficient for this Fermat ring.
        unsafe {
            SsaRing::normalize(left, modulus_bits);
        }
    }
}

/// Reduces one operand modulo `Q`.
fn quotient_residue(dst: &mut [Limb], src: &[Limb], block_len: usize, factor: usize) {
    assert!(block_len > 0, "quotient blocks must be nonempty");
    assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    // The accepted factors are three or five, so this subtraction is exact.
    let factor_minus_one = factor.wrapping_sub(1);
    let quotient_len = block_len
        .checked_mul(factor_minus_one)
        .expect("quotient residue width overflowed");
    let modulus_len = quotient_len
        .checked_add(block_len)
        .expect("negacyclic modulus width overflowed");
    let quotient_coeff_len = quotient_len
        .checked_add(1)
        .expect("quotient coefficient width overflowed");
    let modulus_coeff_len = modulus_len
        .checked_add(1)
        .expect("negacyclic coefficient width overflowed");
    assert_eq!(
        dst.len(),
        quotient_coeff_len,
        "quotient residue destination width differs"
    );
    assert_eq!(
        src.len(),
        modulus_coeff_len,
        "negacyclic source coefficient width differs"
    );
    dst.fill(0);
    // SAFETY: the exact-width checks above leave one guard above this prefix in
    // `dst` and a complete top block after it in `src`.
    unsafe { dst.get_unchecked_mut(..quotient_len) }
        .copy_from_slice(unsafe { src.get_unchecked(..quotient_len) });
    // SAFETY: `modulus_len == quotient_len + block_len < src.len()`.
    let top = unsafe { src.get_unchecked(quotient_len..modulus_len) };

    // X^(k-1) = X^(k-2) - X^(k-3) + ... + X - 1 (mod Q).
    // Add all positive terms first; their sum dominates the later negative
    // terms, so no borrow can escape the retained guard limb.
    for exponent in (1..factor_minus_one).step_by(2) {
        // `exponent < factor - 1`, so this product is strictly below the
        // checked `quotient_len = block_len * (factor - 1)`.
        let shift = exponent.wrapping_mul(block_len);
        // SAFETY: `exponent <= factor - 2` makes the suffix at least
        // `block_len + 1` limbs, while `top.len() == block_len`.
        let escaped = SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, top);
        debug_assert_eq!(escaped, 0, "Q residue retained its guard carry");
    }
    for exponent in (0..factor_minus_one).step_by(2) {
        // The same quotient-width bound proves this multiplication cannot
        // overflow and leaves a complete block-width suffix.
        let shift = exponent.wrapping_mul(block_len);
        // SAFETY: the same block-partition bound leaves room for `top`.
        let escaped = SsaCarry::sub_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, top);
        debug_assert_eq!(escaped, 0, "positive Q residue cannot underflow");
    }

    if compare_with_quotient_modulus(dst, block_len, factor) != Ordering::Less {
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
    assert!(block_len > 0, "fold blocks must be nonempty");
    assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    let coefficient_len = block_len
        .checked_add(1)
        .expect("small coefficient width overflowed");
    let modulus_len = block_len
        .checked_mul(factor)
        .expect("negacyclic modulus width overflowed");
    let modulus_coeff_len = modulus_len
        .checked_add(1)
        .expect("negacyclic coefficient width overflowed");
    assert_eq!(dst.len(), coefficient_len, "fold destination width differs");
    assert_eq!(
        negative.len(),
        coefficient_len,
        "fold scratch width differs"
    );
    assert_eq!(
        src.len(),
        modulus_coeff_len,
        "fold source coefficient width differs"
    );
    dst.fill(0);
    negative.fill(0);
    for exponent in (0..factor).step_by(2) {
        // `exponent < factor`, so this product is below the checked
        // `modulus_len = block_len * factor`.
        let start = exponent.wrapping_mul(block_len);
        // SAFETY: `exponent < factor` partitions the first `modulus_len`
        // source limbs into exact `block_len`-limb blocks.
        let escaped = SsaCarry::add_full_in_place(dst, unsafe {
            src.get_unchecked(start..start.wrapping_add(block_len))
        });
        debug_assert_eq!(escaped, 0, "even block sum fits its guard limb");
    }
    for exponent in (1..factor).step_by(2) {
        // This is the same checked block partition as the even exponents.
        let start = exponent.wrapping_mul(block_len);
        // SAFETY: this is the same exact source-block partition.
        let escaped = SsaCarry::add_full_in_place(negative, unsafe {
            src.get_unchecked(start..start.wrapping_add(block_len))
        });
        debug_assert_eq!(escaped, 0, "odd block sum fits its guard limb");
    }
    // A guard at X^k contributes -guard because k is odd.
    // SAFETY: `src.len() == modulus_len + 1`, so this is the source guard.
    let source_guard = unsafe { *src.get_unchecked(modulus_len) };
    let guard_escape = SsaCarry::add_full_in_place(negative, &[source_guard]);
    debug_assert_eq!(guard_escape, 0, "single source guard fits the fold");

    let small_bits = block_len
        .checked_mul(LIMB_BITS)
        .expect("small negacyclic modulus bit width overflowed");
    // SAFETY: both buffers contain one complete coefficient modulo X+1.
    unsafe {
        SsaRing::normalize(dst, small_bits);
        SsaRing::normalize(negative, small_bits);
        SsaRing::sub_in_place(dst, negative, small_bits);
        SsaRing::normalize(dst, small_bits);
    }
}

/// Adds the unique small multiple of `X+1` that makes `value` divisible by k.
fn make_exactly_divisible_by_factor(value: &mut [Limb], factor: NonZeroUsize) {
    assert!(!value.is_empty(), "the X+1 coefficient must retain a guard");
    assert!(
        factor == FACTOR_THREE || factor == FACTOR_FIVE,
        "unsupported negacyclic factor"
    );
    let remainder = value
        .iter()
        // Both addends are below `factor <= 5`, so this wrapping sum is exact.
        .fold(0_usize, |acc, limb| {
            acc.wrapping_add(*limb % factor) % factor
        })
        % factor;
    let mut multiple = 0_usize;
    // An odd factor makes two invertible, so the loop finds a solution before
    // `multiple == factor`; all wrapping arithmetic below is therefore exact.
    while remainder.wrapping_add(multiple.wrapping_mul(2)) % factor != 0 {
        multiple = multiple.wrapping_add(1);
    }

    let escaped = SsaCarry::add_full_in_place(value, &[multiple]);
    debug_assert_eq!(escaped, 0, "low X+1 adjustment retains its carry");
    let guard_index = value.len().wrapping_sub(1);
    // SAFETY: the release check above proves `value` is nonempty.
    let guard = unsafe { value.get_unchecked_mut(guard_index) };
    *guard = guard.wrapping_add(multiple);
}

/// Builds `value * Q` exactly in a full `X^k+1` coefficient.
fn build_times_quotient(dst: &mut [Limb], value: &[Limb], block_len: usize, factor: usize) {
    assert!(block_len > 0, "quotient blocks must be nonempty");
    assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    let modulus_len = block_len
        .checked_mul(factor)
        .expect("negacyclic modulus width overflowed");
    let modulus_coeff_len = modulus_len
        .checked_add(1)
        .expect("negacyclic coefficient width overflowed");
    let small_coeff_len = block_len
        .checked_add(1)
        .expect("small coefficient width overflowed");
    assert_eq!(
        dst.len(),
        modulus_coeff_len,
        "CRT destination width differs"
    );
    assert_eq!(
        value.len(),
        small_coeff_len,
        "small CRT value width differs"
    );
    dst.fill(0);
    for exponent in (0..factor).step_by(2) {
        // `exponent < factor`, so this product is below the checked modulus
        // width and the destination retains the coefficient guard.
        let shift = exponent.wrapping_mul(block_len);
        // SAFETY: `exponent < factor` leaves at least `block_len + 1` limbs in
        // the exact `modulus_len + 1` destination suffix.
        let escaped = SsaCarry::add_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, value);
        debug_assert_eq!(escaped, 0, "t*Q positive terms fit the coefficient");
    }
    for exponent in (1..factor).step_by(2) {
        // This is the same checked block partition as the even exponents.
        let shift = exponent.wrapping_mul(block_len);
        // SAFETY: the same exact block partition leaves room for `value`.
        let escaped = SsaCarry::sub_full_in_place(unsafe { dst.get_unchecked_mut(shift..) }, value);
        debug_assert_eq!(escaped, 0, "t*Q is nonnegative");
    }
}

fn subtract_quotient_modulus(dst: &mut [Limb], block_len: usize, factor: usize) {
    assert!(block_len > 0, "quotient blocks must be nonempty");
    assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    // The accepted factors are three or five, so this subtraction is exact.
    let factor_minus_one = factor.wrapping_sub(1);
    let quotient_len = block_len
        .checked_mul(factor_minus_one)
        .expect("quotient modulus width overflowed");
    let quotient_coeff_len = quotient_len
        .checked_add(1)
        .expect("quotient coefficient width overflowed");
    assert_eq!(
        dst.len(),
        quotient_coeff_len,
        "quotient modulus destination width differs"
    );
    let low_borrow = SsaCarry::sub_full_in_place(dst, &[1]);
    debug_assert_eq!(low_borrow, 0, "a value at least Q is nonzero");
    for exponent in 1..factor {
        // `exponent < factor` and the last block start equals
        // `quotient_len`, so this multiplication is within the checked width.
        let shift = exponent.wrapping_mul(block_len);
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

fn compare_with_quotient_modulus(value: &[Limb], block_len: usize, factor: usize) -> Ordering {
    assert!(block_len > 0, "quotient blocks must be nonempty");
    assert!(factor == 3 || factor == 5, "unsupported negacyclic factor");
    for index in (0..value.len()).rev() {
        let block = index.div_euclid(block_len);
        // Euclidean division gives `block * block_len <= index`, so neither
        // this product nor the following subtraction can overflow.
        let within_block = index.wrapping_sub(block.wrapping_mul(block_len));
        let modulus_limb = if block >= factor.wrapping_sub(1) || block.is_multiple_of(2) {
            Limb::from(index == 0)
        } else {
            Limb::MAX
        };
        debug_assert!(within_block < block_len, "index belongs to one exact block");
        // SAFETY: the reverse range yields exactly the valid indices of `value`.
        match unsafe { value.get_unchecked(index) }.cmp(&modulus_limb) {
            Ordering::Equal => {}
            ordering @ (Ordering::Less | Ordering::Greater) => return ordering,
        }
    }
    Ordering::Equal
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
    // SAFETY: the contract gives one destination guard above the data limbs.
    *unsafe { dst.get_unchecked_mut(modulus_limbs) } = 0;
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
    }
}

#[cfg(test)]
#[path = "../tests/tiers/negacyclic.rs"]
mod tests;
