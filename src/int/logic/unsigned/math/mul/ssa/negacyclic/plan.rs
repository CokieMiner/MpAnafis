//! Odd-factor selection and workspace planning for Fermat-ring point products.
//!
//! For odd k and X = B^n, X^k + 1 = (X + 1)Q with
//! Q = X^(k-1) - X^(k-2) + ... - X + 1. Products modulo Q and X+1 replace
//! one k*n-limb product with products of (k-1)*n and n limbs.

#![expect(
    unsafe_code,
    reason = "Admitted odd factors and checked ring widths prove quotient dimensions and complete guarded comparison spans"
)]

use core::{cmp::Ordering, num::NonZeroUsize};

use super::{
    LIMB_BITS, Limb, MulPlan, Multiplication, SSA_NEGACYCLIC_FACTOR3_THRESHOLD,
    SSA_NEGACYCLIC_FACTOR5_THRESHOLD, SharedEval, TierCeiling,
};

const FACTOR_THREE: NonZeroUsize = NonZeroUsize::new(3).unwrap();
const FACTOR_FIVE: NonZeroUsize = NonZeroUsize::new(5).unwrap();

/// A preselected odd-factor decomposition for one fixed coefficient width.
#[derive(Clone, Copy, Debug)]
pub struct NegacyclicPlan {
    pub factor: NonZeroUsize,
    pub factor_inverse: Limb,
    pub block_len: NonZeroUsize,
    pub modulus_limbs: usize,
    pub modulus_bits: usize,
    pub small_bits: usize,
    pub quotient_len: usize,
    pub small_coeff_len: usize,
    pub quotient_coeff_len: usize,
    pub quotient_product_len: usize,
    pub small_product_len: usize,
    /// Scratch required by [`Self::mul_assign_left`].
    pub scratch_len: usize,
    pub quotient_plan: MulPlan,
    pub small_plan: MulPlan,
}

impl NegacyclicPlan {
    /// Constructs a legal factor independently of its price.
    /// Only factors three and five with nonempty exact blocks are supported.
    pub fn for_factor(modulus_limbs: usize, factor: NonZeroUsize) -> Option<Self> {
        if !matches!(factor.get(), 3 | 5) || !modulus_limbs.is_multiple_of(factor.get()) {
            return None;
        }
        let modulus_bits = modulus_limbs.checked_mul(LIMB_BITS)?;
        let block_len = NonZeroUsize::new(modulus_limbs.div_euclid(factor.get()))?;
        // SAFETY: 0 < block_len <= modulus_limbs and the checked bit width
        // bounds modulus_limbs by usize::MAX/LIMB_BITS. LIMB_BITS>=16 proves
        // the two guard additions and doubled data widths representable.
        let (
            small_bits,
            quotient_len,
            small_coeff_len,
            quotient_coeff_len,
            quotient_product_len,
            small_product_len,
        ) = unsafe {
            let quotient = modulus_limbs.unchecked_sub(block_len.get());
            (
                block_len.get().unchecked_mul(LIMB_BITS),
                quotient,
                block_len.get().unchecked_add(1),
                quotient.unchecked_add(1),
                quotient.unchecked_mul(2),
                block_len.get().unchecked_mul(2),
            )
        };
        let quotient_work = Multiplication::required_scratch(quotient_len, quotient_len);
        // The quotient product dies before the small product starts. Its
        // 2*(k-1)*block_len limbs contain the latter's 2*block_len output.
        let small_work = Multiplication::required_scratch(block_len.get(), block_len.get());
        let scratch_len = small_coeff_len
            .checked_mul(3)?
            .checked_add(quotient_coeff_len.checked_mul(2)?)?
            .checked_add(quotient_product_len)?
            .checked_add(quotient_work.max(small_work))?;
        Some(Self {
            factor,
            factor_inverse: SharedEval::invert_odd(factor.get()),
            block_len,
            modulus_limbs,
            modulus_bits,
            small_bits,
            quotient_len,
            small_coeff_len,
            quotient_coeff_len,
            quotient_product_len,
            small_product_len,
            scratch_len,
            quotient_plan: Multiplication::select_plan(
                quotient_len,
                quotient_len,
                TierCeiling::Full,
            ),
            small_plan: Multiplication::select_plan(
                block_len.get(),
                block_len.get(),
                TierCeiling::Full,
            ),
        })
    }

    /// Prices the executed factor selection without constructing tower plans.
    pub fn product_cost(modulus_limbs: usize) -> usize {
        Self::select_factor(modulus_limbs).map_or_else(
            || Multiplication::structural_product_work(modulus_limbs, modulus_limbs),
            |factor| Self::factor_cost(modulus_limbs, factor),
        )
    }

    /// Applies the tuned factor gates before comparing overlapping candidates.
    pub fn select_factor(modulus_limbs: usize) -> Option<NonZeroUsize> {
        let admits_five =
            modulus_limbs >= SSA_NEGACYCLIC_FACTOR5_THRESHOLD && modulus_limbs.is_multiple_of(5);
        let admits_three =
            modulus_limbs >= SSA_NEGACYCLIC_FACTOR3_THRESHOLD && modulus_limbs.is_multiple_of(3);
        match (admits_three, admits_five) {
            (false, false) => None,
            (true, false) => Self::cheaper_single_factor(modulus_limbs, FACTOR_THREE),
            (false, true) => Self::cheaper_single_factor(modulus_limbs, FACTOR_FIVE),
            (true, true) => Self::cheaper_factor(modulus_limbs),
        }
    }

    /// Selects a factor only when its modeled cost beats the unfactored product.
    fn cheaper_single_factor(modulus_limbs: usize, factor: NonZeroUsize) -> Option<NonZeroUsize> {
        let full = Multiplication::structural_product_work(modulus_limbs, modulus_limbs);
        let factorized = Self::factor_cost(modulus_limbs, factor);
        (factorized < full).then_some(factor)
    }

    /// Compares both decompositions and the full product, including linear work.
    fn cheaper_factor(modulus_limbs: usize) -> Option<NonZeroUsize> {
        let full = Multiplication::structural_product_work(modulus_limbs, modulus_limbs);
        let cost_three = Self::factor_cost(modulus_limbs, FACTOR_THREE);
        let cost_five = Self::factor_cost(modulus_limbs, FACTOR_FIVE);
        if full <= cost_three.min(cost_five) {
            return None;
        }
        Some(if cost_three <= cost_five {
            FACTOR_THREE
        } else {
            FACTOR_FIVE
        })
    }

    /// Lower-tower product work plus the factorization's linear folding traffic.
    fn factor_cost(modulus_limbs: usize, factor: NonZeroUsize) -> usize {
        let block = modulus_limbs.div_euclid(factor.get());
        // SAFETY: factor is 3 or 5 and block = modulus / factor, so block
        // is strictly smaller than the positive modulus width.
        let quotient = unsafe { modulus_limbs.unchecked_sub(block) };
        Multiplication::structural_product_work(quotient, quotient)
            .saturating_add(Multiplication::structural_product_work(block, block))
            .saturating_add(modulus_limbs.saturating_mul(4))
    }

    /// Compares a guarded quotient residue with Q = X^(k-1) - X^(k-2) + ... - X + 1.
    ///
    /// # Safety
    /// `value` contains exactly this plan's `quotient_coeff_len` initialized
    /// limbs, including the single guard above its `factor-1` complete blocks.
    pub unsafe fn compare_with_quotient_modulus(&self, value: &[Limb]) -> Ordering {
        debug_assert_eq!(
            value.len(),
            self.quotient_coeff_len,
            "quotient comparison retains every data block and its guard"
        );
        // Q<X^(k-1), so its guard is zero. The remaining block ordinals are
        // known from construction; no division recovers them from offsets.
        // SAFETY: the complete coefficient retains its guard at quotient_len.
        if unsafe { *value.get_unchecked(self.quotient_len) } != 0 {
            return Ordering::Greater;
        }
        // SAFETY: for_factor admits k=3 or 5, so the data block count is positive.
        let data_blocks = unsafe { self.factor.get().unchecked_sub(1) };
        let mut end = self.quotient_len;
        for block in (1..data_blocks).rev() {
            // SAFETY: end=(block+1)*block_len by induction, beginning at the
            // admitted quotient_len. Each subtraction stays inside the data.
            let start = unsafe { end.unchecked_sub(self.block_len.get()) };
            // SAFETY: start<=end<=quotient_len, within the complete coefficient.
            let digits = unsafe { value.get_unchecked(start..end) };
            // Q's non-low blocks contain only zero or only B-1. An unequal
            // digit determines the ordering without a general limb comparison.
            if block.is_multiple_of(2) {
                if digits.iter().rev().any(|&limb| limb != 0) {
                    return Ordering::Greater;
                }
            } else if digits.iter().rev().any(|&limb| limb != Limb::MAX) {
                return Ordering::Less;
            }
            end = start;
        }
        // The remaining low block is [1,0,...,0]. Treat its exceptional digit
        // once, after all its more significant zero digits compare equal.
        // SAFETY: after the factor-2 high blocks, end=block_len>=1; the slice
        // is the complete low block except its initialized least-significant digit.
        if unsafe { value.get_unchecked(1..end) }
            .iter()
            .rev()
            .any(|&limb| limb != 0)
        {
            return Ordering::Greater;
        }
        // SAFETY: positive block_len guarantees the initialized low digit.
        unsafe { value.get_unchecked(0) }.cmp(&1)
    }
}
