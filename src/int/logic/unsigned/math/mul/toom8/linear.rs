//! Fixed-width linear combinations and exact divisions for Toom-8 interpolation.
//!
//! Matrix scalars and divisors are compile-time parameters. Their nonzero and
//! word-width properties therefore follow from the matrix rather than a runtime
//! assumption. A nonzero word divisor retains that invariant in the leaf type.
//! The largest scalar is `15_181_425` and the largest divisor is `48_070_897_875`.
//! Fixed word factorizations retain exact signed division and modular scalar
//! products on 16-bit limbs without runtime trial division. Factored divisions
//! propagate each factor's borrow in one pass, storing only the final quotient.

#![expect(
    unsafe_code,
    reason = "Fixed matrix constants and equal initialized packed widths bound scalar carries, factored quotients, and temporary spans"
)]

use core::num::NonZeroUsize;

use super::{AddMulKernel, ArchKernels, Limb, SharedEval, Toom8};

// Rows correspond to the two wide divisors 48_070_897_875 and 46_591_793_325.
// Their positive odd word factors and B-adic inverses are established once.
#[cfg(target_pointer_width = "16")]
const FIRST_FACTORS: [NonZeroUsize; 2] = [
    NonZeroUsize::new(56_133).unwrap(),
    NonZeroUsize::new(45_927).unwrap(),
];
#[cfg(not(target_pointer_width = "16"))]
const FIRST_FACTORS: [NonZeroUsize; 2] = [
    NonZeroUsize::new(1_550_674_125).unwrap(),
    NonZeroUsize::new(1_502_961_075).unwrap(),
];
const FIRST_INVERSES: [Limb; 2] = [
    SharedEval::invert_odd(FIRST_FACTORS[0].get()),
    SharedEval::invert_odd(FIRST_FACTORS[1].get()),
];
#[cfg(target_pointer_width = "16")]
const SECOND_FACTORS: [NonZeroUsize; 2] = [
    NonZeroUsize::new(27_625).unwrap(),
    NonZeroUsize::new(32_725).unwrap(),
];
#[cfg(target_pointer_width = "16")]
const SECOND_INVERSES: [Limb; 2] = [
    SharedEval::invert_odd(SECOND_FACTORS[0].get()),
    SharedEval::invert_odd(SECOND_FACTORS[1].get()),
];

impl Toom8 {
    /// Adds `scalar*src` modulo the common packed interpolation width.
    /// Every matrix row has destination coefficient one and `0<|scalar|<2^24`.
    /// `SUBTRACT` selects the negative coefficient at compile time.
    /// The five scalars above a 16-bit word have the factorizations below.
    #[cfg_attr(
        not(target_pointer_width = "16"),
        expect(
            clippy::needless_pass_by_ref_mut,
            reason = "The common matrix interface requires writable factorization scratch only for 16-bit limbs"
        )
    )]
    pub fn add_mul_signed_in_place<const SCALAR: u32, const SUBTRACT: bool>(
        dst: &mut [Limb],
        src: &[Limb],
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        debug_assert_eq!(dst.len(), src.len(), "linear-combination widths differ");
        debug_assert_eq!(dst.len(), temporary.len(), "temporary width differs");
        // Only 16-bit limbs can be narrower than a fixed matrix scalar.
        #[cfg(target_pointer_width = "16")]
        let (word_source, src_word) = if SCALAR > u32::from(u16::MAX) {
            // These are all five wide matrix scalars. Each product is exact
            // and each factor fits one limb. Two word factors are required
            // because the complete scalar exceeds one limb; the first pass
            // overwrites temporary and the second adds/subtracts directly in dst.
            let [first, second]: [Limb; 2] = match SCALAR {
                1_052_688 => [4_368, 241],
                12_567_555 => [2_835, 4_433],
                524_288 => [1_024, 512],
                112_896 => [256, 441],
                _ => {
                    debug_assert_eq!(SCALAR, 15_181_425, "unknown wide matrix scalar");
                    [18_225, 833]
                }
            };
            // SAFETY: the matrix reserves a disjoint initialized temporary of
            // src.len() limbs. Restricting that view makes the overwrite widths
            // identical without copying or reading the old temporary contents.
            let active = unsafe { temporary.get_unchecked_mut(..src.len()) };
            let mut carry = 0;
            for (slot, source) in active.iter_mut().zip(src) {
                let (low, high) = ArchKernels::mul_limb_lo_hi(*source, first);
                let (with_carry, overflow) = low.overflowing_add(carry);
                *slot = with_carry;
                // SAFETY: high<=first-1, hence high plus a binary carry is
                // <=first<=Limb::MAX. Escaping carry is discarded modulo B^n.
                carry = unsafe { high.unchecked_add(Limb::from(overflow)) };
            }
            (&*active, second)
        } else {
            #[expect(
                clippy::as_conversions,
                reason = "the constant-selected 16-bit word path admits SCALAR<=u16::MAX"
            )]
            let word = SCALAR as Limb;
            (src, word)
        };

        #[cfg(not(target_pointer_width = "16"))]
        #[expect(
            clippy::as_conversions,
            reason = "the compile-time matrix bound is below 2^24 and this branch has 32- or 64-bit limbs"
        )]
        let src_word = SCALAR as Limb;
        #[cfg(not(target_pointer_width = "16"))]
        let word_source = src;
        // The fixed-width ring discards final carry/borrow: there is no suffix
        // above src to update and no destination-scalar multiplication to perform.
        if SUBTRACT {
            // SAFETY: checked layouts give dst and src identical initialized
            // packed widths. Their live borrows and matrix slots are disjoint.
            unsafe {
                let _ = ArchKernels::sub_mul_limbs_unchecked(
                    dst.as_mut_ptr(),
                    word_source.as_ptr(),
                    word_source.len(),
                    src_word,
                );
            }
        } else {
            // SAFETY: the same layout establishes initialized, disjoint spans
            // of src.len() limbs; the driver selected this process-stable kernel.
            unsafe {
                let _ = add_mul_kernel(
                    dst.as_mut_ptr(),
                    word_source.as_ptr(),
                    word_source.len(),
                    src_word,
                );
            }
        }
    }

    /// Evaluates an exact matrix row `(dst-SCALAR*src)/DIVISOR`.
    /// The fixed rows in `interpolate_values` supply only positive odd divisors.
    pub fn exact_sub_mul_u64_odd_in_place<const SCALAR: u32, const DIVISOR: u64>(
        dst: &mut [Limb],
        src: &[Limb],
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "the compile-time word test proves both narrowed matrix constants fit Limb; usize and u32 widen exactly to u64 on all targets"
        )]
        if const { SCALAR as u64 <= Limb::MAX as u64 && DIVISOR <= Limb::MAX as u64 } {
            let scalar_word = const { SCALAR as Limb };
            let divisor_word = const { DIVISOR as Limb };
            SharedEval::exact_sub_mul_word_odd_in_place(dst, src, scalar_word, divisor_word);
            return;
        }

        // Some Toom-8 constants exceed one limb on 16- and 32-bit targets. Keep
        // the portable factored path there; 64-bit targets use the fused pass.
        Self::add_mul_signed_in_place::<SCALAR, true>(dst, src, temporary, add_mul_kernel);
        Self::exact_signed_div_u64::<DIVISOR>(dst);
    }

    /// Evaluates the two-source matrix row with positive odd divisor
    /// `48_070_897_875` and scalars `1300` and `1_052_688`.
    pub fn exact_sub_mul_two_u64_odd_in_place<
        const PRIMARY_SCALAR: u32,
        const SECONDARY_SCALAR: u32,
        const DIVISOR: u64,
    >(
        dst: &mut [Limb],
        primary_src: &[Limb],
        secondary_src: &[Limb],
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "the compile-time word test bounds all narrowed matrix constants; usize and u32 widen exactly to u64 on all targets"
        )]
        if const {
            PRIMARY_SCALAR as u64 <= Limb::MAX as u64
                && SECONDARY_SCALAR as u64 <= Limb::MAX as u64
                && (PRIMARY_SCALAR as u64 + SECONDARY_SCALAR as u64) <= Limb::MAX as u64
                && DIVISOR <= Limb::MAX as u64
        } {
            let primary_word = const { PRIMARY_SCALAR as Limb };
            let secondary_word = const { SECONDARY_SCALAR as Limb };
            let divisor_word = const { DIVISOR as Limb };
            let inverse = const { SharedEval::invert_odd(DIVISOR as Limb) };
            debug_assert_eq!(dst.len(), primary_src.len(), "primary widths differ");
            debug_assert_eq!(dst.len(), secondary_src.len(), "secondary widths differ");
            // SAFETY: matrix rows and their two source slots have the same
            // initialized packed width. The views preserve their disjoint
            // borrows and give all three iterators the destination trip count.
            let (primary, secondary) = unsafe {
                (
                    primary_src.get_unchecked(..dst.len()),
                    secondary_src.get_unchecked(..dst.len()),
                )
            };
            let mut product_carry = 0;
            let mut division_borrow = 0;
            for ((dst_limb, primary_limb), secondary_limb) in
                dst.iter_mut().zip(primary).zip(secondary)
            {
                let (primary_low, primary_high) =
                    ArchKernels::mul_limb_lo_hi(*primary_limb, primary_word);
                let (secondary_low, secondary_high) =
                    ArchKernels::mul_limb_lo_hi(*secondary_limb, secondary_word);
                let (product_sum, sum_overflow) = primary_low.overflowing_add(secondary_low);
                let (low_with_carry, carry_overflow) = product_sum.overflowing_add(product_carry);
                let (difference, subtraction_underflow) = dst_limb.overflowing_sub(low_with_carry);
                // SAFETY: the constant word test proves s=primary+secondary
                // fits Limb. Inductively carry<=s gives T<=B*s. If high=s,
                // low=0 cannot borrow; otherwise high+borrow<=s. Each positive
                // partial sum is bounded by that complete carry.
                product_carry = unsafe {
                    primary_high
                        .unchecked_add(secondary_high)
                        .unchecked_add(Limb::from(sum_overflow))
                        .unchecked_add(Limb::from(carry_overflow))
                        .unchecked_add(Limb::from(subtraction_underflow))
                };
                let (adjusted, division_underflow) = difference.overflowing_sub(division_borrow);
                let quotient = adjusted.wrapping_mul(inverse);
                let (_, quotient_high) = ArchKernels::mul_limb_lo_hi(quotient, divisor_word);
                // SAFETY: quotient<B gives quotient_high<=divisor-1; the
                // binary borrow raises it at most to divisor<=Limb::MAX.
                division_borrow =
                    unsafe { quotient_high.unchecked_add(Limb::from(division_underflow)) };
                *dst_limb = quotient;
            }
            // Both carries beyond the packed guard are discarded sign extension.
            return;
        }

        Self::add_mul_signed_in_place::<PRIMARY_SCALAR, true>(
            dst,
            primary_src,
            temporary,
            add_mul_kernel,
        );
        Self::exact_sub_mul_u64_odd_in_place::<SECONDARY_SCALAR, DIVISOR>(
            dst,
            secondary_src,
            temporary,
            add_mul_kernel,
        );
    }

    /// Divides an exact matrix row by one of its fixed positive divisors:
    /// `9`, `1020`, `2835`, `42_525`, `48_070_897_875`, or `46_591_793_325`.
    /// Only `1020` is even; the two wide divisors use the prescribed factors.
    pub fn exact_signed_div_u64<const DIVISOR: u64>(value: &mut [Limb]) {
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "the compile-time word test proves DIVISOR fits usize; usize::MAX widens exactly to u64 on every supported target"
        )]
        if const { DIVISOR <= usize::MAX as u64 } {
            // The matrix's sole even divisor is 1020=4*255. All remaining
            // admitted word divisors are odd, so they need no identity shift.
            if const { DIVISOR == 1_020 } {
                SharedEval::exact_signed_div_power_of_two_in_place(value, 2);
                SharedEval::exact_div_radix_minus_one_in_place::<255>(value);
            } else {
                let word = const { NonZeroUsize::new(DIVISOR as usize).unwrap() };
                let inverse = const { SharedEval::invert_odd(DIVISOR as Limb) };
                SharedEval::exact_div_odd_in_place(value, word.get(), inverse);
            }
            return;
        }
        // Only these two matrix divisors exceed a 16- or 32-bit limb:
        // 48_070_897_875 = 56_133 * 27_625 * 31,
        // 46_591_793_325 = 45_927 * 32_725 * 31.
        // The products of the first two factors are 1_550_674_125 and
        // 1_502_961_075. Thus two/three word divisions are sufficient and
        // minimal on 32-/16-bit limbs; no runtime factorization is needed.
        let first = const {
            if DIVISOR == 48_070_897_875 {
                FIRST_FACTORS[0]
            } else {
                FIRST_FACTORS[1]
            }
        };
        let first_inverse = const {
            if DIVISOR == 48_070_897_875 {
                FIRST_INVERSES[0]
            } else {
                FIRST_INVERSES[1]
            }
        };
        #[cfg(target_pointer_width = "16")]
        let second = const {
            if DIVISOR == 48_070_897_875 {
                SECOND_FACTORS[0]
            } else {
                SECOND_FACTORS[1]
            }
        };
        #[cfg(target_pointer_width = "16")]
        let second_inverse = const {
            if DIVISOR == 48_070_897_875 {
                SECOND_INVERSES[0]
            } else {
                SECOND_INVERSES[1]
            }
        };
        let last = const { NonZeroUsize::new(31).unwrap() };
        let last_inverse = const { SharedEval::invert_odd(31) };
        let mut first_borrow = 0;
        #[cfg(target_pointer_width = "16")]
        let mut second_borrow = 0;
        let mut last_borrow = 0;
        for limb in value {
            let (first_adjusted, first_underflow) = limb.overflowing_sub(first_borrow);
            let first_quotient = first_adjusted.wrapping_mul(first_inverse);
            let (_, first_high) = ArchKernels::mul_limb_lo_hi(first_quotient, first.get());
            // SAFETY: q<B gives high<=first-1; its binary borrow gives <=first.
            first_borrow = unsafe { first_high.unchecked_add(Limb::from(first_underflow)) };
            #[cfg(target_pointer_width = "16")]
            let last_numerator = {
                let (second_adjusted, second_underflow) =
                    first_quotient.overflowing_sub(second_borrow);
                let second_quotient = second_adjusted.wrapping_mul(second_inverse);
                let (_, second_high) = ArchKernels::mul_limb_lo_hi(second_quotient, second.get());
                // SAFETY: q<B gives high<=second-1; its borrow gives <=second.
                second_borrow = unsafe { second_high.unchecked_add(Limb::from(second_underflow)) };
                second_quotient
            };
            #[cfg(not(target_pointer_width = "16"))]
            let last_numerator = first_quotient;
            let (adjusted, underflow) = last_numerator.overflowing_sub(last_borrow);
            let quotient = adjusted.wrapping_mul(last_inverse);
            let (_, high) = ArchKernels::mul_limb_lo_hi(quotient, last.get());
            // SAFETY: q<B gives high<=30; its borrow gives <=31 on every target.
            last_borrow = unsafe { high.unchecked_add(Limb::from(underflow)) };
            *limb = quotient;
        }
        // Each recurrence consumes the preceding quotient limb immediately;
        // uniqueness modulo B^n gives the same exact signed quotient as
        // separate passes. Intermediate quotients never need a buffer store.
    }
}
