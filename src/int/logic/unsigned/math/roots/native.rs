//! Native integer roots for single- and double-limb inputs.

#![expect(
    unsafe_code,
    reason = "native root admission proves materialized limbs, positive divisors, and descending Newton updates"
)]

use core::{
    num::{NonZero, NonZeroU32, NonZeroU64},
    ops::Div,
};

use super::{DoubleLimb, EXP2_LOWER, InternalMpUint, LOG2_LOWER, Limb, NthRootScratch, Roots};

impl NthRootScratch {
    /// Computes a two-limb root with descending native Newton iteration.
    ///
    /// Callers supply two normalized limbs, `n>=2`, the exact bit count, and
    /// `ceil(bits/n)>=ceil(log2(n))+3`. This admission gives `R>=4n` for the
    /// real root R, and `bits<=128` implies `n<=18`.
    /// Let Q be the fixed-point logarithm obtained from the leading nine bits.
    /// Mantissa truncation loses less than 2/256 and the logarithm table less
    /// than 1/256, hence `log2(a)<(Q+3)/256`. Rounding `(Q+3)/n` upwards and
    /// rounding the exponential table upwards gives an upper root estimate.
    /// Each descending Newton update retains an estimate at least the floor
    /// root. The first nondecreasing update certifies that exact floor root.
    ///
    /// The initial estimate E is below `(65/64)*R+1`: rounding the logarithm
    /// contributes less than `(3/n+1)/256`, the exponential table less than
    /// a factor 257/256, and integer rounding less than one. Thus
    /// `E/R<1+1/64+1/(4n)`. For `m=n-1<=17`, the binomial series is below
    /// `1/(1-m*(1/64+1/(4n)))<64/31<3<R`. Consequently `E^(n-1)<R^n=a`.
    /// All later estimates decrease, so every denominator power fits the
    /// native input width and needs no overflow test.
    pub fn nth_root_double_limb<const CACHE_POWER: bool>(
        &mut self,
        a: &InternalMpUint,
        n: u32,
        bits: usize,
    ) -> InternalMpUint {
        debug_assert_eq!(a.limbs().len(), 2, "double-limb root domain");
        debug_assert!(n >= 2, "positive root degree");
        // SAFETY: two normalized limbs supply both initialized halves. Their
        // concatenation fills DoubleLimb on 16-, 32-, and 64-bit targets.
        // n>=2 fits its at-least-32-bit width and defines a positive divisor.
        let (value, degree, mask) = unsafe {
            (
                DoubleLimb::try_from(*a.limbs().get_unchecked(0)).unwrap_unchecked()
                    | (DoubleLimb::try_from(*a.limbs().get_unchecked(1)).unwrap_unchecked()
                        << Limb::BITS),
                NonZero::<DoubleLimb>::new_unchecked(DoubleLimb::from(n)),
                DoubleLimb::try_from(Limb::MAX).unwrap_unchecked(),
            )
        };
        // SAFETY: Limb::BITS<bits<=2*Limb::BITS<=128 bounds the mantissa
        // shift and the eight-bit index. Q+3<=32770 fits u32 on every target.
        let logarithm = unsafe {
            let shift = u32::try_from(bits.unchecked_sub(9)).unwrap_unchecked();
            let index = usize::try_from(value.unchecked_shr(shift) & 255).unwrap_unchecked();
            u32::try_from(bits)
                .unwrap_unchecked()
                .unchecked_sub(1)
                .unchecked_mul(256)
                .unchecked_add(u32::from(*LOG2_LOWER.get_unchecked(index)))
                .unchecked_add(3)
        };
        let root_logarithm = logarithm.div_ceil(n);
        // SAFETY: the low eight bits index the table. Its rounded upper
        // mantissa is in 257..=512 and fits even a 16-bit Limb.
        let mantissa = unsafe {
            let index = usize::try_from(root_logarithm & 255).unwrap_unchecked();
            DoubleLimb::from(257_u32.unchecked_add(u32::from(*EXP2_LOWER.get_unchecked(index))))
        };
        let exponent_bits = root_logarithm >> 8;
        let mut estimate = if exponent_bits >= 8 {
            // SAFETY: n>=2 gives exponent_bits<=Limb::BITS. Multiplying
            // the nine-bit mantissa by 2^(exponent_bits-8) fits DoubleLimb.
            unsafe { mantissa.unchecked_shl(exponent_bits.unchecked_sub(8)) }
        } else {
            // SAFETY: 0<=exponent_bits<8 gives a shift in 1..=8. The sum
            // is at most 767, so rounding upwards is exact on every target.
            unsafe {
                let shift = 8_u32.unchecked_sub(exponent_bits);
                let rounding = 1_u32.unchecked_shl(shift).unchecked_sub(1);
                mantissa
                    .unchecked_add(DoubleLimb::from(rounding))
                    .unchecked_shr(shift)
            }
        };
        // SAFETY: n>=2 proves the exponent n-1 is positive.
        let exponent = unsafe { NonZeroU32::new_unchecked(n.unchecked_sub(1)) };
        let power_bits = exponent.ilog2();
        // SAFETY: the highest exponent bit is in 0..32 and its value is at
        // most the exponent. Removing it retains only the lower set bits.
        let lower_exponent = unsafe {
            exponent
                .get()
                .unchecked_sub(1_u32.unchecked_shl(power_bits))
        };
        loop {
            // The leading bit initializes the accumulator. Each remaining
            // set bit contributes one product; intervening zeros contribute
            // only squares. Every binary prefix remains at most n-1.
            let mut power = estimate;
            let mut remaining = lower_exponent;
            let mut previous_bit = power_bits;
            while let Some(bits_left) = NonZeroU32::new(remaining) {
                let bit = bits_left.ilog2();
                // SAFETY: the next set bit is strictly below previous_bit.
                // Its first square and all later prefixes are at most n-1,
                // whose admitted estimate power is below DoubleLimb::MAX.
                power = unsafe { power.unchecked_mul(power) };
                // SAFETY: previous_bit>bit makes this difference positive.
                let squares = unsafe { previous_bit.unchecked_sub(bit).unchecked_sub(1) };
                for _ in 0..squares {
                    // SAFETY: every square remains within that binary prefix.
                    power = unsafe { power.unchecked_mul(power) };
                }
                // SAFETY: adding the next set bit retains a prefix at most
                // n-1. Its highest bit is in 0..32 and can be removed exactly.
                unsafe {
                    power = power.unchecked_mul(estimate);
                    remaining = bits_left.get().unchecked_sub(1_u32.unchecked_shl(bit));
                }
                previous_bit = bit;
            }
            for _ in 0..previous_bit {
                // SAFETY: trailing zero bits increase the final prefix to
                // n-1, whose admitted power remains below the input value.
                power = unsafe { power.unchecked_mul(power) };
            }
            // SAFETY: estimate>=floor(root(value))>=1 keeps every power positive.
            let native_denominator = unsafe { NonZero::<DoubleLimb>::new_unchecked(power) };
            let quotient = Div::div(value, native_denominator);
            if quotient >= estimate {
                if CACHE_POWER {
                    // SAFETY: the certified floor root proves estimate*power<=value.
                    // Masking and shifting each double-limb power gives two
                    // native limbs without truncation on any pointer width.
                    let (denominator, product) = unsafe {
                        let product = estimate.unchecked_mul(power);
                        (
                            InternalMpUint::from_limbs_2(
                                Limb::try_from(power & mask).unwrap_unchecked(),
                                Limb::try_from(power >> Limb::BITS).unwrap_unchecked(),
                            ),
                            InternalMpUint::from_limbs_2(
                                Limb::try_from(product & mask).unwrap_unchecked(),
                                Limb::try_from(product >> Limb::BITS).unwrap_unchecked(),
                            ),
                        )
                    };
                    self.x_pow_n_minus_1.clone_from(&denominator);
                    self.temp_prod.clone_from(&product);
                }
                // SAFETY: for n>=2, root(value)<B since value<B^2, proving
                // the certified root fits one native limb on every target.
                return InternalMpUint::from_limb(unsafe {
                    Limb::try_from(estimate).unwrap_unchecked()
                });
            }
            // SAFETY: quotient<estimate makes the difference positive.
            // Descending Newton preserves the floor root, which is positive;
            // ceil((estimate-quotient)/n) is therefore at most estimate-1.
            unsafe {
                let difference = estimate.unchecked_sub(quotient);
                estimate = estimate.unchecked_sub(difference.div_ceil(degree.get()));
            }
        }
    }
}

impl Roots {
    /// Computes a single-limb root with native integer arithmetic.
    ///
    /// Callers must pass a one-limb value and `n >= 2`.
    pub fn nth_root_single_limb(a: &InternalMpUint, n: u32) -> InternalMpUint {
        let limbs = a.limbs();
        debug_assert_eq!(limbs.len(), 1, "the native root path requires one limb");
        debug_assert!(n >= 2, "the native root path requires degree at least two");
        // SAFETY: dispatch supplies n>=2; widening a positive u32 preserves
        // its nonzero value independently of the target pointer width.
        let native_degree = unsafe { NonZeroU32::new_unchecked(n) };
        let denominator = NonZeroU64::from(native_degree);
        // SAFETY: the caller guarantees one initialized limb.
        let val = unsafe { *limbs.get_unchecked(0) };
        if val <= 1 {
            return a.clone();
        }
        // SAFETY: Limb is usize (16/32/64 bits), so every value fits u64.
        let val_u64 = unsafe { u64::try_from(val).unwrap_unchecked() };
        // SAFETY: val>=2 bounds leading_zeros below 64.
        let bits = unsafe { u64::BITS.unchecked_sub(val_u64.leading_zeros()) };
        if n >= bits {
            return InternalMpUint::one();
        }
        // n >= 2 and bits <= 64 imply ceil(bits/n) <= 32.
        let mut estimate = 1_u64 << bits.div_ceil(native_degree.get());
        // SAFETY: n >= 2 bounds the positive exponent n-1.
        let exponent = unsafe { n.unchecked_sub(1) };
        loop {
            // An overflowing denominator exceeds a, giving exact quotient zero.
            let quotient = estimate.checked_pow(exponent).map_or(0, |power| {
                // SAFETY: estimate >= 1 throughout descending Newton, so
                // every representable positive power is also nonzero.
                Div::div(val_u64, unsafe { NonZeroU64::new_unchecked(power) })
            });
            if quotient >= estimate {
                // SAFETY: the floor root is bounded by val <= Limb::MAX.
                return InternalMpUint::from_limb(unsafe {
                    Limb::try_from(estimate).unwrap_unchecked()
                });
            }
            // SAFETY: quotient < estimate makes the difference positive.
            // n>=2 gives ceil(difference/n)<=estimate-1 whenever estimate>=2;
            // estimate=1 already returned because val>=2 gives quotient>=1.
            unsafe {
                let difference = estimate.unchecked_sub(quotient);
                estimate = estimate.unchecked_sub(difference.div_ceil(denominator.get()));
            }
        }
    }
}
