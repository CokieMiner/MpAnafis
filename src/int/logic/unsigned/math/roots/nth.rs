//! Integer roots by precision growth and bounded power comparisons.
//!
//! Reference: R. P. Brent and P. Zimmermann, "Modern Computer Arithmetic",
//! Cambridge University Press, 2011, Section 1.5.2 (integer kth roots).

#![expect(
    unsafe_code,
    reason = "root precision bounds prove native divisors, shifted spans, and Newton differences"
)]

use core::{
    cmp::Ordering,
    mem::swap,
    num::{NonZero, NonZeroU32, NonZeroUsize},
    ops::Div,
    ptr::copy_nonoverlapping,
};

use super::{
    ArchKernels, Division, DoubleLimb, EXP2_LOWER, InternalMpUint, LIMB_BITS, LOG2_LOWER, Limb,
    NthRootScratch, Roots,
};

impl NthRootScratch {
    /// Computes a general multi-limb `n`th root.
    ///
    /// Callers pass `n >= 2`, a value that is neither zero nor one, and its
    /// exact validated significant bit count. Recursive stages derive that
    /// count by subtraction rather than repeating representation validation.
    /// `CACHE_POWER` retains the returned root raised to n-1 and n for its parent.
    pub fn nth_root_multi_limb<const CACHE_POWER: bool>(
        &mut self,
        a: &InternalMpUint,
        n: u32,
        bits: usize,
    ) -> InternalMpUint {
        debug_assert!(
            n >= 2 && !a.is_zero() && !a.is_one(),
            "nontrivial root domain"
        );
        let Ok(native_value) = Limb::try_from(n) else {
            // An unrepresentable degree exceeds the validated usize bit count.
            if CACHE_POWER {
                self.x_pow_n_minus_1.clone_from(&InternalMpUint::one());
                self.temp_prod.clone_from(&InternalMpUint::one());
            }
            return InternalMpUint::one();
        };
        // SAFETY: the dispatch boundary supplies n >= 2, and conversion
        // preserves its positive value on every supported pointer width.
        let degree = unsafe { NonZeroUsize::new_unchecked(native_value) };
        if degree.get() >= bits {
            if CACHE_POWER {
                self.x_pow_n_minus_1.clone_from(&InternalMpUint::one());
                self.temp_prod.clone_from(&InternalMpUint::one());
            }
            return InternalMpUint::one();
        }
        let root_bits = bits.div_ceil(degree.get());
        // SAFETY: n >= 2 gives n-1 >= 1 and 1 <= ceil(log2(n)) <= 32.
        // This count and its successor fit even a 16-bit usize.
        let log_degree = unsafe {
            usize::try_from(u32::BITS.unchecked_sub(n.unchecked_sub(1).leading_zeros()))
                .unwrap_unchecked()
        };
        // SAFETY: log_degree <= 32, so adding one cannot overflow usize.
        let precision = root_bits.saturating_sub(unsafe { log_degree.unchecked_add(1) });
        if precision < 2 {
            // Here L<=ceil(log2(n))+2. If L>=LIMB_BITS>=16, then
            // n>2^(L-3) and bits>n*(L-1)>usize::MAX, contradicting the
            // validated bit count. The exact bracket [2^(L-1),2^L) fits one limb.
            return self.nth_root_bisect::<CACHE_POWER>(a, n, root_bits, bits);
        }
        if a.limbs().len() == 2 {
            // precision>=2 gives L>=ceil(log2(n))+3 and a real root R>=4n.
            // Native iteration needs neither a prefix nor virtual guard bits.
            return self.nth_root_double_limb::<CACHE_POWER>(a, n, bits);
        }

        // With beta=2^s and a prefix root q, this schedule initially gives q>=n*beta.
        // A root-only stage may retain g guard bits by treating a as a*2^(ng).
        // Taking 3g<=L-ceil(log2(n))-1 guarantees s>=2g, so every numerator
        // suffix still comes from a right shift of the original input.
        let guard = if CACHE_POWER {
            0
        } else {
            precision.div_euclid(3).min(LIMB_BITS)
        };
        // SAFETY: guard<=precision/3<=root_bits/3 and n>=2 bound this sum
        // below bits: guard>0 implies root_bits>=6 and bits>n*(root_bits-1).
        let available = unsafe { precision.unchecked_add(guard) };
        let mut root_shift = available >> 1;
        if available & 1 != 0 && root_shift < LIMB_BITS - 1 {
            // For k=ceil(log2(n)), the candidate prefix minimum is 2^(s+k).
            // If available were even, it would be 2^(s+k-1), which cannot
            // reach (n-1)*(2^s+1) because n-1>=2^(k-1). In the odd case,
            // admission is equivalent to 2^k-n+1 > floor((n-2)/2^s).
            // SAFETY: 1<=k<=Limb::BITS makes the mask shift valid even when
            // 2^k itself exceeds Limb::MAX. mask>=n-1 and the gap is at most
            // 2^(k-1), so both arithmetic operations fit. n>=2 and
            // 0<s<Limb::BITS-1 prove the nonnegative difference and shift.
            let admitted = unsafe {
                let mask = Limb::MAX.unchecked_shr(
                    u32::try_from(LIMB_BITS.unchecked_sub(log_degree)).unwrap_unchecked(),
                );
                let gap = mask
                    .unchecked_sub(degree.get().unchecked_sub(1))
                    .unchecked_add(1);
                gap > degree
                    .get()
                    .unchecked_sub(2)
                    .unchecked_shr(u32::try_from(root_shift).unwrap_unchecked())
            };
            if admitted {
                // SAFETY: the original schedule bounds s+1<=root_bits-1;
                // its admitted successor preserves s>=2g and shift<bits.
                root_shift = unsafe { root_shift.unchecked_add(1) };
            }
        }
        // precision>=2 proves available>=2 and root_shift>=1.
        self.nth_root_lift::<CACHE_POWER>(a, n, degree, (root_shift, guard), bits)
    }

    /// Extends a certified prefix root by `s` bits and corrects at most once.
    ///
    /// Let `beta=2^s`, `q=floor(root_n(floor(a/beta^n)))` and `S=q*beta`.
    /// The schedule guarantees q>=(n-1)*(beta/2+1). For delta=root_n(a)-S<beta,
    /// the tangent quotient T=(a-S^n)/(n*S^(n-1)) satisfies delta<=T<delta+1:
    /// its excess is below (n-1)*beta/(2q)*(1+1/q)^(n-2). Bounding the
    /// binomial coefficients by powers of n-2 gives the geometric bound
    /// (n-1)*beta/(2*(q-n+2))<1 under the admitted prefix condition.
    /// Thus S+min(floor(T),beta-1) is the floor root or its successor.
    ///
    /// With g>0, the same proof applies to the virtual input a*2^(ng).
    /// Truncating the approximate root by g bits can cross an integer boundary
    /// only when its low g bits are zero. Any nonzero guard certifies the root
    /// without evaluating its final power; a zero guard retains exact repair.
    fn nth_root_lift<const CACHE_POWER: bool>(
        &mut self,
        a: &InternalMpUint,
        n: u32,
        degree: NonZeroUsize,
        precision: (usize, usize),
        bits: usize,
    ) -> InternalMpUint {
        let (s, guard) = precision;
        // SAFETY: the initial schedule gives s>=2g and s<=root_bits-1;
        // its admitted successor preserves both bounds. Thus
        // n*s<=n*(root_bits-1)<bits, proving all
        // products fit usize; (n-1)*s>=n*g makes the omitted count nonnegative.
        let (shift, omitted, exponent) = unsafe {
            (
                s.unchecked_sub(guard).unchecked_mul(degree.get()),
                s.unchecked_mul(degree.get().unchecked_sub(1))
                    .unchecked_sub(guard.unchecked_mul(degree.get())),
                NonZeroU32::new_unchecked(n.unchecked_sub(1)),
            )
        };
        let head = a.shr(shift);
        // SAFETY: the precision schedule proves shift<bits; shifting a
        // normalized bits-bit value retains exactly bits-shift significant bits.
        let head_bits = unsafe { bits.unchecked_sub(shift) };
        let mut root = if head.limbs().len() == 1 {
            let root = Roots::nth_root_single_limb(&head, n);
            self.root_powers(&root, exponent);
            root
        } else {
            self.nth_root_multi_limb::<true>(&head, n, head_bits)
        };

        // The child retains both q^(n-1) and q^n from its certification.
        // A native leaf initializes them once; every parent reuses both.
        self.temp_prod.shl_assign(s);

        // floor(T) = floor(floor((a >> ((n-1)s))-q^n*beta)/q^(n-1))/n).
        // Shifting the numerator avoids expanding the denominator by (n-1)s
        // bits. Dividing the short quotient by n avoids multiplying the
        // longer denominator by n.
        let word_shift = omitted >> LIMB_BITS.trailing_zeros();
        let bit_shift = omitted & (LIMB_BITS - 1);
        // SAFETY: omitted<bits, so the retained initialized suffix is nonempty.
        let source = unsafe { a.limbs().get_unchecked(word_shift..) };
        let mut pending = self.difference.prepare_limb_write(source.len());
        if bit_shift == 0 {
            // SAFETY: the disjoint destination reserves source.len() aligned
            // limbs, all of which the initialized source copy fills.
            unsafe {
                copy_nonoverlapping(source.as_ptr(), pending.as_mut_ptr(), source.len());
            }
        } else {
            // SAFETY: bit_shift is in 1..Limb::BITS and fits u32 on all
            // pointer widths. The nonempty, initialized source and reserved
            // output are aligned and disjoint; the kernel fills every slot.
            unsafe {
                let _ = ArchKernels::rshift_into_unchecked(
                    pending.as_mut_ptr(),
                    source.as_ptr(),
                    source.len(),
                    u32::try_from(bit_shift).unwrap_unchecked(),
                );
            }
        }
        // SAFETY: the copy or shift initialized the complete prepared span.
        let _ = unsafe { pending.commit() };
        // SAFETY: shifting a nonzero bits-bit value by omitted<bits gives
        // exactly ceil((bits-omitted)/LIMB_BITS) initialized, normalized limbs.
        unsafe {
            self.difference
                .set_len(bits.unchecked_sub(omitted).div_ceil(LIMB_BITS));
        }
        self.difference.sub_assign(&self.temp_prod);
        Division::div_into::<true, false>(
            &self.difference,
            &self.x_pow_n_minus_1,
            &mut self.quotient,
            &mut self.div_scratch,
        );
        let _ =
            Division::div_rem_1::<true>(self.quotient.limbs(), degree.get(), &mut self.correction);
        if self.correction.get_bit(s) {
            // floor(T)<=beta, and the true low part is below beta.
            self.correction.decrement();
        }
        root.shl_assign(s);
        root.add_assign(&self.correction);

        if guard > 0 {
            // SAFETY: 1<=guard<=LIMB_BITS bounds the mask shift below
            // LIMB_BITS. The admitted prefix root is positive, proving that
            // the lifted root has an initialized low limb.
            let certified = unsafe {
                *root.limbs().get_unchecked(0) & (Limb::MAX >> LIMB_BITS.unchecked_sub(guard)) != 0
            };
            root.shr_assign(guard);
            if certified {
                return root;
            }
        }
        self.root_powers(&root, exponent);
        if self.temp_prod.cmp(a) == Ordering::Greater {
            root.decrement();
            if CACHE_POWER {
                self.root_powers(&root, exponent);
            }
        }
        root
    }

    /// Certifies a one-limb root bracket using bounded positive powers.
    ///
    /// Callers supply `n>=2`, the exact positive root width, and
    /// `root_bits<Limb::BITS` with the exact input bit count. Two-limb inputs
    /// use a logarithmic lower estimate and one primitive trial power; overflow
    /// proves rejection. Wider inputs retain multi-precision bisection.
    fn nth_root_bisect<const CACHE_POWER: bool>(
        &mut self,
        a: &InternalMpUint,
        n: u32,
        root_bits: usize,
        bits: usize,
    ) -> InternalMpUint {
        // SAFETY: the leaf schedule proves 1<=root_bits<Limb::BITS, and
        // dispatch supplies n>=2. Both bracket endpoints therefore fit Limb.
        let (mut lower, mut upper, power) = unsafe {
            let lower =
                1_usize.unchecked_shl(u32::try_from(root_bits.unchecked_sub(1)).unwrap_unchecked());
            (lower, lower.unchecked_mul(2), NonZeroU32::new_unchecked(n))
        };
        if let &[low, high] = a.limbs() {
            // SAFETY: each native limb fits DoubleLimb, and the high shift
            // fills its upper half on every supported pointer width.
            let limit = unsafe {
                DoubleLimb::try_from(low).unwrap_unchecked()
                    | (DoubleLimb::try_from(high).unwrap_unchecked() << Limb::BITS)
            };
            lower = native_lower(limit, power, bits, root_bits);
            // SAFETY: the seed is below 2^root_bits<=128, so its successor
            // fits every supported Limb, and widening to DoubleLimb is exact.
            let (next, candidate) = unsafe {
                let next = lower.unchecked_add(1);
                (next, DoubleLimb::try_from(next).unwrap_unchecked())
            };
            let lower_power = if let Some(value) = candidate.checked_pow(power.get())
                && value <= limit
            {
                lower = next;
                value
            } else if CACHE_POWER {
                // The lower estimate never exceeds the real root; its power
                // is bounded by limit, so primitive exponentiation cannot overflow.
                // SAFETY: lower is a native limb and widens infallibly.
                unsafe { DoubleLimb::try_from(lower).unwrap_unchecked() }.pow(power.get())
            } else {
                0
            };
            if CACHE_POWER {
                // The retained value is root^n. Exact primitive division by
                // the positive native root gives root^(n-1), replacing a
                // second multi-precision exponentiation and its final product.
                // SAFETY: lower>=1 and fits Limb, hence widens infallibly;
                // masking and high-half extraction produce native limbs.
                let (denominator, product) = unsafe {
                    let divisor = NonZero::<DoubleLimb>::new_unchecked(
                        DoubleLimb::try_from(lower).unwrap_unchecked(),
                    );
                    let denominator_power = Div::div(lower_power, divisor);
                    let mask = DoubleLimb::try_from(Limb::MAX).unwrap_unchecked();
                    (
                        InternalMpUint::from_limbs_2(
                            Limb::try_from(denominator_power & mask).unwrap_unchecked(),
                            Limb::try_from(denominator_power >> Limb::BITS).unwrap_unchecked(),
                        ),
                        InternalMpUint::from_limbs_2(
                            Limb::try_from(lower_power & mask).unwrap_unchecked(),
                            Limb::try_from(lower_power >> Limb::BITS).unwrap_unchecked(),
                        ),
                    )
                };
                self.x_pow_n_minus_1.clone_from(&denominator);
                self.temp_prod.clone_from(&product);
            }
        } else {
            let power_bits = power.ilog2();
            // SAFETY: the exact bracket starts ordered; each update retains
            // its strictly interior midpoint, preserving upper>lower.
            while unsafe { upper.unchecked_sub(lower) } > 1 {
                // SAFETY: lower<upper<Limb::MAX bounds this exact midpoint.
                let middle = unsafe { lower.unchecked_add(upper.unchecked_sub(lower) >> 1) };
                let candidate = InternalMpUint::from_limb(middle);
                if self.bounded_power(&candidate, power, power_bits, a) {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
            if CACHE_POWER {
                // SAFETY: dispatch supplies n>=2, so n-1 is positive.
                let exponent = unsafe { NonZeroU32::new_unchecked(n.unchecked_sub(1)) };
                self.root_powers(&InternalMpUint::from_limb(lower), exponent);
            }
        }
        InternalMpUint::from_limb(lower)
    }

    /// Computes the root denominator and its next power for certification.
    /// Every intermediate power is bounded by the final root-stage power.
    fn root_powers(&mut self, base: &InternalMpUint, exponent: NonZeroU32) {
        debug_assert!(!base.is_zero(), "positive root power");
        self.x_pow_n_minus_1.clone_from(base);
        for bit in (0..exponent.ilog2()).rev() {
            self.temp_prod
                .assign_square_with_scratch(&self.x_pow_n_minus_1, &mut self.mul_scratch);
            swap(&mut self.x_pow_n_minus_1, &mut self.temp_prod);
            if (exponent.get() >> bit) & 1 != 0 {
                self.temp_prod.assign_product_with_scratch(
                    &self.x_pow_n_minus_1,
                    base,
                    &mut self.mul_scratch,
                );
                swap(&mut self.x_pow_n_minus_1, &mut self.temp_prod);
            }
        }
        self.temp_prod.assign_product_with_scratch(
            &self.x_pow_n_minus_1,
            base,
            &mut self.mul_scratch,
        );
    }

    /// Compares a positive power with the input, stopping as soon as it exceeds
    /// the limit. Intermediate exponents increase, so later products cannot fit.
    fn bounded_power(
        &mut self,
        base: &InternalMpUint,
        exponent: NonZeroU32,
        exponent_bits: u32,
        limit: &InternalMpUint,
    ) -> bool {
        self.x_pow_n_minus_1.clone_from(base);
        if exponent_bits == 0 {
            return base.cmp(limit) != Ordering::Greater;
        }
        for bit in (0..exponent_bits).rev() {
            self.temp_prod
                .assign_square_with_scratch(&self.x_pow_n_minus_1, &mut self.mul_scratch);
            if self.temp_prod.cmp(limit) == Ordering::Greater {
                return false;
            }
            swap(&mut self.x_pow_n_minus_1, &mut self.temp_prod);
            if (exponent.get() >> bit) & 1 == 1 {
                self.temp_prod.assign_product_with_scratch(
                    &self.x_pow_n_minus_1,
                    base,
                    &mut self.mul_scratch,
                );
                if self.temp_prod.cmp(limit) == Ordering::Greater {
                    return false;
                }
                swap(&mut self.x_pow_n_minus_1, &mut self.temp_prod);
            }
        }
        true
    }
}

/// Returns a lower estimate at most one below the integer root.
///
/// The leaf schedule gives `L<=ceil(log2(n))+2`, hence `root/n<8`.
/// A two-limb input has at most 128 bits. `L>=8` would require
/// `n>=33` and `bits>7*n>=231`, so `2<=L<=7` and the root is below 128.
/// Truncating the normalized input to nine bits and its logarithm to
/// eight fractional bits loses less than `3/256` in `log2(input)`.
/// Dividing by n loses another `1/256` in the root logarithm.
/// Before integer truncation, the combined input and table errors are
/// below `3/32+3*2^(L-1)/256<=27/32<1`. The integer root is therefore
/// the returned lower estimate or its successor.
fn native_lower(input: DoubleLimb, degree: NonZeroU32, bits: usize, root_bits: usize) -> Limb {
    // SAFETY: the two-limb admission proves 17<=bits<=128; the top
    // nine bits have an implicit one, leaving an eight-bit table index.
    // The full fixed-point logarithm is at most 32767 and fits u32.
    let logarithm = unsafe {
        let shift = u32::try_from(bits.unchecked_sub(9)).unwrap_unchecked();
        let index = usize::try_from(input.unchecked_shr(shift) & 255).unwrap_unchecked();
        u32::try_from(bits)
            .unwrap_unchecked()
            .unchecked_sub(1)
            .unchecked_mul(256)
            .unchecked_add(u32::from(*LOG2_LOWER.get_unchecked(index)))
    };
    let root_logarithm = Div::div(logarithm, degree);
    // SAFETY: masking retains an eight-bit index on all pointer widths.
    // The leaf proof gives 2<=root_bits<=7, so 9-root_bits is a valid
    // native right shift. The table mantissa is in 256..=511.
    unsafe {
        let index = usize::try_from(root_logarithm & 255).unwrap_unchecked();
        let mantissa = 256_usize.unchecked_add(usize::from(*EXP2_LOWER.get_unchecked(index)));
        mantissa.unchecked_shr(u32::try_from(9_usize.unchecked_sub(root_bits)).unwrap_unchecked())
    }
}
