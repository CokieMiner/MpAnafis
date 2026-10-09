//! Jacobi symbols from recursive HGCD, Lehmer batches and a binary scalar tail.
//!
//! Quotient signs follow Möller, "Efficient computation of the Jacobi symbol",
//! Algorithms 2–3: <https://arxiv.org/abs/1907.07795>. Two low bits per operand
//! suffice even when a remainder is even; no powers of two are removed from
//! the full-width operands between accepted Euclidean batches.

#![expect(
    unsafe_code,
    reason = "normalized operands and bounded sign states establish limb and table indices; scalar Jacobi divisors remain odd and positive"
)]

use core::{
    cmp::Ordering,
    mem::swap,
    num::{NonZero, NonZeroUsize},
    ops::{Div, Rem},
};

use super::{
    BINARY_EUCLID_DIVISION_SHIFT, DivScratch, Division, DoubleLimb, Gcd, HGCD_CROSSOVER_THRESHOLD,
    HgcdWorkspace, InternalMpUint, LEHMER_BRANCHLESS_THRESHOLD, Limb,
};

impl Gcd {
    /// Exhaustive six-bit-state/two-bit-quotient transition table. Its dimensions
    /// follow the residue representation, independently of hardware tuning.
    /// Bits 0–1 and 2–3 hold operand residues; bit 4 holds the sign and bit 5
    /// identifies the most recent divisor. Each transition subtracts `q*b` from
    /// `a` and exchanges the operands, leaving that divisor index zero.
    #[expect(
        clippy::indexing_slicing,
        reason = "the bounded loop initializes this table at compile time; no indexed access executes in the runtime kernel"
    )]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "each generated state contains two two-bit residues and one sign bit, hence is below 32"
    )]
    pub const JACOBI_QUOTIENT: [u8; 256] = {
        let mut table = [0; 256];
        let mut index = 0_usize;
        while index < table.len() {
            let state = index >> 2;
            let quotient = index & 3;
            let first = state & 3;
            let second = (state >> 2) & 3;
            let mut negative = (state >> 4) & 1;
            if state & 32 == 0 && first == 3 && second == 3 {
                negative ^= 1;
            }
            if second == 2 {
                // For even b = 2 (mod 4), changing the odd denominator a
                // contributes q*(a-1)/2 + q*(q-1)/2 modulo two. Computing
                // in the usize ring preserves the required low two bits.
                negative ^= quotient
                    .wrapping_mul(first.wrapping_sub(1))
                    .wrapping_add(quotient.wrapping_mul(quotient.wrapping_sub(1)))
                    .wrapping_shr(1)
                    & 1;
            }
            let remainder = first.wrapping_sub(quotient.wrapping_mul(second)) & 3;
            table[index] = (second | (remainder << 2) | (negative << 4)) as u8;
            index = index.wrapping_add(1);
        }
        table
    };
}

impl InternalMpUint {
    /// Computes the Jacobi symbol `(self | other)` for positive odd `other`.
    ///
    /// The result is zero for noncoprime operands, otherwise `1` or `-1`.
    /// A positive symbol need not imply a quadratic residue for composite moduli.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "the scalar, simulated and exact transitions share the same operand pair and transactional Jacobi sign state"
    )]
    pub fn jacobi_symbol(&self, other: &Self) -> i8 {
        debug_assert!(
            !other.is_zero() && other.is_odd(),
            "Jacobi modulus must be nonzero and odd"
        );
        if other.is_one() {
            return 1;
        }
        if self.is_zero() {
            return 0;
        }
        // SAFETY: both operands are normalized and nonzero.
        let (first_low, second_low) = unsafe {
            (
                *self.limbs().get_unchecked(0),
                *other.limbs().get_unchecked(0),
            )
        };
        if other.limbs().len() == 1 {
            if self.limbs().len() == 1 {
                return Gcd::jacobi_limb(first_low, second_low, false);
            }
            // The residue r satisfies r*B^k = -A (mod d). Every supported
            // limb width is even, so B is a square coprime to odd d. Thus
            // (A|d) = (-1|d)*(r|d); (-1|d) is negative for d = 3 (mod 4).
            let residue = Division::modexact_1_odd(self.limbs(), second_low);
            return Gcd::jacobi_limb(residue, second_low, second_low & 2 != 0);
        }
        let scalar = if self.limbs().len() == 1 {
            Some((first_low, false))
        } else {
            // SAFETY: both nonzero operands have more than one limb here.
            // Equal high suffixes make a-b exactly the signed difference of
            // their low limbs, without a full-width subtraction or copy.
            let equal_high =
                unsafe { self.limbs().get_unchecked(1..) == other.limbs().get_unchecked(1..) };
            equal_high.then_some((
                first_low.abs_diff(second_low),
                first_low < second_low && second_low & 2 != 0,
            ))
        };
        if let Some((difference, minus_sign)) = scalar {
            if difference == 0 {
                return 0;
            }
            // (a|b) = (a-b|b); a negative difference contributes (-1|b).
            // Strip powers of two and exchange the remaining odd scalar
            // using reciprocity before reducing the original denominator.
            let twos = difference.trailing_zeros();
            let denominator = difference >> twos;
            // (2|b) is negative for b = 3,5 (mod 8); exchanging two odd
            // operands changes the sign only when both are 3 (mod 4).
            let negative = minus_sign
                ^ (twos & 1 != 0 && matches!(second_low & 7, 3 | 5))
                ^ (denominator & second_low & 2 != 0);
            if denominator == 1 {
                return if negative { -1 } else { 1 };
            }
            // The exact residue represents the negated numerator times a
            // square unit. Fold (-1|denominator) into the reciprocity sign.
            let residue = Division::modexact_1_odd(other.limbs(), denominator);
            return Gcd::jacobi_limb(residue, denominator, negative ^ (denominator & 2 != 0));
        }

        if let ([low_a, high_a], [low_b, high_b]) = (self.limbs(), other.limbs()) {
            return jacobi_2([*low_a, *high_a], [*low_b, *high_b]);
        }

        let mut scratch = DivScratch::default();
        let mut a = if self.limbs().len() > other.limbs().len() {
            // The initial denominator is odd, so (a | b) = (a mod b | b)
            // without a quotient-dependent sign update. Reduce before copying
            // b or allocating recursive state; exact multiples terminate here.
            let mut remainder = Self::zero();
            Division::rem_into(self, other, &mut remainder, &mut scratch);
            if remainder.is_zero() {
                return 0;
            }
            if remainder.is_one() {
                return 1;
            }
            remainder
        } else {
            self.clone()
        };
        let mut b = other.clone();
        // SAFETY: the original zero case and zero initial remainder returned.
        let reduced_low = unsafe { *a.limbs().get_unchecked(0) };
        let mut state = (reduced_low & 3) | ((second_low & 3) << 2) | 32;
        let mut next_a = Self::zero();
        let mut next_b = Self::zero();
        let mut quotient = Self::zero();
        let operand_len = self.limbs().len().max(other.limbs().len());
        HgcdWorkspace::with_thread_local(operand_len, |workspace| {
            let branchless = a.limbs().len().max(b.limbs().len()) >= LEHMER_BRANCHLESS_THRESHOLD;
            while !b.is_zero() {
                if a.limbs().len() == 1 && b.limbs().len() == 1 {
                    // SAFETY: the length checks prove both low limbs exist.
                    let (mut numerator, mut denominator) =
                        unsafe { (*a.limbs().get_unchecked(0), *b.limbs().get_unchecked(0)) };
                    // Choose the odd denominator prescribed by the state.
                    // At least one operand is odd throughout Euclidean reduction.
                    if (state & 32 == 0 && numerator & 1 != 0)
                        || (state & 32 != 0 && denominator & 1 == 0)
                    {
                        swap(&mut numerator, &mut denominator);
                    }
                    return Gcd::jacobi_limb(numerator, denominator, state & 16 != 0);
                }
                match a.cmp(&b) {
                    Ordering::Equal => return 0,
                    Ordering::Less => {
                        swap(&mut a, &mut b);
                        // Relabel the residues and divisor index together. This
                        // exchange changes the encoding, not the Jacobi symbol.
                        state = ((state & 3) << 2) | ((state >> 2) & 3) | ((state ^ 32) & 0x30);
                    }
                    Ordering::Greater => {}
                }
                if b.limbs().len() >= HGCD_CROSSOVER_THRESHOLD
                    && Gcd::hgcd_step::<true>(&mut a, &mut b, &mut scratch, workspace, &mut state)
                {
                    continue;
                }
                let scalar_quotient = if b.limbs().len() >= Gcd::LEHMER_MIN_LIMBS {
                    let mut candidate = state;
                    let (u0, v0, u1, v1, even) =
                        Gcd::simulate_step::<true>(a.limbs(), b.limbs(), None, branchless, |q| {
                            let index = (candidate << 2) | (q & 3);
                            // SAFETY: construction, exchange and table outputs
                            // preserve six state bits. Two masked quotient bits
                            // bound index below the initialized 256-entry table.
                            candidate =
                                usize::from(unsafe { *Gcd::JACOBI_QUOTIENT.get_unchecked(index) });
                        });
                    let identity = u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1;
                    if !identity
                        && Gcd::lehmer_update_dispatched(
                            &mut a,
                            &mut b,
                            &mut next_a,
                            &mut next_b,
                            u0,
                            v0,
                            u1,
                            v1,
                            even,
                            None,
                        )
                    {
                        // A rejected full-width application commits neither its
                        // operands nor its speculative quotient-sign transitions.
                        state = candidate;
                        continue;
                    }
                    Gcd::fast_small_div_step(&mut a, &b)
                } else {
                    None
                };
                let quotient_low = scalar_quotient.unwrap_or_else(|| {
                    Division::div_rem_into(&a, &b, &mut quotient, &mut next_a, &mut scratch);
                    swap(&mut a, &mut next_a);
                    // SAFETY: the ordered dividend exceeded its positive divisor,
                    // so the exact quotient has an initialized low limb.
                    unsafe { *quotient.limbs().get_unchecked(0) }
                });
                let index = (state << 2) | (quotient_low & 3);
                // SAFETY: construction, exchange and table outputs retain six
                // state bits; the masked quotient bounds index below 256.
                state = usize::from(unsafe { *Gcd::JACOBI_QUOTIENT.get_unchecked(index) });
                swap(&mut a, &mut b);
            }
            if !a.is_one() {
                0
            } else if state & 16 == 0 {
                1
            } else {
                -1
            }
        })
    }
}

/// Computes a two-limb Jacobi symbol through register-only Lehmer batches.
/// The denominator is normalized, positive and odd. Accepted leading-limb
/// matrices act on the full pair; quotient residues update the same sign
/// state used by the general Euclidean algorithm.
#[must_use]
pub fn jacobi_2(first: [Limb; 2], second: [Limb; 2]) -> i8 {
    debug_assert!(
        second[0] & 1 != 0 && second[1] != 0,
        "odd two-limb denominator"
    );
    // SAFETY: each native limb fits DoubleLimb; shifting a high limb
    // by Limb::BITS and combining the low limb fills exactly two limbs.
    let (mut numerator, mut denominator) = unsafe {
        (
            DoubleLimb::try_from(first[0]).unwrap_unchecked()
                | (DoubleLimb::try_from(first[1]).unwrap_unchecked() << Limb::BITS),
            DoubleLimb::try_from(second[0]).unwrap_unchecked()
                | (DoubleLimb::try_from(second[1]).unwrap_unchecked() << Limb::BITS),
        )
    };
    if numerator == 0 {
        return 0;
    }
    let mut state = (first[0] & 3) | ((second[0] & 3) << 2) | 32;
    // SAFETY: a native limb's maximum fits the twice-wide type exactly.
    let mask = unsafe { DoubleLimb::try_from(Limb::MAX).unwrap_unchecked() };
    while denominator != 0 && (numerator | denominator) > mask {
        if numerator < denominator {
            swap(&mut numerator, &mut denominator);
            state = ((state & 3) << 2) | ((state >> 2) & 3) | ((state ^ 32) & 0x30);
        }
        if denominator <= mask {
            // SAFETY: the loop proves a positive denominator. The exact
            // quotient may span two limbs; its low two bits suffice.
            let divisor = unsafe { NonZero::<DoubleLimb>::new_unchecked(denominator) };
            let quotient = Div::div(numerator, divisor);
            let remainder = Rem::rem(numerator, divisor);
            // SAFETY: masking bounds the infallible quotient conversion;
            // six state bits and two quotient bits bound the table index.
            state =
                unsafe {
                    usize::from(*Gcd::JACOBI_QUOTIENT.get_unchecked(
                        (state << 2) | Limb::try_from(quotient & 3).unwrap_unchecked(),
                    ))
                };
            numerator = denominator;
            denominator = remainder;
            break;
        }
        (numerator, denominator, state) = jacobi_2_step(numerator, denominator, state);
    }
    if denominator == 0 && numerator > mask {
        return 0;
    }
    // SAFETY: either both operands fit one limb or their terminal GCD
    // fits one limb. The other zero-GCD-width case returned above.
    let (mut native_numerator, mut native_denominator) = unsafe {
        (
            Limb::try_from(numerator).unwrap_unchecked(),
            Limb::try_from(denominator).unwrap_unchecked(),
        )
    };
    if (state & 32 == 0 && native_numerator & 1 != 0)
        || (state & 32 != 0 && native_denominator & 1 == 0)
    {
        swap(&mut native_numerator, &mut native_denominator);
    }
    Gcd::jacobi_limb(native_numerator, native_denominator, state & 16 != 0)
}

/// Reduces an ordered two-limb pair through a certified quotient prefix.
/// Requires `numerator>=denominator>Limb::MAX` and a six-bit sign state.
fn jacobi_2_step(
    numerator: DoubleLimb,
    denominator: DoubleLimb,
    state: Limb,
) -> (DoubleLimb, DoubleLimb, Limb) {
    // SAFETY: native maxima widen exactly; masking and high-half extraction
    // each produce one native limb from the ordered two-limb values.
    let (numerator_limbs, denominator_limbs) = unsafe {
        let mask = DoubleLimb::try_from(Limb::MAX).unwrap_unchecked();
        (
            [
                Limb::try_from(numerator & mask).unwrap_unchecked(),
                Limb::try_from(numerator >> Limb::BITS).unwrap_unchecked(),
            ],
            [
                Limb::try_from(denominator & mask).unwrap_unchecked(),
                Limb::try_from(denominator >> Limb::BITS).unwrap_unchecked(),
            ],
        )
    };
    let (upper_a, upper_b) = Gcd::extract_top_limb(&numerator_limbs, &denominator_limbs);
    let mut candidate = state;
    let (u0, v0, u1, v1, even) = Gcd::lehmer_simulate::<false>(upper_a, upper_b, |quotient| {
        // SAFETY: transitions retain six state bits; the quotient's
        // low two bits bound the table index below 256.
        candidate = unsafe {
            usize::from(*Gcd::JACOBI_QUOTIENT.get_unchecked((candidate << 2) | (quotient & 3)))
        };
    });
    if u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1 {
        // With numerator<B^2 and denominator>=B, the exact quotient fits
        // one limb. One remainder handles arbitrarily separated operands.
        let (quotient, remainder) = Gcd::div_rem_wide(numerator, denominator);
        // SAFETY: six state bits and two masked quotient bits bound the
        // table index; the low quotient conversion is below four.
        return unsafe {
            (
                denominator,
                remainder,
                usize::from(
                    *Gcd::JACOBI_QUOTIENT.get_unchecked(
                        (state << 2) | Limb::try_from(quotient & 3).unwrap_unchecked(),
                    ),
                ),
            )
        };
    }
    // Accepted simulation keeps each leading remainder at least its
    // adverse coefficient. For A=a*2^k+x and B=b*2^k+y, x,y<2^k,
    // each full matrix difference is therefore positive and no greater
    // than an original operand. Its scalar products may exceed B^2;
    // subtraction in Z/(B^2) still reconstructs the exact reduced pair.
    // SAFETY: every native coefficient widens without truncation.
    let (diagonal_first, cross_first, cross_second, diagonal_second) = unsafe {
        (
            DoubleLimb::try_from(u0).unwrap_unchecked(),
            DoubleLimb::try_from(v0).unwrap_unchecked(),
            DoubleLimb::try_from(u1).unwrap_unchecked(),
            DoubleLimb::try_from(v1).unwrap_unchecked(),
        )
    };
    let first = numerator
        .wrapping_mul(diagonal_first)
        .wrapping_sub(denominator.wrapping_mul(cross_first));
    let second = denominator
        .wrapping_mul(diagonal_second)
        .wrapping_sub(numerator.wrapping_mul(cross_second));
    if even {
        (first, second, candidate)
    } else {
        (first.wrapping_neg(), second.wrapping_neg(), candidate)
    }
}

impl Gcd {
    /// Computes `(numerator | denominator)` in registers, with an accumulated
    /// sign, for an odd positive `denominator`.
    ///
    /// Operands separated by the scalar division gap start with one remainder:
    /// a much larger numerator is reduced modulo the denominator, while a much
    /// smaller one exchanges places by reciprocity before that reduction.
    #[must_use]
    pub fn jacobi_limb(mut numerator: Limb, mut denominator: Limb, negative: bool) -> i8 {
        debug_assert!(denominator & 1 != 0, "binary Jacobi has an odd denominator");
        // SAFETY: the caller supplies an odd denominator, hence a positive
        // limb. The type exposes that invariant to the initial remainder path.
        let divisor = unsafe { NonZeroUsize::new_unchecked(denominator) };
        let mut sign = Limb::from(negative);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "profile validation bounds the scalar bit gap by 64, which fits u32 on every target"
        )]
        let division_shift = BINARY_EUCLID_DIVISION_SHIFT as u32;
        if let Some(high) = numerator.checked_shr(division_shift)
            && high > denominator
        {
            // Reduction modulo that denominator preserves the Jacobi symbol.
            numerator = Rem::rem(numerator, divisor);
        } else if numerator != 0
            && denominator
                .checked_shr(division_shift)
                .is_some_and(|high| high > numerator)
        {
            let twos = numerator.trailing_zeros();
            let odd = numerator >> twos;
            sign ^= Limb::from(
                (twos & 1 != 0 && matches!(denominator & 7, 3 | 5)) ^ (odd & denominator & 2 != 0),
            );
            // SAFETY: the odd part of a nonzero numerator is positive.
            numerator = Rem::rem(denominator, unsafe { NonZeroUsize::new_unchecked(odd) });
            denominator = odd;
        }
        if denominator == 1 {
            return if sign & 1 != 0 { -1 } else { 1 };
        }
        if numerator == 0 {
            return 0;
        }
        let initial_twos = numerator.trailing_zeros();
        numerator >>= initial_twos;
        denominator >>= 1;
        sign ^= Limb::from(initial_twos & 1 != 0) & (denominator ^ (denominator >> 1));
        numerator >>= 1;
        // Encode positive odd a,b as (a-1)/2,(b-1)/2. Both fit below B/2.
        // Subtraction identifies their order and the absolute difference. The
        // omitted low ones cancel; v_2(a-b)=1+v_2(encoded_a-encoded_b).
        while numerator != 0 {
            let (difference, exchange_needed) = numerator.overflowing_sub(denominator);
            if difference == 0 {
                return 0;
            }
            let exchange = 0_usize.wrapping_sub(Limb::from(exchange_needed));
            sign ^= exchange & numerator & denominator;
            denominator = if exchange_needed {
                numerator
            } else {
                denominator
            };
            // v_2(t)=v_2(-t): count directly from the subtraction, independently
            // of the absolute-value dependency chain. |t| < B/2 and t != 0 prove
            // trailing_zeros+1 < LIMB_BITS on all supported pointer widths.
            // SAFETY: |difference|<B/2 and difference!=0 prove this exact sum
            // is below LIMB_BITS on every supported pointer width.
            let twos = unsafe { difference.trailing_zeros().unchecked_add(1) };
            let magnitude = if exchange_needed {
                difference.wrapping_neg()
            } else {
                difference
            };
            numerator = magnitude >> twos;
            sign ^= Limb::from(twos & 1 != 0) & (denominator ^ (denominator >> 1));
        }
        if sign & 1 != 0 { -1 } else { 1 }
    }
}
