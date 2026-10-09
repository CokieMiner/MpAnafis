//! Euler's totient through trial division and Pollard-Brent factorization.
//!
//! One-limb inputs stay in native arithmetic: a short trial screen, one
//! Baillie-PSW test, the remaining trial table, then Pollard-Brent in
//! one-limb Montgomery form. Wider inputs share the trial table and split
//! their cofactors with Barrett-reduced Pollard-Brent; a one-limb factor
//! continues in the native factorization.
//!
//! References:
//! - J. M. Pollard, "A Monte Carlo method for factorization",
//!   BIT Numerical Mathematics 15(3), 331-334, 1975.
//!   DOI: 10.1007/BF01933667.
//! - R. P. Brent, "An improved Monte Carlo factorization algorithm",
//!   BIT Numerical Mathematics 20(2), 176-184, 1980.
//!   DOI: 10.1007/BF01933190.

#![expect(
    unsafe_code,
    reason = "accepted prime factors are nonzero divisors of the accumulator and cofactor, and a one-limb value has a zero high limb"
)]

use core::cmp::Ordering;

use alloc::vec::Vec;

use super::{
    ArchKernels, BarrettDomain, BarrettScratch, DivScratch, Division, Gcd, InternalMpUint, Limb,
    LimbMontgomery, MulScratch, PRIMALITY_PREFIX, PRIME_COUNT, Primality, Roots, TRIAL_BOUND,
    Totient,
};

/// Empirical amortization of GCD across Pollard products, independent of the
/// modulus width. A failed batch is recoverable one difference at a time.
const RHO_GCD_BATCH: u32 = 128;

/// Per-polynomial resource budget, not a mathematical convergence guarantee.
const RHO_ITERATION_LIMIT: u32 = 1_000_000;

/// Largest constant of the polynomials `x^2 + c`; the odd constants from one
/// form a bounded retry policy of 101 polynomials. Trial division leaves
/// composites above `50_000^2`, so the seed and every constant are reduced.
const RHO_CONSTANT_MAX: Limb = 201;

/// Largest modulus for which every redundant orbit square is below n*B.
const RHO_REDUNDANT_LIMIT: Limb = Limb::MAX.div_euclid(9);

impl InternalMpUint {
    /// Euler's totient function `phi(self)`.
    ///
    /// Returns `None` if `self` is zero or the bounded Pollard-Brent retry
    /// policy cannot factor a composite cofactor.
    #[must_use]
    pub fn euler_phi(&self) -> Option<Self> {
        let value = match self.limbs() {
            [] => return None,
            [value] => *value,
            _ => {
                let mut result = self.clone();
                let mut div_scratch = DivScratch::default();
                let cofactor = Totient::trial_phi(self.clone(), &mut result, &mut div_scratch);
                if !cofactor.is_one() {
                    // Trial division removes each factor completely. Every
                    // factor of the remaining cofactor is new; only recursive
                    // factorization needs duplicate tracking.
                    factorize_recursive_phi(
                        &cofactor,
                        &mut result,
                        &mut Vec::new(),
                        &mut div_scratch,
                        &mut MulScratch::default(),
                    )?;
                }
                return Some(result);
            }
        };

        // The first trial primes remove most factors cheaply. A cofactor that
        // survives them is tested once before the complete table runs, so a
        // prime input costs one short screen and one Baillie-PSW test.
        let twos = value.trailing_zeros();
        let mut n = value >> twos;
        let mut result = value >> u32::from(twos != 0);
        let screen_complete = Totient::trial_phi_limb(&mut n, &mut result, 0..PRIMALITY_PREFIX)
            || Primality::is_prime_limb(n);
        let screened = n;
        let complete = screen_complete
            || Totient::trial_phi_limb(&mut n, &mut result, PRIMALITY_PREFIX..PRIME_COUNT);
        let mut apply = |prime: Limb| {
            // SAFETY: a prime factor is nonzero and divides the accumulator;
            // its exact quotient is at most result.
            result = unsafe { result.unchecked_sub(result.checked_div(prime).unwrap_unchecked()) };
        };
        if !complete {
            factor_limb(n, n == screened, &mut apply)?;
        } else if n != 1 {
            apply(n);
        }
        Some(Self::from_limb(result))
    }
}

fn factorize_recursive_phi(
    n: &InternalMpUint,
    result: &mut InternalMpUint,
    seen: &mut Vec<InternalMpUint>,
    div_scratch: &mut DivScratch,
    mul_scratch: &mut MulScratch,
) -> Option<()> {
    debug_assert!(
        n.is_odd() && !n.is_one(),
        "trial division and each proper recursive factor leave odd n > 1"
    );
    if let [value] = n.limbs() {
        return factor_limb(*value, false, |prime| {
            apply_prime_phi(result, &InternalMpUint::from_limb(prime), seen, div_scratch);
        });
    }
    if Roots::may_be_square(n) {
        let (root, remainder) = n.sqrt_rem();
        if remainder.is_zero() {
            // phi(n)/n depends only on distinct prime factors. A square and
            // its root have exactly the same factors, so factor the smaller
            // root once while retaining the original totient accumulator.
            return factorize_recursive_phi(&root, result, seen, div_scratch, mul_scratch);
        }
    }
    if n.is_prime() {
        apply_prime_phi(result, n, seen, div_scratch);
        return Some(());
    }

    let d = pollard_brent(n, &BarrettDomain::new(n), mul_scratch)?;
    let mut q = InternalMpUint::zero();
    // Pollard returns gcd(product, n); an accepted nontrivial d divides n.
    Division::div_exact_into(n, &d, &mut q, div_scratch);

    factorize_recursive_phi(&d, result, seen, div_scratch, mul_scratch)?;
    factorize_recursive_phi(&q, result, seen, div_scratch, mul_scratch)
}

/// Multiplies the accumulator by `(p - 1)/p` once per distinct prime.
///
/// Unprocessed factors still divide the accumulator, so `result/p` is exact.
fn apply_prime_phi(
    result: &mut InternalMpUint,
    prime: &InternalMpUint,
    seen: &mut Vec<InternalMpUint>,
    div_scratch: &mut DivScratch,
) {
    if seen.iter().any(|known| known.cmp(prime) == Ordering::Equal) {
        return;
    }
    seen.push(prime.clone());
    let mut quotient = InternalMpUint::zero();
    Division::div_exact_into(result, prime, &mut quotient, div_scratch);
    result.sub_assign(&quotient);
}

/// Factors an odd limb greater than one, applying each distinct prime once.
///
/// A composite splits through Pollard-Brent until the retained divisor is
/// prime; every power of that prime is then divided out of the cofactor, so
/// no prime is applied twice. The complete trial table has already removed
/// every prime below `TRIAL_BOUND`. This property is inherited by all factors,
/// so a remaining factor below its square is prime without a primality test.
/// `known_composite` records a preceding failed test of the unchanged cofactor.
fn factor_limb(mut n: Limb, mut known_composite: bool, mut apply: impl FnMut(Limb)) -> Option<()> {
    while n != 1 {
        let mut factor = n;
        while known_composite
            // SAFETY: a limb has at most 64 bits on every supported target.
            || (unsafe { u64::try_from(factor).unwrap_unchecked() } >= TRIAL_BOUND * TRIAL_BOUND
                && !Primality::is_prime_limb(factor))
        {
            // Odd squares are 1 mod 8. Their roots have the same distinct
            // prime factors, so exact square reduction avoids a rho search.
            if factor & 7 == 1 {
                let root = factor.isqrt();
                if root.checked_mul(root) == Some(factor) {
                    factor = root;
                    known_composite = false;
                    continue;
                }
            }
            factor = if factor <= RHO_REDUNDANT_LIMIT {
                pollard_brent_limb::<true, true>(factor)?
            } else if factor <= Limb::MAX >> 2 {
                pollard_brent_limb::<true, false>(factor)?
            } else {
                pollard_brent_limb::<false, false>(factor)?
            };
            known_composite = false;
        }
        apply(factor);
        let inverse = Division::modular_inverse_limb(factor);
        // SAFETY: the accepted prime is odd, nonzero, and divides n exactly.
        let limit = unsafe { Limb::MAX.checked_div(factor).unwrap_unchecked() };
        n = n.wrapping_mul(inverse);
        loop {
            // The inverse test both recognizes divisibility and produces the
            // exact quotient, avoiding one division for each repeated factor.
            let quotient = n.wrapping_mul(inverse);
            if quotient > limit {
                break;
            }
            n = quotient;
        }
        known_composite = false;
    }
    Some(())
}

/// Returns a proper divisor of an odd composite with at least two limbs, or
/// `None` when every polynomial of the bounded retry policy fails.
///
/// Barrett reduction keeps both squares and products at multiplication
/// complexity. Each polynomial follows the native search below.
fn pollard_brent(
    n: &InternalMpUint,
    domain: &BarrettDomain,
    mul_scratch: &mut MulScratch,
) -> Option<InternalMpUint> {
    let mut temporary = InternalMpUint::zero();
    let mut reduced = InternalMpUint::zero();
    let mut diff = InternalMpUint::zero();
    let mut barrett_scratch = BarrettScratch::default();
    let mut fixed = InternalMpUint::zero();
    let mut saved = InternalMpUint::zero();
    let mut product = InternalMpUint::zero();
    let mut moving = InternalMpUint::zero();
    let mut constant = InternalMpUint::zero();
    for constant_limb in (1..=RHO_CONSTANT_MAX).step_by(2) {
        constant.set_limb(constant_limb);
        moving.set_limb(2);
        let mut power = 1_u32;
        let mut iterations = 0_u32;
        'polynomial: loop {
            // Reserve this complete round before entering either inner loop.
            // The prefix advances the orbit without forming redundant products.
            // SAFETY: every preceding round or replay stays within the
            // iteration limit; subtracting its used budget is exact.
            if power > unsafe { RHO_ITERATION_LIMIT.unchecked_sub(iterations) } >> 1 {
                break 'polynomial;
            }
            fixed.clone_from(&moving);
            for _ in 0..power {
                temporary.assign_square_with_scratch(&moving, mul_scratch);
                domain.reduce_into_with_barrett_scratch(
                    &temporary,
                    &mut reduced,
                    mul_scratch,
                    &mut barrett_scratch,
                );
                reduced.add_mod_into(&constant, n, &mut moving);
            }
            // SAFETY: the admission check reserves 2*power evaluations.
            iterations = unsafe { iterations.unchecked_add(power) };
            let mut remaining = power;
            while remaining != 0 {
                let batch = remaining.min(RHO_GCD_BATCH);
                saved.clone_from(&moving);
                product.set_limb(1);
                for _ in 0..batch {
                    temporary.assign_square_with_scratch(&moving, mul_scratch);
                    domain.reduce_into_with_barrett_scratch(
                        &temporary,
                        &mut reduced,
                        mul_scratch,
                        &mut barrett_scratch,
                    );
                    reduced.add_mod_into(&constant, n, &mut moving);
                    let (larger, smaller) = if moving >= fixed {
                        (&moving, &fixed)
                    } else {
                        (&fixed, &moving)
                    };
                    let underflow = diff.assign_difference(larger, smaller);
                    debug_assert!(!underflow, "the orbit difference is ordered");
                    // Products stay below n^2 and share the polynomial's reciprocal.
                    temporary.assign_product_with_scratch(&product, &diff, mul_scratch);
                    domain.reduce_into_with_barrett_scratch(
                        &temporary,
                        &mut product,
                        mul_scratch,
                        &mut barrett_scratch,
                    );
                }
                // SAFETY: batch <= remaining <= power; this round's prefix
                // and all processed batches use at most 2*power evaluations.
                iterations = unsafe { iterations.unchecked_add(batch) };
                let divisor = product.gcd(n);
                if divisor == *n {
                    // The batch may combine distinct factors. Replay its saved
                    // prefix until the first non-unit GCD, preserving useful work.
                    loop {
                        if iterations == RHO_ITERATION_LIMIT {
                            break 'polynomial;
                        }
                        temporary.assign_square_with_scratch(&saved, mul_scratch);
                        domain.reduce_into_with_barrett_scratch(
                            &temporary,
                            &mut reduced,
                            mul_scratch,
                            &mut barrett_scratch,
                        );
                        reduced.add_mod_into(&constant, n, &mut saved);
                        // SAFETY: replay stops at the limit before advancing.
                        iterations = unsafe { iterations.unchecked_add(1) };
                        let (larger, smaller) = if saved >= fixed {
                            (&saved, &fixed)
                        } else {
                            (&fixed, &saved)
                        };
                        let underflow = diff.assign_difference(larger, smaller);
                        debug_assert!(!underflow, "the replay difference is ordered");
                        let factor = diff.gcd(n);
                        if factor == *n {
                            break 'polynomial;
                        }
                        if !factor.is_one() {
                            return Some(factor);
                        }
                    }
                }
                if !divisor.is_one() {
                    return Some(divisor);
                }
                // SAFETY: batch=min(remaining,RHO_GCD_BATCH) <= remaining.
                remaining = unsafe { remaining.unchecked_sub(batch) };
            }
            // SAFETY: admission bounds power <= RHO_ITERATION_LIMIT/2.
            power = unsafe { power.unchecked_mul(2) };
        }
    }
    None
}

/// Returns a proper divisor of an odd composite limb, or `None` when every
/// polynomial of the bounded retry policy fails.
///
/// Brent's cycle search runs entirely in the one-limb Montgomery ring.
/// Advancing each checkpoint skips its already-covered prefix, and therefore
/// needs no product accumulation during that half of the cycle search.
/// Multiplying a difference by any power of `B` preserves its GCD with the
/// odd modulus. With `LAZY`, the caller proves `4*n < B`. Orbit values may
/// then lie in `[0, 2*n)`: their squares are below `4*n^2 < n*B`, within
/// Montgomery's input bound. Each square returns to `[0,n)`, so adding the
/// canonical constant restores `[0,2*n)` without a reduction. A canonical
/// product times an orbit difference is below `2*n^2 < n*B` as well.
/// With `REDUNDANT`, `9*n < B` also permits products in `(0,2*n)` and orbit
/// values in `(0,3*n)`: their squares are below `9*n^2 < n*B`, and batch
/// products times differences are below `6*n^2 < n*B`. Adding n to every
/// Montgomery difference then replaces its conditional canonical correction.
fn pollard_brent_limb<const LAZY: bool, const REDUNDANT: bool>(n: Limb) -> Option<Limb> {
    debug_assert!(
        !LAZY || n <= Limb::MAX >> 2,
        "unreduced orbit additions require four moduli to fit one limb"
    );
    debug_assert!(
        !REDUNDANT || (LAZY && n <= RHO_REDUNDANT_LIMIT),
        "redundant products and orbit additions require nine moduli to fit one limb"
    );
    let domain = LimbMontgomery::new(n);
    let seed = domain.multiply(2, domain.radix_square);
    let multiply = |left: Limb, right: Limb| {
        if REDUNDANT {
            // Both call sites maintain left*right < n*B. The product's
            // high limb and the cancellation product's high limb are below
            // n. Their difference plus n lies in (0,2*n), below B.
            let (low, high) = ArchKernels::mul_limb_lo_hi(left, right);
            let (_, correction) = ArchKernels::mul_limb_lo_hi(low.wrapping_mul(domain.inverse), n);
            // SAFETY: high,correction < n and 9*n < B prove high+n < B
            // and correction < high+n. The result represents left*right/B.
            unsafe { high.unchecked_add(n).unchecked_sub(correction) }
        } else {
            domain.multiply(left, right)
        }
    };
    for constant in (1..=RHO_CONSTANT_MAX).step_by(2) {
        let c = domain.multiply(constant, domain.radix_square);
        let advance = |value: Limb| {
            let square = multiply(value, value);
            if LAZY {
                // SAFETY: c < n. With REDUNDANT, square < 2*n and 9*n < B
                // give sum < 3*n < B. Otherwise square < n and 4*n < B
                // give sum < 2*n < B. Both preserve the orbit's input bound.
                unsafe { square.unchecked_add(c) }
            } else {
                domain.add(square, c)
            }
        };
        let mut moving = seed;
        let mut power = 1_u32;
        let mut iterations = 0_u32;
        'polynomial: loop {
            // This round takes at most 2*power polynomial evaluations. Checking
            // once bounds both loops and every counter addition on 16-bit hosts
            // as well: counters are u32, never allocation sizes or limb indices.
            // SAFETY: each admitted round and replay preserves iterations
            // <= RHO_ITERATION_LIMIT, so the remaining budget is exact.
            let available = unsafe { RHO_ITERATION_LIMIT.unchecked_sub(iterations) };
            if power > available >> 1 {
                break 'polynomial;
            }
            let fixed = moving;
            for _ in 0..power {
                moving = advance(moving);
            }
            // SAFETY: admission reserves 2*power evaluations for this round.
            iterations = unsafe { iterations.unchecked_add(power) };
            let mut remaining = power;
            while remaining != 0 {
                let batch = remaining.min(RHO_GCD_BATCH);
                let saved = moving;
                let mut product = domain.one;
                for _ in 0..batch {
                    moving = advance(moving);
                    product = multiply(product, moving.abs_diff(fixed));
                }
                // SAFETY: the prefix and processed batches use at most the
                // admitted 2*power evaluations, within the iteration limit.
                iterations = unsafe { iterations.unchecked_add(batch) };
                let divisor = Gcd::gcd_1(product, n);
                if divisor == n {
                    // A zero batch product can combine different proper
                    // factors. Replay only that batch, stopping at its first
                    // non-unit GCD; a full-modulus GCD ends this polynomial.
                    let mut replay = saved;
                    loop {
                        if iterations == RHO_ITERATION_LIMIT {
                            break 'polynomial;
                        }
                        replay = advance(replay);
                        // SAFETY: the preceding limit test admits one step.
                        iterations = unsafe { iterations.unchecked_add(1) };
                        match Gcd::gcd_1(replay.abs_diff(fixed), n) {
                            1 => {}
                            factor if factor == n => break 'polynomial,
                            factor => return Some(factor),
                        }
                    }
                }
                if divisor != 1 {
                    return Some(divisor);
                }
                // SAFETY: batch=min(remaining,RHO_GCD_BATCH) <= remaining.
                remaining = unsafe { remaining.unchecked_sub(batch) };
            }
            // SAFETY: admission proves power <= RHO_ITERATION_LIMIT/2.
            power = unsafe { power.unchecked_mul(2) };
        }
    }
    None
}
