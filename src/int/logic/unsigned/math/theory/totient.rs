//! Euler's totient through trial division and Pollard-Brent factorization.

#![allow(
    unsafe_code,
    reason = "Magnitude ordering proves the smaller limb length is at most the larger resized output length; wheel_idx is masked to 0..=7 for the eight-element WHEEL_30 table."
)]

use core::{cmp::Ordering, mem::swap};

use alloc::vec::Vec;

use super::{
    ArchKernels, BarrettDomain, BarrettScratch, DivScratch, Division, InternalMpUint, Limb,
    MontgomeryDomain, MulScratch,
};

impl InternalMpUint {
    /// Euler's totient function `phi(self)`.
    ///
    /// For a prime `p`, `phi(p) = p - 1`.
    /// For composite inputs this function computes `phi` by performing
    /// trial division up to sqrt(self).
    ///
    /// Computes Euler's totient function `phi(n)`.
    ///
    /// Returns `None` if `self` is zero.
    #[must_use]
    pub fn euler_phi(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        if self.is_one() {
            return Some(Self::one());
        }
        if self.cmp(&Self::from_limb(2)) == Ordering::Equal {
            return Some(Self::one());
        }

        let mut result = self.clone();
        let mut seen = Vec::new();
        let mut div_scratch = DivScratch::default();
        let mut mul_scratch = MulScratch::default();

        get_prime_factors_phi(
            self.clone(),
            &mut result,
            &mut seen,
            &mut div_scratch,
            &mut mul_scratch,
        )?;

        Some(result)
    }
}

#[allow(
    clippy::many_single_char_names,
    reason = "Standard notation for Pollard's Rho algorithm: n, c, x, y, d, t1, t2"
)]
fn pollards_rho(
    n: &InternalMpUint,
    c_val: Limb,
    domain: &MontgomeryDomain,
    mul_scratch: &mut MulScratch,
) -> InternalMpUint {
    const BATCH: u32 = 128;
    const MAX_ITERS: u32 = 1_000_000;

    let mut temp_prod = InternalMpUint::zero();
    let mut t1 = InternalMpUint::zero();
    let mut t2 = InternalMpUint::zero();
    // #20: Pre-allocated diff buffer avoids allocation on every |x-y| computation.
    let mut diff = InternalMpUint::zero();
    debug_assert!(!n.is_zero(), "Pollard rho requires n greater than one");
    // SAFETY: Pollard rho is entered only for a composite n greater than one.
    let barrett_domain = BarrettDomain::new(n);
    let mut barrett_scratch = BarrettScratch::default();

    let mut x = domain.transform_into_with_scratch(
        &InternalMpUint::from_limb(2),
        &mut temp_prod,
        mul_scratch,
    );
    let mut y = x.clone();
    let c = domain.transform_into_with_scratch(
        &InternalMpUint::from_limb(c_val),
        &mut temp_prod,
        mul_scratch,
    );

    let mut product = InternalMpUint::one();
    // These are iteration counters, not limb indices. `u32` represents the
    // proven one-million-step limit on every supported pointer width.
    let mut power: u32 = 1;
    let mut lam: u32 = 0;
    let mut next_gcd: u32 = BATCH;

    loop {
        if lam >= MAX_ITERS {
            return n.clone();
        }

        // x = (x^2 + c) mod n  (single evaluation per iteration)
        domain.square_into_with_scratch(&x, &mut t1, &mut temp_prod, mul_scratch);
        t1.add_mod_into(&c, n, &mut x);

        lam = lam.wrapping_add(1);

        // #20: Compute |x - y| into pre-allocated diff buffer without allocation.
        compute_abs_diff(&x, &y, &mut diff);
        if diff.is_zero() {
            return n.clone();
        }

        // Accumulate diff in the standard domain to avoid a Montgomery transform per iteration.
        barrett_domain.mul_into_with_barrett_scratch(
            &product,
            &diff,
            &mut t2,
            &mut temp_prod,
            mul_scratch,
            &mut barrett_scratch,
        );
        swap(&mut product, &mut t2);

        // Batch GCD every BATCH iterations — counter avoids modulo.
        if lam == next_gcd {
            let d = product.gcd(n);
            if !d.is_one() {
                return d;
            }
            product.clone_from(&InternalMpUint::one());
            next_gcd = next_gcd.wrapping_add(BATCH);
        }

        // Brent's checkpoint: when we've done a power-of-two steps
        if lam == power {
            y.clone_from(&x);
            power = power.wrapping_mul(2);
        }
    }
}
/// Computes `|a - b|` into `out` without allocating a new `InternalMpUint`.
///
/// #20: Uses `sub_limbs_unchecked` to compute the difference in-place,
/// avoiding an allocated owned subtraction result.
pub fn compute_abs_diff(a: &InternalMpUint, b: &InternalMpUint, out: &mut InternalMpUint) {
    let a_limbs = a.limbs();
    let b_limbs = b.limbs();
    let a_len = a_limbs.len();
    let b_len = b_limbs.len();

    // Determine which is larger and compute larger - smaller.
    match a.cmp(b) {
        Ordering::Greater => {
            out.resize(a_len);
            let out_limbs = out.limbs_mut();
            out_limbs.copy_from_slice(a_limbs);
            if b_len > 0 {
                // SAFETY: normalized magnitude ordering `a > b` proves
                // `b_len <= a_len`; `out` has `a_len` initialized limbs, the
                // source has `b_len`, and the distinct borrows do not alias.
                let mut borrow = unsafe {
                    ArchKernels::sub_limbs_unchecked(
                        out_limbs.as_mut_ptr(),
                        b_limbs.as_ptr(),
                        b_len,
                    )
                };
                // Propagate borrow through the remaining higher limbs.
                // When a_len > b_len, the lower b_len limbs of a may be less than b,
                // producing a borrow that must ripple into the higher limbs.
                // SAFETY: normalized ordering proves `b_len <= a_len`, and
                // `out_limbs.len() = a_len`, so this possibly empty range is valid.
                for v in unsafe { out_limbs.get_unchecked_mut(b_len..a_len) } {
                    if borrow == 0 {
                        break;
                    }
                    let (diff, bw) = v.overflowing_sub(1);
                    *v = diff;
                    borrow = Limb::from(bw);
                }
                // Since a > b, the borrow must be zero after full propagation.
                debug_assert_eq!(borrow, 0, "borrow should be zero after subtraction");
            }
            out.normalize();
        }
        Ordering::Less => {
            out.resize(b_len);
            let out_limbs = out.limbs_mut();
            out_limbs.copy_from_slice(b_limbs);
            if a_len > 0 {
                // SAFETY: normalized magnitude ordering `b > a` proves
                // `a_len <= b_len`; `out` has `b_len` initialized limbs, the
                // source has `a_len`, and the distinct borrows do not alias.
                let mut borrow = unsafe {
                    ArchKernels::sub_limbs_unchecked(
                        out_limbs.as_mut_ptr(),
                        a_limbs.as_ptr(),
                        a_len,
                    )
                };
                // Propagate borrow through the remaining higher limbs.
                // When b_len > a_len, the lower a_len limbs of b may be less than a,
                // producing a borrow that must ripple into the higher limbs.
                // SAFETY: normalized ordering proves `a_len <= b_len`, and
                // `out_limbs.len() = b_len`, so this possibly empty range is valid.
                for v in unsafe { out_limbs.get_unchecked_mut(a_len..b_len) } {
                    if borrow == 0 {
                        break;
                    }
                    let (diff, bw) = v.overflowing_sub(1);
                    *v = diff;
                    borrow = Limb::from(bw);
                }
                // Since b > a, the borrow must be zero after full propagation.
                debug_assert_eq!(borrow, 0, "borrow should be zero after subtraction");
            }
            out.normalize();
        }
        Ordering::Equal => {
            out.clear();
        }
    }
}

fn apply_phi_factor(
    result: &mut InternalMpUint,
    p: &InternalMpUint,
    seen: &mut Vec<InternalMpUint>,
    div_scratch: &mut DivScratch,
    mul_scratch: &mut MulScratch,
) {
    if seen.iter().any(|s| s.cmp(p) == Ordering::Equal) {
        return;
    }
    seen.push(p.clone());
    let mut q = InternalMpUint::zero();
    let mut r = InternalMpUint::zero();
    Division::div_rem_into(result, p, &mut q, &mut r, div_scratch);
    debug_assert!(r.is_zero(), "a discovered prime factor divides the input");
    // Every discovered prime factor is at least two.
    let p_minus_1 = p.sub(&InternalMpUint::one());
    result.assign_product_with_scratch(&q, &p_minus_1, mul_scratch);
}

fn get_prime_factors_phi(
    mut n: InternalMpUint,
    result: &mut InternalMpUint,
    seen: &mut Vec<InternalMpUint>,
    div_scratch: &mut DivScratch,
    mul_scratch: &mut MulScratch,
) -> Option<()> {
    const WHEEL_30: [Limb; 8] = [4, 2, 4, 2, 4, 6, 2, 6];
    /// Trial division bound: primes below this threshold are found by
    /// direct division; larger factors are left to Pollard's rho.
    /// 50 000 keeps trial division fast even for multi-thousand-limb
    /// numbers while catching many small factors that would otherwise
    /// force a heavyweight Pollard rho run.
    const TRIAL_BOUND: usize = 50_000;

    if n.is_zero() || n.is_one() {
        return Some(());
    }

    if n.is_even() {
        let tz = n.trailing_zeros();
        n.shr_assign(tz);
        apply_phi_factor(
            result,
            &InternalMpUint::from_limb(2),
            seen,
            div_scratch,
            mul_scratch,
        );
        if n.is_one() {
            return Some(());
        }
    }

    let mut q = InternalMpUint::zero();
    let mut r = InternalMpUint::zero();

    let mut limit = n.isqrt();
    for &p_val in &[3_usize, 5_usize] {
        let p = InternalMpUint::from_limb(p_val);
        if p.cmp(&limit) == Ordering::Greater {
            break;
        }
        Division::div_rem_into(&n, &p, &mut q, &mut r, div_scratch);
        if r.is_zero() {
            apply_phi_factor(result, &p, seen, div_scratch, mul_scratch);
            // q already holds n / p; avoid a redundant division.
            n.clone_from(&q);
            limit = n.isqrt();
            loop {
                Division::div_rem_into(&n, &p, &mut q, &mut r, div_scratch);
                if r.is_zero() {
                    n.clone_from(&q);
                    limit = n.isqrt();
                } else {
                    break;
                }
            }
        }
    }

    let mut p = InternalMpUint::from_limb(7);
    let mut wheel_idx: usize = 0;
    let trial_bound = InternalMpUint::from_limb(TRIAL_BOUND);

    while p.cmp(&trial_bound) == Ordering::Less {
        if n.is_one() {
            return Some(());
        }
        if p.cmp(&limit) == Ordering::Greater {
            break;
        }

        Division::div_rem_into(&n, &p, &mut q, &mut r, div_scratch);
        if r.is_zero() {
            apply_phi_factor(result, &p, seen, div_scratch, mul_scratch);
            // q already holds n / p; avoid a redundant division.
            n.clone_from(&q);
            limit = n.isqrt();
            loop {
                Division::div_rem_into(&n, &p, &mut q, &mut r, div_scratch);
                if r.is_zero() {
                    n.clone_from(&q);
                    limit = n.isqrt();
                } else {
                    break;
                }
            }
        }

        // SAFETY: `wheel_idx` starts at zero and every update masks it with
        // `& 7`, so it remains in `0..=7` for the eight-element table.
        let step = InternalMpUint::from_limb(unsafe { *WHEEL_30.get_unchecked(wheel_idx) });
        p.add_assign(&step);
        wheel_idx = wheel_idx.wrapping_add(1) & 7;
    }

    if n.is_one() {
        return Some(());
    }

    factorize_recursive_phi(&n, result, seen, div_scratch, mul_scratch)
}

fn factorize_recursive_phi(
    n: &InternalMpUint,
    result: &mut InternalMpUint,
    seen: &mut Vec<InternalMpUint>,
    div_scratch: &mut DivScratch,
    mul_scratch: &mut MulScratch,
) -> Option<()> {
    if n.is_one() {
        return Some(());
    }
    if n.is_probably_prime(24) {
        apply_phi_factor(result, n, seen, div_scratch, mul_scratch);
        return Some(());
    }

    // #19: Create MontgomeryDomain once, reuse across all Pollard rho retries.
    debug_assert!(n.is_odd(), "even factors are removed before Pollard rho");
    // SAFETY: recursive factors are nonzero and odd after the initial factor-of-two removal.
    let domain = MontgomeryDomain::new(n);

    let mut c_val: Limb = 1;
    let mut d = pollards_rho(n, c_val, &domain, mul_scratch);
    let mut retries = 0_u32;
    while d.is_one() || d.cmp(n) == Ordering::Equal {
        retries = retries.wrapping_add(1);
        if retries > 100 {
            return None;
        }
        c_val = c_val.wrapping_add(2);
        d = pollards_rho(n, c_val, &domain, mul_scratch);
    }

    let mut q = InternalMpUint::zero();
    let mut r = InternalMpUint::zero();
    Division::div_rem_into(n, &d, &mut q, &mut r, div_scratch);
    debug_assert!(r.is_zero(), "Pollard rho result divides n");

    factorize_recursive_phi(&d, result, seen, div_scratch, mul_scratch)?;
    factorize_recursive_phi(&q, result, seen, div_scratch, mul_scratch)?;

    Some(())
}
