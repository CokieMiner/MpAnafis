//! Baillie-PSW: strong base-2 Miller-Rabin and strong Lucas-Selfridge testing.
//!
//! Lucas parameters follow Selfridge's method A, with P=1 and Q=(1-D)/4.
//! The doubling and increment identities are equations (13)-(18) in Baillie,
//! Fiori, and Wagstaff, "Strengthening the Baillie-PSW primality test":
//! <https://homes.cerias.purdue.edu/~ssw/bfw.pdf>.
//! Passing establishes probable primality, not a general primality proof.

#![expect(
    unsafe_code,
    reason = "A positive odd Lucas exponent and n+1's positive two-adic valuation prove exact loop bounds."
)]

use core::mem::swap;

use super::{InternalMpUint, MontgomeryDomain, MontgomeryScratch, Primality};

impl Primality {
    /// Tests an odd integer greater than one with the Baillie-PSW combination.
    pub fn baillie_psw(n: &InternalMpUint) -> bool {
        debug_assert!(n.is_odd() && !n.is_one(), "Baillie-PSW requires odd n > 1");
        if !Self::trial_division(n) {
            return false;
        }
        let domain = MontgomeryDomain::new::<true>(n);
        let mut mul_scratch = MontgomeryScratch::default();
        let mut product = InternalMpUint::zero();
        let mut residue = InternalMpUint::zero();
        let one = domain.transform_into_with_scratch(
            &InternalMpUint::one(),
            &mut product,
            &mut mul_scratch,
        );
        // Montgomery encoding is linear: the representation of -1 is n-R.
        let minus_one = n.sub(&one);
        // For odd n > 1, (n - 1) >> twos == n >> twos.
        let twos = Self::trailing_zeros_odd_minus_one(n);
        let odd_part = n.shr(twos);
        if !Self::miller_rabin_test(
            &InternalMpUint::from_limb(2),
            &odd_part,
            twos,
            &minus_one,
            &one,
            &mut product,
            &mut residue,
            &domain,
            &mut mul_scratch,
        ) {
            return false;
        }
        strong_lucas_selfridge(&domain, &one, &mut product, &mut mul_scratch)
    }
}

/// Tests strong Lucas conditions with Selfridge parameters in one domain.
///
/// `one` is the Montgomery encoding of one; the modulus must be odd and > 1.
pub fn strong_lucas_selfridge(
    domain: &MontgomeryDomain,
    one: &InternalMpUint,
    product: &mut InternalMpUint,
    mul_scratch: &mut MontgomeryScratch,
) -> bool {
    let n = &domain.modulus;
    let Some((q_mont, d_mont)) = selfridge_parameters(domain, product, mul_scratch) else {
        return false;
    };

    // n+1 = odd_part * 2^twos. Start at subscript one, with U_1=V_1=1.
    let mut odd_part = n.add(&InternalMpUint::one());
    let twos = odd_part.trailing_zeros();
    odd_part.shr_assign(twos);
    let mut u = one.clone();
    let mut v = one.clone();
    let mut q_power = q_mont.clone();
    let mut next_u = InternalMpUint::zero();
    let mut next_v = InternalMpUint::zero();
    let mut next_q = InternalMpUint::zero();
    let mut sum = InternalMpUint::zero();
    // SAFETY: n+1 > 0; removing its factors of two leaves odd_part > 0.
    let remaining = unsafe { odd_part.significant_bits().unchecked_sub(1) };
    for bit in (0..remaining).rev() {
        // U_2k=U_k*V_k, V_2k=V_k^2-2Q^k, Q^2k=(Q^k)^2.
        domain.mul_into_with_scratch(&u, &v, &mut next_u, product, mul_scratch);
        product.assign_square_with_scratch(&v, &mut mul_scratch.multiplication);
        domain.reduce_into(product, &mut next_v, mul_scratch);
        // Both operands are canonical: their sum is below 2n and their
        // absolute difference is below n. Each needs at most one correction.
        sum.assign_sum(&q_power, &q_power);
        if sum >= *n {
            sum.sub_assign(n);
        }
        let negative = next_v < sum;
        let underflow = if negative {
            v.assign_difference(&sum, &next_v)
        } else {
            v.assign_difference(&next_v, &sum)
        };
        debug_assert!(!underflow, "Lucas residue subtraction is ordered");
        if negative {
            v.negate_reduced_modulo(n);
        }
        product.assign_square_with_scratch(&q_power, &mut mul_scratch.multiplication);
        domain.reduce_into(product, &mut next_q, mul_scratch);
        swap(&mut u, &mut next_u);
        swap(&mut q_power, &mut next_q);
        if odd_part.get_bit(bit) {
            // P=1: U_(k+1)=(U_k+V_k)/2 and V_(k+1)=(D*U_k+V_k)/2.
            next_u.assign_sum(&u, &v);
            if next_u >= *n {
                next_u.sub_assign(n);
            }
            // Adding odd n to an odd residue gives its even representative
            // below 2n; halving it then returns a canonical residue.
            if next_u.is_odd() {
                next_u.add_assign(n);
            }
            next_u.shr_assign(1);
            domain.mul_into_with_scratch(&d_mont, &u, &mut next_v, product, mul_scratch);
            sum.assign_sum(&next_v, &v);
            if sum >= *n {
                sum.sub_assign(n);
            }
            if sum.is_odd() {
                sum.add_assign(n);
            }
            sum.shr_assign(1);
            swap(&mut u, &mut next_u);
            swap(&mut v, &mut sum);
            domain.mul_into_with_scratch(&q_power, &q_mont, &mut next_q, product, mul_scratch);
            swap(&mut q_power, &mut next_q);
        }
    }
    if u.is_zero() || v.is_zero() {
        return true;
    }
    // SAFETY: odd n gives n+1 even, hence twos >= 1.
    let last_round = unsafe { twos.unchecked_sub(1) };
    for round in 1..twos {
        product.assign_square_with_scratch(&v, &mut mul_scratch.multiplication);
        domain.reduce_into(product, &mut next_v, mul_scratch);
        sum.assign_sum(&q_power, &q_power);
        if sum >= *n {
            sum.sub_assign(n);
        }
        let negative = next_v < sum;
        let underflow = if negative {
            v.assign_difference(&sum, &next_v)
        } else {
            v.assign_difference(&next_v, &sum)
        };
        debug_assert!(!underflow, "Lucas residue subtraction is ordered");
        if negative {
            v.negate_reduced_modulo(n);
        }
        if v.is_zero() {
            return true;
        }
        if round < last_round {
            product.assign_square_with_scratch(&q_power, &mut mul_scratch.multiplication);
            domain.reduce_into(product, &mut next_q, mul_scratch);
            swap(&mut q_power, &mut next_q);
        }
    }
    false
}

/// Selects P=1, Q=(1-D)/4 with (D/n)=-1 and returns their domain residues.
///
/// Squares and proper factors encountered during selection reject n.
fn selfridge_parameters(
    domain: &MontgomeryDomain,
    product: &mut InternalMpUint,
    mul_scratch: &mut MontgomeryScratch,
) -> Option<(InternalMpUint, InternalMpUint)> {
    let n = &domain.modulus;
    // Squares have no Jacobi nonresidue. Reject them before the parameter
    // search so its termination follows from n being a positive nonsquare.
    if n.is_perfect_square() {
        return None;
    }
    let mut discriminant = InternalMpUint::from_limb(5);
    let mut negative = false;
    let negate_symbol = n.get_bit(1);
    let two = InternalMpUint::from_limb(2);
    loop {
        let mut symbol = discriminant.jacobi_symbol(n);
        if negative && negate_symbol {
            symbol = symbol.wrapping_neg();
        }
        if symbol == -1 {
            break;
        }
        if symbol == 0 && !discriminant.is_divisible_by(n) {
            // Jacobi zero supplies a factor. If n divides D this is not
            // a proper-factor certificate, and parameter selection continues.
            return None;
        }
        discriminant.add_assign(&two);
        negative = !negative;
    }
    let mut q_magnitude = discriminant.clone();
    if negative {
        q_magnitude.increment();
    } else {
        q_magnitude.decrement();
    }
    q_magnitude.shr_assign(2);
    if !q_magnitude.gcd(n).is_one() {
        return None;
    }
    let mut q_residue = if q_magnitude < *n {
        q_magnitude
    } else {
        q_magnitude.rem(n)
    };
    if !negative && !q_residue.is_zero() {
        q_residue = n.sub(&q_residue);
    }
    let mut d_residue = if discriminant < *n {
        discriminant
    } else {
        discriminant.rem(n)
    };
    if negative && !d_residue.is_zero() {
        d_residue = n.sub(&d_residue);
    }
    let q_mont = domain.transform_into_with_scratch(&q_residue, product, mul_scratch);
    let d_mont = domain.transform_into_with_scratch(&d_residue, product, mul_scratch);
    Some((q_mont, d_mont))
}
