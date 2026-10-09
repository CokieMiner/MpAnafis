//! Deterministic primality for one-limb integers.
//!
//! A compact bitmap decides small inputs. Larger odd inputs are prime exactly
//! when they are a strong base-2 probable prime
//! and an extra strong Lucas probable prime with `Q = 1` and the least
//! `P = 3, 4, 5, ...` for which `(P^2 - 4 | n) = -1`. No base-2 Fermat
//! pseudoprime below `2^64` passes this Lucas test, so the Baillie-PSW
//! combination is exact on 16-, 32-, and 64-bit limbs. Both tests run in
//! one-limb Montgomery arithmetic; neither needs a double-limb remainder.
//!
//! References:
//! - R. Baillie and S. S. Wagstaff Jr., "Lucas Pseudoprimes", Mathematics of
//!   Computation, Vol. 35, No. 152, pp. 1391-1417, 1980. DOI: 10.2307/2006406.
//! - J. Grantham, "Frobenius Pseudoprimes", Mathematics of Computation,
//!   Vol. 70, No. 234, pp. 873-891, 2001. DOI: 10.1090/S0025-5718-00-01197-2.
//! - J. Feitsma and W. Galway, tables of base-2 Fermat pseudoprimes below
//!   `2^64`, against which the Baillie-PSW variants have been verified.

#![expect(
    unsafe_code,
    reason = "Positive exponent positions, Lucas parameters below the modulus, and canonical residues prove exact arithmetic and bounded native division."
)]

use super::{ArchKernels, Gcd, Limb, LimbMontgomery, ODD_COMPOSITE, Primality};

/// Bit `k` is set exactly when `k < 64` is prime.
const PRIMES_BELOW_64: u64 = 0x2820_8A20_A08A_28AC;

impl Primality {
    /// Decides primality of one limb.
    ///
    /// The strong base-2 test computes `2^s` for `n - 1 = s*2^r`, `s` odd,
    /// doubling instead of multiplying for each set exponent bit, then squares
    /// at most `r - 1` times while looking for `-1`.
    #[must_use]
    pub fn is_prime_limb(n: Limb) -> bool {
        if n < 64 {
            return (PRIMES_BELOW_64 >> n) & 1 != 0;
        }
        if n & 1 == 0 {
            return false;
        }
        if let Some(&byte) = ODD_COMPOSITE.get(n >> 4) {
            return byte & (1 << ((n >> 1) & 7)) == 0;
        }
        let domain = LimbMontgomery::new(n);
        // SAFETY: n >= 64 and the even cases returned above, so n > 1.
        let even = unsafe { n.unchecked_sub(1) };
        let twos = even.trailing_zeros();
        let odd = even >> twos;
        // -1 is represented by n - (B mod n); B mod n is a nonzero unit.
        // SAFETY: Montgomery one is a canonical nonzero residue below odd n.
        let minus_one = unsafe { n.unchecked_sub(domain.one) };
        // A prefix of at most log2(LIMB_BITS) exponent bits has value
        // k < LIMB_BITS. Its power 2^k fits a native limb, so one domain
        // conversion replaces the prefix's entire square-and-double ladder.
        // SAFETY: even > 0 and removing its factors of two leaves odd > 0.
        let bits = unsafe { Limb::BITS.unchecked_sub(odd.leading_zeros()) };
        let prefix_bits = bits.min(Limb::BITS.ilog2());
        // SAFETY: prefix_bits = min(bits, log2(LIMB_BITS)) <= bits.
        let mut bit = unsafe { bits.unchecked_sub(prefix_bits) };
        let prefix = odd >> bit;
        let mut value = domain.multiply(1_usize << prefix, domain.radix_square);
        while bit != 0 {
            // SAFETY: the loop condition establishes bit > 0.
            bit = unsafe { bit.unchecked_sub(1) };
            value = domain.multiply(value, value);
            if (odd >> bit) & 1 != 0 {
                value = domain.add(value, value);
            }
        }
        if value != domain.one && value != minus_one {
            let mut squarings = 1;
            loop {
                if squarings == twos {
                    return false;
                }
                value = domain.multiply(value, value);
                if value == minus_one {
                    break;
                }
                if value == domain.one {
                    // Squaring fixes one, so minus one is no longer reachable.
                    return false;
                }
                // SAFETY: squarings < twos <= Limb::BITS; equality returned above.
                squarings = unsafe { squarings.unchecked_add(1) };
            }
        }
        extra_strong_lucas(&domain)
    }
}

/// Extra strong Lucas test for a strong base-2 probable prime above 64.
///
/// The parameter search stops at the least `P >= 3` with
/// `(P^2 - 4 | n) = -1`. A nonsquare odd `n > 64` has such a `P` below `n`:
/// choose `P` modulo one prime of odd multiplicity so that its symbol is `-1`
/// (for `3`, `P = 0`), and modulo every other prime so that its factor is
/// `+1`; the Chinese remainder class meets `[3, n)`. Squares, whose symbols
/// are never `-1`, are rejected once `P` reaches eight, still below `n`. A
/// nonzero discriminant with symbol zero shares a proper factor with `n`.
///
/// With `n + 1 = s*2^r`, `s` odd, `n` passes when `U_s = 0` and `V_s = +-2`,
/// or when `V_(s*2^t) = 0` for some `t < r - 1`. The ladder keeps
/// `(V_k, V_(k+1))`; since `D = P^2 - 4` is a unit, `U_s = (2V_(s+1) - P*V_s)/D`
/// vanishes exactly when `2V_(s+1) = P*V_s`.
fn extra_strong_lucas(domain: &LimbMontgomery) -> bool {
    let n = domain.modulus;
    let mut parameter: Limb = 3;
    loop {
        let (low, high) = ArchKernels::mul_limb_lo_hi(parameter, parameter);
        let square = if high == 0 && low < n {
            low
        } else {
            // SAFETY: the search stops at or before the least admissible
            // P < n, so P^2 < n*B and its high limb is below the nonzero n.
            unsafe {
                let (_, square) = ArchKernels::divrem_1_unchecked(low, high, n);
                square
            }
        };
        // Both residues are canonical because n > 64 > 4.
        let discriminant = domain.subtract(square, 4);
        if discriminant != 0 {
            match Gcd::jacobi_limb(discriminant, n, false) {
                -1 => break,
                0 => return false,
                _ => {}
            }
        }
        // SAFETY: an admissible parameter exists below n <= Limb::MAX; the
        // search has not reached it, and square inputs return at parameter eight.
        parameter = unsafe { parameter.unchecked_add(1) };
        if parameter == 8 {
            let root = n.isqrt();
            if root.checked_mul(root) == Some(n) {
                return false;
            }
        }
    }

    let p = domain.multiply(parameter, domain.radix_square);
    let two = domain.add(domain.one, domain.one);
    // n is odd, so n + 1 = 2*(floor(n/2) + 1) with a nonzero halved value.
    // SAFETY: n >> 1 <= Limb::MAX / 2, so adding one is representable.
    let half = unsafe { (n >> 1).unchecked_add(1) };
    let odd = half >> half.trailing_zeros();
    // SAFETY: half > 0 bounds its zero count below Limb::BITS <= 64.
    let twos = unsafe { half.trailing_zeros().unchecked_add(1) };

    // Start at k = 1 with (V_1, V_2) = (P, P^2 - 2). Each remaining bit maps
    // k to 2k or 2k + 1 through V_2k = V_k^2 - 2 and
    // V_(2k+1) = V_k*V_(k+1) - P; the square and the product are independent.
    let mut low = p;
    let mut high = domain.subtract(domain.multiply(p, p), two);
    // SAFETY: half > 0 gives odd > 0 and a significant width in 1..=Limb::BITS.
    let mut bit = unsafe {
        Limb::BITS
            .unchecked_sub(odd.leading_zeros())
            .unchecked_sub(1)
    };
    while bit != 0 {
        // SAFETY: the loop condition establishes bit > 0.
        bit = unsafe { bit.unchecked_sub(1) };
        let mixed = domain.subtract(domain.multiply(low, high), p);
        if (odd >> bit) & 1 == 0 {
            high = mixed;
            low = domain.subtract(domain.multiply(low, low), two);
        } else {
            low = mixed;
            high = domain.subtract(domain.multiply(high, high), two);
        }
    }

    let minus_two = domain.subtract(0, two);
    if (low == two || low == minus_two) && domain.add(high, high) == domain.multiply(p, low) {
        return true;
    }
    for _ in 1..twos {
        if low == 0 {
            return true;
        }
        low = domain.subtract(domain.multiply(low, low), two);
    }
    false
}
