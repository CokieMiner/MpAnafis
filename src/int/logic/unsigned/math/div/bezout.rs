//! Extended Euclid and modular inversion.
//!
//! Recursive HGCD blocks and Lehmer batches reduce the operands in
//! [`Division::compute_extended_euclid_core`]. This module reconstructs coefficient magnitudes and
//! handles single-limb inputs.
//!
//! References:
//! - D. H. Lehmer, "Euclid's Algorithm for Large Numbers", The American Mathematical
//!   Monthly, Vol. 45, No. 4, pp. 227–233, Apr. 1938. DOI: 10.2307/2302609.

#![expect(
    unsafe_code,
    reason = "normalized coefficient and modulus shapes establish initialized limb access and nonzero scalar divisors"
)]

use core::cmp::Ordering;

use super::{DivScratch, Division, InternalMpUint, Limb};

/// Extended-GCD result with absolute coefficients and their opposing signs.
///
/// `a*x_magnitude - b*y_magnitude = gcd` when `x_is_positive`, and
/// `b*y_magnitude - a*x_magnitude = gcd` otherwise. The signed API uses
/// these magnitudes directly; the unsigned API complements the negative
/// coefficient to obtain its residue representative.
#[derive(Debug)]
pub struct BezoutResult {
    pub gcd: InternalMpUint,
    pub x_magnitude: InternalMpUint,
    pub y_magnitude: InternalMpUint,
    pub x_is_positive: bool,
}

impl InternalMpUint {
    /// Computes the modular inverse of `self` modulo `modulus` using the extended
    /// Euclidean algorithm.
    ///
    /// Returns `None` when the inverse does not exist (i.e. when
    /// `gcd(self, modulus) != 1`). `modulus` must be non-zero.
    #[must_use]
    pub fn invert(&self, modulus: &Self) -> Option<Self> {
        debug_assert!(
            !modulus.is_zero(),
            "modular inversion requires a non-zero modulus"
        );
        if modulus.is_one() {
            return Some(Self::zero());
        }
        if self.is_zero() {
            return None;
        }

        // The units 1 and -1 have themselves as inverses. The predecessor check
        // is limb-wise so the common `a = m - 1` case avoids extended-GCD state
        // and a temporary subtraction of the modulus.
        if self.is_one() {
            return Some(Self::one());
        }
        if is_modulus_predecessor(self, modulus) {
            return Some(self.clone());
        }

        let a_limbs = self.limbs();
        let m_limbs = modulus.limbs();
        if let [scalar_modulus] = m_limbs {
            let numerator = if let [scalar] = a_limbs {
                *scalar
            } else {
                Division::div_rem_1::<false>(a_limbs, *scalar_modulus, &mut Self::zero())
            };
            let (gcd_val, coefficient, _, parity) =
                Division::extended_gcd_limb(numerator, *scalar_modulus);
            if gcd_val != 1 {
                return None;
            }
            debug_assert_ne!(coefficient, 0, "an inverse modulo m > 1 is nonzero");
            let residue = if parity == 0 {
                coefficient
            } else {
                // SAFETY: the coprime surviving Euclidean coefficient
                // satisfies 0 < coefficient < scalar_modulus.
                unsafe { scalar_modulus.unchecked_sub(coefficient) }
            };
            return Some(Self::from_limb(residue));
        }

        if let Some((gcd, mut coefficient, _, positive, _)) =
            close_coefficients::<true>(self, modulus)
        {
            if gcd != 1 {
                return None;
            }
            if !positive {
                coefficient.negate_reduced_modulo(modulus);
            }
            return Some(coefficient);
        }

        let reversed = a_limbs.len() < m_limbs.len();
        let (gcd, mut coefficient, mut step) = if let [scalar] = a_limbs {
            let remainder = Division::div_rem_1::<false>(m_limbs, *scalar, &mut Self::zero());
            let (g, value, _, parity) = Division::extended_gcd_limb(remainder, *scalar);
            (Self::from_limb(g), Self::from_limb(value), parity)
        } else if reversed {
            // Track the smaller cofactor family; reject noncoprime operands
            // before reconstructing the requested modulus-wide coefficient.
            Division::compute_extended_euclid_core(modulus, self)
        } else {
            Division::compute_extended_euclid_core(self, modulus)
        };
        if !gcd.is_one() {
            return None;
        }
        if reversed {
            // m*s - a*t = (-1)^step. The coefficient of a therefore has
            // opposite sign and magnitude (m*s - (-1)^step)/a, exactly.
            let mut product = Self::zero();
            let mut scratch = DivScratch::default();
            product.assign_product_with_scratch(modulus, &coefficient, &mut scratch.mul_scratch);
            if step & 1 == 0 {
                product.sub_assign(&gcd);
            } else {
                product.add_assign(&gcd);
            }
            Division::div_exact_into(&product, self, &mut coefficient, &mut scratch);
            step ^= 1;
        }

        // The terminal zero-row coefficient is m/gcd = m. The surviving
        // coefficient is strictly smaller: the final positive quotient
        // combines it with its predecessor to make that zero row. For
        // m > 1 and gcd = 1 it is also nonzero. No residue division is needed.
        debug_assert!(
            !coefficient.is_zero() && coefficient < *modulus,
            "a coprime pair with modulus > 1 has a nonzero coefficient below the modulus"
        );
        if step & 1 != 0 {
            coefficient.negate_reduced_modulo(modulus);
        }
        Some(coefficient)
    }

    /// Returns the GCD, absolute coefficients, and the sign of the
    /// coefficient of `self`.
    ///
    /// With magnitudes `(x, y)`, either `self*x - other*y = gcd` or
    /// `other*y - self*x = gcd`, as recorded by `x_is_positive`. Public
    /// signed and unsigned APIs choose their coefficient representation.
    #[must_use]
    pub fn extended_gcd(&self, other: &Self) -> BezoutResult {
        if self.is_zero() {
            return BezoutResult {
                gcd: other.clone(),
                x_magnitude: Self::zero(),
                y_magnitude: Self::one(),
                x_is_positive: false,
            };
        }
        if other.is_zero() || self.cmp(other) == Ordering::Equal {
            return BezoutResult {
                gcd: self.clone(),
                x_magnitude: Self::one(),
                y_magnitude: Self::zero(),
                x_is_positive: true,
            };
        }

        let a_limbs = self.limbs();
        let b_limbs = other.limbs();
        if let ([left], [right]) = (a_limbs, b_limbs) {
            let (gcd_val, s_val, t_val, parity) = Division::extended_gcd_limb(*left, *right);
            return BezoutResult {
                gcd: Self::from_limb(gcd_val),
                x_magnitude: Self::from_limb(s_val),
                y_magnitude: Self::from_limb(t_val),
                x_is_positive: parity == 0,
            };
        }

        if let Some((gcd, x_magnitude, scalar, x_is_positive, add_scalar)) =
            close_coefficients::<false>(self, other)
        {
            let mut y_magnitude = x_magnitude.clone();
            let adjustment = Self::from_limb(scalar);
            if add_scalar {
                y_magnitude.add_assign(&adjustment);
            } else {
                y_magnitude.sub_assign(&adjustment);
            }
            return BezoutResult {
                gcd: Self::from_limb(gcd),
                x_magnitude,
                y_magnitude,
                x_is_positive,
            };
        }

        // The second operand bounds the growing coefficients. Tracking the
        // longer operand's coefficient therefore minimizes their maximum width;
        // swapping the two final magnitudes restores the public operand order.
        let reversed = a_limbs.len() < b_limbs.len();
        let (head, tail) = if reversed {
            (other, self)
        } else {
            (self, other)
        };
        let (gcd, final_s, step) = if let [scalar] = tail.limbs() {
            let remainder = Division::div_rem_1::<false>(head.limbs(), *scalar, &mut Self::zero());
            let (g, coefficient, _, parity) = Division::extended_gcd_limb(remainder, *scalar);
            (Self::from_limb(g), Self::from_limb(coefficient), parity)
        } else {
            Division::compute_extended_euclid_core(head, tail)
        };
        let final_t = if final_s.is_zero() {
            // With nonzero inputs, a zero coefficient of head gives gcd =
            // tail and the surviving coefficient of tail is exactly one.
            Self::one()
        } else {
            let mut coefficient = Self::zero();
            let mut product = Self::zero();
            let mut scratch = DivScratch::default();
            product.assign_product_with_scratch(head, &final_s, &mut scratch.mul_scratch);
            if step & 1 == 0 {
                product.sub_assign(&gcd);
            } else {
                product.add_assign(&gcd);
            }
            Division::div_exact_into(&product, tail, &mut coefficient, &mut scratch);
            coefficient
        };
        let (x_magnitude, y_magnitude) = if reversed {
            (final_t, final_s)
        } else {
            (final_s, final_t)
        };
        BezoutResult {
            gcd,
            x_magnitude,
            y_magnitude,
            x_is_positive: (step & 1 == 0) != reversed,
        }
    }
}

/// Reduces a shared-high-limb pair through its nonzero limb difference.
/// Returns `(g, |x|, s, x_positive, add_s)`, where `|y| = |x| +/- s`.
/// An inverse caller rejects g > 1 before constructing either cofactor;
/// the unused magnitude and scalar are then zero.
fn close_coefficients<const INVERSE: bool>(
    a: &InternalMpUint,
    b: &InternalMpUint,
) -> Option<(Limb, InternalMpUint, Limb, bool, bool)> {
    let left = a.limbs();
    let right = b.limbs();
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    // SAFETY: both initialized slices contain at least two limbs; their
    // high suffixes and low limbs exist independently of pointer width.
    let (low_a, low_b) = unsafe {
        if left.get_unchecked(1..) != right.get_unchecked(1..) {
            return None;
        }
        (*left.get_unchecked(0), *right.get_unchecked(0))
    };
    let difference = low_a.abs_diff(low_b);
    if difference == 0 {
        return None;
    }
    let remainder = Division::div_rem_1::<false>(right, difference, &mut InternalMpUint::zero());
    let (gcd, scalar, _, parity) = Division::extended_gcd_limb(remainder, difference);
    let add_scalar = low_a > low_b;
    let positive = (parity != 0) == add_scalar;
    if INVERSE && gcd != 1 {
        return Some((gcd, InternalMpUint::zero(), 0, positive, add_scalar));
    }

    // For d=|a-b|, scalar Euclid gives b*s-d*K=(-1)^parity*g.
    // K=(b*s-(-1)^parity*g)/d is exact. If a=b+d then |x|=K,
    // |y|=K+s; if a=b-d then |x|=K, |y|=K-s and x changes sign.
    // Since b >= B > d and g divides the positive b-d, K >= s; no signed
    // multiprecision intermediate or full-width Euclidean state is required.
    let coefficient = if scalar == 0 {
        // The first remainder is zero: g=d and the d coefficient is one.
        InternalMpUint::one()
    } else {
        let mut product = b.mul(&InternalMpUint::from_limb(scalar));
        let magnitude = InternalMpUint::from_limb(gcd);
        if parity == 0 {
            product.sub_assign(&magnitude);
        } else {
            product.add_assign(&magnitude);
        }
        let residual = Division::div_rem_1_assign(&mut product, difference);
        debug_assert_eq!(
            residual, 0,
            "the scalar Bezout identity proves divisibility"
        );
        product
    };
    Some((gcd, coefficient, scalar, positive, add_scalar))
}
/// Returns whether `value + 1 == modulus` without constructing `value + 1`.
///
/// The equal-length case propagates one carry through the value. The only
/// valid length-changing case is `modulus = B^n` and `value = B^n - 1`, where
/// `B = 2^LIMB_BITS`; normalized representations make those limb tests
/// sufficient on every supported pointer width.
fn is_modulus_predecessor(value: &InternalMpUint, modulus: &InternalMpUint) -> bool {
    let value_limbs = value.limbs();
    let modulus_limbs = modulus.limbs();
    debug_assert!(
        !value_limbs.is_empty() && !modulus_limbs.is_empty(),
        "inversion resolves zero operands"
    );

    if value_limbs.len() == modulus_limbs.len() {
        let mut carry = 1;
        for (&value_limb, &modulus_limb) in value_limbs.iter().zip(modulus_limbs) {
            let (sum, overflow) = value_limb.overflowing_add(carry);
            if sum != modulus_limb {
                return false;
            }
            carry = Limb::from(overflow);
        }
        return carry == 0;
    }

    // SAFETY: a materialized Limb slice occupies at most isize::MAX bytes
    // with at least two bytes per element; one additional limb fits usize.
    let extended_width = unsafe { value_limbs.len().unchecked_add(1) };
    if modulus_limbs.len() != extended_width {
        return false;
    }
    // SAFETY: the length equality proves `modulus_limbs` has one more limb
    // than `value_limbs`, hence it is nonempty and this last index exists.
    if unsafe { *modulus_limbs.get_unchecked(value_limbs.len()) } != 1 {
        return false;
    }

    modulus_limbs
        .iter()
        .take(value_limbs.len())
        .all(|&limb| limb == 0)
        && value_limbs.iter().all(|&limb| limb == Limb::MAX)
}
