//! 2x2 integer reduction matrix for Subquadratic Half-GCD (HGCD).

use super::InternalMpUint;

/// A 2x2 transformation matrix used in the Half-GCD algorithm.
///
/// ```text
/// [ u0  v0 ]
/// [ u1  v1 ]
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HgcdMatrix {
    /// Element (0, 0)
    pub u0: InternalMpUint,
    /// Element (0, 1)
    pub v0: InternalMpUint,
    /// Element (1, 0)
    pub u1: InternalMpUint,
    /// Element (1, 1)
    pub v1: InternalMpUint,
}

impl HgcdMatrix {
    /// Constructs the 2x2 identity matrix:
    ///
    /// ```text
    /// [ 1  0 ]
    /// [ 0  1 ]
    /// ```
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            u0: InternalMpUint::one(),
            v0: InternalMpUint::zero(),
            u1: InternalMpUint::zero(),
            v1: InternalMpUint::one(),
        }
    }

    /// Checks if this matrix is the identity matrix.
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.u0.is_one() && self.v0.is_zero() && self.u1.is_zero() && self.v1.is_one()
    }

    /// Constructs a matrix from a single quotient `q`:
    ///
    /// ```text
    /// [ q  1 ]
    /// [ 1  0 ]
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_quotient(q: InternalMpUint) -> Self {
        Self {
            u0: q,
            v0: InternalMpUint::one(),
            u1: InternalMpUint::one(),
            v1: InternalMpUint::zero(),
        }
    }

    /// Multiplies this matrix `self` by another matrix `rhs`: `self * rhs`.
    ///
    /// ```text
    /// [ a00 a01 ] * [ b00 b01 ] = [ a00*b00 + a01*b10   a00*b01 + a01*b11 ]
    /// [ a10 a11 ]   [ b10 b11 ]   [ a10*b00 + a11*b10   a10*b01 + a11*b11 ]
    /// ```
    #[allow(
        clippy::similar_names,
        reason = "Matrix product entry variables follow mathematical notation."
    )]
    #[must_use]
    pub fn mul(&self, rhs: &Self) -> Self {
        if self.is_identity() {
            return rhs.clone();
        }
        if rhs.is_identity() {
            return self.clone();
        }

        let mut r00 = self.u0.mul(&rhs.u0);
        let p_v0_u1 = self.v0.mul(&rhs.u1);
        r00.add_assign(&p_v0_u1);

        let mut r01 = self.u0.mul(&rhs.v0);
        let p_v0_v1 = self.v0.mul(&rhs.v1);
        r01.add_assign(&p_v0_v1);

        let mut r10 = self.u1.mul(&rhs.u0);
        let p_v1_u1 = self.v1.mul(&rhs.u1);
        r10.add_assign(&p_v1_u1);

        let mut r11 = self.u1.mul(&rhs.v0);
        let p_v1_v1 = self.v1.mul(&rhs.v1);
        r11.add_assign(&p_v1_v1);

        Self {
            u0: r00,
            v0: r01,
            u1: r10,
            v1: r11,
        }
    }

    /// Applies the matrix inverse to a pair `(u, v)`:
    /// Computes `u' = |v1*u - v0*v|` and `v' = |u0*v - u1*u|`.
    ///
    /// Returns `true` if the reduction was successful and non-negative without divergence.
    #[allow(
        clippy::similar_names,
        reason = "Mathematical matrix entries u0, v0, u1, v1 follow standard algebraic notation."
    )]
    pub fn apply_to_pair(&self, u: &mut InternalMpUint, v: &mut InternalMpUint) -> bool {
        if self.is_identity() {
            return true;
        }

        let v1_u = self.v1.mul(u);
        let v0_v = self.v0.mul(v);
        let u0_v = self.u0.mul(v);
        let u1_u = self.u1.mul(u);

        let next_u = if v1_u >= v0_v {
            let mut diff = v1_u;
            diff.sub_assign(&v0_v);
            diff
        } else {
            let mut diff = v0_v;
            diff.sub_assign(&v1_u);
            diff
        };

        let next_v = if u0_v >= u1_u {
            let mut diff = u0_v;
            diff.sub_assign(&u1_u);
            diff
        } else {
            let mut diff = u1_u;
            diff.sub_assign(&u0_v);
            diff
        };

        // Safety check: both transformed numbers must be <= original u
        if next_u.cmp(u).is_gt() || next_v.cmp(u).is_gt() {
            return false;
        }

        *u = next_u;
        *v = next_v;
        true
    }
}
