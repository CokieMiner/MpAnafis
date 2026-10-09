//! GCD, LCM, and coprimality operations on [`InternalMpUint`].
//!
//! [`Gcd`] dispatches magnitude reduction; these methods restore common
//! powers of two and construct nonnegative least common multiples.

#![expect(
    unsafe_code,
    reason = "coprimality dispatch excludes zero operands before reading normalized low limbs"
)]

use core::{cmp::min, num::NonZeroUsize};

use super::{DivScratch, Division, Gcd, HGCD_CROSSOVER_THRESHOLD, HgcdWorkspace, InternalMpUint};

impl InternalMpUint {
    /// Computes the greatest common divisor using binary leaves, Lehmer
    /// batches, and recursive half-GCD above its empirical crossover.
    ///
    /// Returns zero when both inputs are zero.
    ///
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        // Both operands must contain a reducible high block. A shorter
        // divisor would immediately leave the recursive tier after its first
        // exact remainder, paying workspace setup without recursive progress.
        let common_len = min(self.limbs().len(), other.limbs().len());
        if common_len >= HGCD_CROSSOVER_THRESHOLD {
            let operand_len = self.limbs().len().max(other.limbs().len());
            HgcdWorkspace::with_thread_local(operand_len, |workspace| {
                Gcd::compute_half_gcd(self, other, workspace)
            })
        } else {
            Gcd::compute_lehmer(self, other)
        }
    }

    /// Computes the least common multiple.
    #[must_use]
    pub fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        if self.is_one() {
            return other.clone();
        }
        if other.is_one() {
            return self.clone();
        }
        let g = self.gcd(other);
        if g.is_one() {
            return self.mul(other);
        }
        if g == *self {
            return other.clone();
        }
        if g == *other {
            return self.clone();
        }
        // lcm(a,b) = (a/g)*b = a*(b/g). Dividing the shorter operand
        // reduces quotient work and the smaller multiplication dimension.
        let (shorter, longer) = if self.limbs().len() <= other.limbs().len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut quotient = Self::zero();
        Division::div_exact_into(shorter, &g, &mut quotient, &mut DivScratch::default());
        quotient.mul(longer)
    }

    /// Computes both the GCD and the LCM in a single pass.
    #[must_use]
    pub fn gcd_lcm(&self, other: &Self) -> (Self, Self) {
        if self.is_zero() {
            return (other.clone(), Self::zero());
        }
        if other.is_zero() {
            return (self.clone(), Self::zero());
        }
        if self.is_one() {
            return (Self::one(), other.clone());
        }
        if other.is_one() {
            return (Self::one(), self.clone());
        }
        let g = self.gcd(other);
        if g.is_one() {
            let l = self.mul(other);
            return (g, l);
        }
        if g == *self {
            return (g, other.clone());
        }
        if g == *other {
            return (g, self.clone());
        }
        let (shorter, longer) = if self.limbs().len() <= other.limbs().len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut quotient = Self::zero();
        Division::div_exact_into(shorter, &g, &mut quotient, &mut DivScratch::default());
        (g, quotient.mul(longer))
    }

    /// Returns `true` when `self` and `other` are coprime (i.e. `gcd == 1`).
    #[must_use]
    pub fn is_coprime(&self, other: &Self) -> bool {
        if self.is_zero() {
            return other.is_one();
        }
        if other.is_zero() {
            return self.is_one();
        }
        if self.is_one() || other.is_one() {
            return true;
        }
        // SAFETY: both zero cases returned above, and every normalized nonzero
        // integer has an initialized low limb.
        let (u0, v0) = unsafe {
            (
                *self.limbs().get_unchecked(0),
                *other.limbs().get_unchecked(0),
            )
        };
        // If both are even, gcd >= 2, so they cannot be coprime.
        if (u0 & 1 == 0) && (v0 & 1 == 0) {
            return false;
        }
        let u_len = self.limbs().len();
        let v_len = other.limbs().len();
        if u_len == 1 && v_len == 1 {
            return Gcd::gcd_1(u0, v0) == 1;
        }
        if u_len == 1 {
            // SAFETY: the nonzero one-limb operand stays positive after
            // removing its trailing zero bits.
            let divisor = unsafe { NonZeroUsize::new_unchecked(u0 >> u0.trailing_zeros()) };
            if divisor.get() == 1 {
                return true;
            }
            return Gcd::gcd_odd_limb(other.limbs(), divisor) == 1;
        }
        if v_len == 1 {
            // SAFETY: the nonzero one-limb operand stays positive after
            // removing its trailing zero bits.
            let divisor = unsafe { NonZeroUsize::new_unchecked(v0 >> v0.trailing_zeros()) };
            if divisor.get() == 1 {
                return true;
            }
            return Gcd::gcd_odd_limb(self.limbs(), divisor) == 1;
        }
        if u_len == 2 && v_len == 2 {
            // SAFETY: the exact two-limb widths prove both high limbs exist.
            let (u1, v1) = unsafe {
                (
                    *self.limbs().get_unchecked(1),
                    *other.limbs().get_unchecked(1),
                )
            };
            let [low, high] = Gcd::gcd_2([u0, u1], [v0, v1]);
            return low == 1 && high == 0;
        }
        self.gcd(other).is_one()
    }
}
