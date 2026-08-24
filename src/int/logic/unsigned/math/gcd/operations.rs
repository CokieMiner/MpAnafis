//! GCD and LCM entry points owned by [`InternalMpUint`].

use core::{
    cmp::{Ordering, min},
    mem::swap,
};

use alloc::vec::Vec;

use super::{DivScratch, Division, InternalMpUint, Limb};

/// Namespace for the cross-file GCD algorithm surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gcd;

impl Gcd {
    pub const LEHMER_THRESHOLD: usize = 2;
    pub const WIDE_LEHMER_THRESHOLD: usize = 32;
}

impl InternalMpUint {
    /// Computes the greatest common divisor using Lehmer's algorithm for
    /// large numbers, and Stein's (binary GCD) for small numbers.
    ///
    /// Returns zero when both inputs are zero.
    ///
    /// # Panics
    ///
    /// May panic when internal invariants are violated (should not happen for
    /// well-formed inputs).
    #[allow(
        clippy::too_many_lines,
        reason = "Top-level GCD orchestration with all in-register fast paths."
    )]
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        let u_limbs = self.limbs();
        let v_limbs = other.limbs();
        if u_limbs.len() <= 4 && v_limbs.len() <= 4 {
            let g = Gcd::gcd_4(self.extract_4(), other.extract_4());
            return Self::from_limbs_4(g[0], g[1], g[2], g[3]);
        }

        if v_limbs.len() <= 4 {
            let mut rem = Self::zero();
            let mut scratch = DivScratch::default();
            Division::rem_into(self, other, &mut rem, &mut scratch);
            let g = Gcd::gcd_4(other.extract_4(), rem.extract_4());
            return Self::from_limbs_4(g[0], g[1], g[2], g[3]);
        }
        if u_limbs.len() <= 4 {
            let mut rem = Self::zero();
            let mut scratch = DivScratch::default();
            Division::rem_into(other, self, &mut rem, &mut scratch);
            let g = Gcd::gcd_4(self.extract_4(), rem.extract_4());
            return Self::from_limbs_4(g[0], g[1], g[2], g[3]);
        }

        let shift = min(self.trailing_zeros(), other.trailing_zeros());
        let mut u = self.clone();
        let mut v = other.clone();
        u.shr_assign(shift);
        v.shr_assign(shift);

        let mut rem = Self::zero();
        // DivScratch allocated lazily inside the branches that need it,
        // avoiding allocation on the common Lehmer-success path.
        let mut scratch: Option<DivScratch> = None;
        let mut u_backup: Vec<Limb> = Vec::new();
        let mut v_backup: Vec<Limb> = Vec::new();

        loop {
            if u.cmp(&v) == Ordering::Less {
                swap(&mut u, &mut v);
            }
            if v.is_zero() {
                break;
            }

            if v.limbs().len() <= 4 {
                let s = scratch.get_or_insert_with(DivScratch::default);
                Division::rem_into(&u, &v, &mut rem, s);
                let g = Gcd::gcd_4(v.extract_4(), rem.extract_4());
                let mut res = Self::from_limbs_4(g[0], g[1], g[2], g[3]);
                if shift > 0 {
                    res.shl_assign(shift);
                }
                return res;
            }

            if u.limbs().len() >= Gcd::HGCD_THRESHOLD && v.limbs().len() >= Gcd::HGCD_THRESHOLD {
                let prev_lens = (u.limbs().len(), v.limbs().len());
                let s = scratch.get_or_insert_with(DivScratch::default);
                Gcd::hgcd_reduce(&mut u, &mut v, s, &mut u_backup, &mut v_backup);
                if u.limbs().len() < prev_lens.0 || v.limbs().len() < prev_lens.1 {
                    continue;
                }
            }

            let (u0, v0, u1, v1, even) = if u.limbs().len() == v.limbs().len()
                && u.limbs().len() >= Gcd::WIDE_LEHMER_THRESHOLD
            {
                let (u_hat, v_hat) = Gcd::extract_top_two_limbs(u.limbs(), v.limbs());
                Gcd::lehmer_simulate_wide(u_hat, v_hat)
            } else {
                let (u_hat, v_hat) = Gcd::extract_top_limb(u.limbs(), v.limbs());
                Gcd::lehmer_simulate(u_hat, v_hat)
            };

            let is_identity = u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1;
            if is_identity {
                let s = scratch.get_or_insert_with(DivScratch::default);
                Division::rem_into(&u, &v, &mut rem, s);
                swap(&mut u, &mut v);
                swap(&mut v, &mut rem);
            } else {
                // lehmer_update saves/restores internally on failure,
                // reusing the u_backup and v_backup vectors.
                let ok = Gcd::lehmer_update(
                    &mut u,
                    &mut v,
                    &mut u_backup,
                    &mut v_backup,
                    u0,
                    v0,
                    u1,
                    v1,
                    even,
                );
                if !ok {
                    let s = scratch.get_or_insert_with(DivScratch::default);
                    Division::rem_into(&u, &v, &mut rem, s);
                    swap(&mut u, &mut v);
                    swap(&mut v, &mut rem);
                }
            }
        }

        if shift > 0 {
            u.shl_assign(shift);
        }
        u
    }

    /// Computes the least common multiple.
    #[must_use]
    pub fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let g = self.gcd(other);
        // lcm = self / gcd * other. The division is exact, so the remainder is
        // known to be zero and the quotient-only path is the right one.
        let q = self.div(&g);
        q.mul(other)
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
        let g = self.gcd(other);
        let q = self.div(&g);
        let l = q.mul(other);
        (g, l)
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
        let u_limbs = self.limbs();
        let v_limbs = other.limbs();
        if u_limbs.len() <= 4 && v_limbs.len() <= 4 {
            let g = Gcd::gcd_4(self.extract_4(), other.extract_4());
            return g[0] == 1 && g[1] == 0 && g[2] == 0 && g[3] == 0;
        }
        self.gcd(other).is_one()
    }

    /// Extended GCD returning `(gcd, x, y)` with unsigned residue
    /// representatives of the Bezout coefficients.
    ///
    /// For nonzero operands, `self*x` is congruent to `gcd` modulo `other`,
    /// `other*y` is congruent to `gcd` modulo `self`, `x < other`, and
    /// `y < self`. Recovering ordinary signed Bezout coefficients requires
    /// interpreting one of these residues as negative.
    ///
    /// `other` must be non-zero because the coefficient representatives are
    /// defined modulo it.
    #[must_use]
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        debug_assert!(
            !other.is_zero(),
            "internal extended GCD requires a non-zero second operand"
        );
        Division::extended_gcd(self, other)
    }
}
