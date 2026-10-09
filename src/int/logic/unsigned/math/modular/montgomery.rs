//! Montgomery domains and reduction for positive odd moduli.
//!
//! References:
//! - P. L. Montgomery, "Modular Multiplication Without Trial Division",
//!   Mathematics of Computation 44(170), 519-521, 1985.
//!   DOI: 10.1090/S0025-5718-1985-0777282-X.
//! - Ç. K. Koç, T. Acar, and B. S. Kaliski Jr., "Analyzing and Comparing
//!   Montgomery Multiplication Algorithms", IEEE Micro 16(3), 26-33, 1996.
//!   DOI: 10.1109/40.502403. Small-modulus multiplication uses CIOS;
//!   large-modulus reduction forms the radix-inverse product separately.

#![expect(
    unsafe_code,
    reason = "Odd moduli and materialized limb widths bound initialized carry slots and disjoint row spans"
)]

use core::{cmp::Ordering, ptr::eq};

use super::{
    ArchKernels, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb, MONTGOMERY_CIOS_MAX_LIMBS,
    MontgomeryScratch, ScratchBuffer,
};

/// Montgomery reduction domain for modular arithmetic with odd moduli.
///
/// Stores the modulus and constants for Montgomery multiplication:
/// - `modulus`: the odd modulus M
/// - `m_inv`: `-M^{-1} mod 2^LIMB_BITS`
/// - `inverse`: `M^{-1} mod R` for product-based reduction
/// - `r2`: `R^2 mod M` when encoding is requested at construction
///
/// Here `R = 2^{LIMB_BITS*n}` and `n` is the number of limbs in M.
#[derive(Clone, Debug)]
pub struct MontgomeryDomain {
    pub modulus: InternalMpUint,
    pub m_inv: Limb,
    pub inverse: ScratchBuffer,
    pub r2: InternalMpUint,
}

impl MontgomeryDomain {
    /// Prepares reduction modulo a nonzero odd modulus.
    ///
    /// `ENCODE` also prepares `R^2 mod M` for transformations and powers.
    /// A raw Montgomery product uses `ENCODE=false` and skips that division.
    ///
    /// # Panics
    /// Panics when encoding is requested and the radix-square bit width
    /// cannot be represented by `usize`.
    #[must_use]
    pub fn new<const ENCODE: bool>(modulus: &InternalMpUint) -> Self {
        let m_limbs = modulus.limbs();
        debug_assert!(
            modulus.is_odd(),
            "Montgomery modulus must be non-zero and odd"
        );
        // SAFETY: callers establish a positive odd modulus. Its normalized
        // magnitude contains at least one initialized limb.
        let m0 = unsafe { *m_limbs.get_unchecked(0) };
        let n = m_limbs.len();
        // Only encoding constructs R^2. An addressable modulus can have
        // an unrepresentable doubled bit width on narrow pointer targets.
        let shift_amount = if ENCODE {
            n.checked_mul(LIMB_BITS)
                .and_then(|bits| bits.checked_mul(2))
                .expect("Montgomery radix width exceeds addressable memory")
        } else {
            0
        };
        let inverse_limb = Division::modular_inverse_limb(m0);
        // The cancellation coefficient is the negative inverse modulo B.
        let m_inv = inverse_limb.wrapping_neg();
        let inverse = if n > MONTGOMERY_CIOS_MAX_LIMBS {
            Self::inverse_mod_radix(m_limbs, inverse_limb)
        } else {
            ScratchBuffer::acquire(0)
        };
        let mut r2 = InternalMpUint::zero();
        if ENCODE {
            let mut radix_square = InternalMpUint::one();
            radix_square.shl_assign(shift_amount);
            Division::rem_into(&radix_square, modulus, &mut r2, &mut DivScratch::default());
        }

        Self {
            modulus: modulus.clone(),
            m_inv,
            inverse,
            r2,
        }
    }

    /// Returns `a*R mod M`; construction must request `ENCODE=true`.
    #[must_use]
    pub fn transform_into_with_scratch(
        &self,
        a: &InternalMpUint,
        temp_prod: &mut InternalMpUint,
        scratch: &mut MontgomeryScratch,
    ) -> InternalMpUint {
        let mut out = InternalMpUint::zero();
        if a.limbs().len() > self.modulus.limbs().len() {
            // Reducing a establishes a < R, hence a * r2 < R * M.
            let mut div_scratch = DivScratch::default();
            let mut rem = InternalMpUint::zero();
            Division::rem_into(a, &self.modulus, &mut rem, &mut div_scratch);
            self.mul_into_with_scratch(&rem, &self.r2, &mut out, temp_prod, scratch);
        } else {
            // Equal or shorter limb width establishes a < R; r2 < M.
            self.mul_into_with_scratch(a, &self.r2, &mut out, temp_prod, scratch);
        }
        out
    }

    /// Squares a Montgomery residue using reusable product and reduction storage.
    #[inline]
    pub fn square_into_with_scratch(
        &self,
        a: &InternalMpUint,
        out: &mut InternalMpUint,
        t: &mut InternalMpUint,
        scratch: &mut MontgomeryScratch,
    ) {
        t.assign_square_with_scratch(a, &mut scratch.multiplication);
        self.reduce_into(t, out, scratch);
    }

    /// Multiplies Montgomery residues using reusable product and reduction storage.
    pub fn mul_into_with_scratch(
        &self,
        a: &InternalMpUint,
        b: &InternalMpUint,
        out: &mut InternalMpUint,
        t: &mut InternalMpUint,
        scratch: &mut MontgomeryScratch,
    ) {
        if eq(a, b) {
            self.square_into_with_scratch(a, out, t, scratch);
            return;
        }
        let m_limbs = self.modulus.limbs();
        let n = m_limbs.len();
        let a_limbs = a.limbs();
        let a_len = a_limbs.len();
        let b_limbs = b.limbs();
        let b_len = b_limbs.len();

        // Coarsely Integrated Operand Scanning bounds scratch to n+1 limbs.
        if self.inverse.is_empty() {
            // SAFETY: the positive modulus occupies at most isize::MAX
            // bytes, with at least two bytes per limb. Its n+1 count fits usize.
            out.resize(unsafe { n.unchecked_add(1) });
            let out_limbs = out.limbs_mut();
            out_limbs.fill(0);

            let b_slice = if b_len >= n {
                // SAFETY: this branch proves n <= b_len, so the immutable
                // initialized prefix exists and cannot overlap the output.
                unsafe { b_limbs.get_unchecked(..n) }
            } else {
                t.resize(n);
                // SAFETY: this branch has b_len<n, and resize initialized n
                // limbs; the copied prefix and zero padding are disjoint.
                let (prefix, padding) = unsafe { t.limbs_mut().split_at_mut_unchecked(b_len) };
                prefix.copy_from_slice(b_limbs);
                padding.fill(0);
                t.limbs()
            };

            let mut c_prev: Limb = 0;
            let monty_step = ArchKernels::selected_monty_redc_step_unchecked();
            for i in 0..n {
                let a_i = if i < a_len {
                    // SAFETY: i < a_len guarantees an initialized limb.
                    unsafe { *a_limbs.get_unchecked(i) }
                } else {
                    0
                };
                // SAFETY: out_limbs has n+1 initialized limbs; b_slice and
                // m_limbs have n initialized limbs. The mutable output is
                // disjoint from both immutable inputs, and the selected
                // backend satisfies the current target prerequisites.
                let mut c_out = unsafe {
                    monty_step(
                        out_limbs.as_mut_ptr(),
                        b_slice.as_ptr(),
                        m_limbs.as_ptr(),
                        n,
                        a_i,
                        self.m_inv,
                    )
                };
                if c_prev != 0 {
                    // SAFETY: the nonzero modulus establishes n > 0;
                    // n-1 is within the initialized n+1-limb output.
                    let out_top = unsafe { out_limbs.get_unchecked_mut(n.unchecked_sub(1)) };
                    let (sum, ov) = out_top.overflowing_add(c_prev);
                    *out_top = sum;
                    // SAFETY: the kernel returns a binary overflow and ov
                    // is a binary addition carry; their sum is at most two.
                    c_out = unsafe { c_out.unchecked_add(Limb::from(ov)) };
                }
                c_prev = c_out;
            }
            // SAFETY: out_limbs has n+1 initialized limbs; n is in bounds
            // and this exclusive write cannot alias either input.
            unsafe {
                *out_limbs.get_unchecked_mut(n) = c_prev;
            }
            out.normalize();
            if (*out).cmp(&self.modulus) != Ordering::Less {
                out.sub_assign(&self.modulus);
            }
        } else {
            t.assign_product_with_scratch(a, b, &mut scratch.multiplication);
            self.reduce_into(t, out, scratch);
        }
    }
}

/// Montgomery arithmetic modulo one odd limb `n > 1`.
///
/// A residue `x` is stored as `x*B mod n`, with `B = 2^LIMB_BITS`. Every
/// operation consumes and returns canonical representatives in `[0, n)`.
/// Construction costs two hardware divisions; each product costs three limb
/// multiplications and no division. Any limb `x` enters the domain as
/// `multiply(x, radix_square)`: `x < B` and `radix_square < n` meet the
/// product bound.
#[derive(Clone, Copy, Debug)]
pub struct LimbMontgomery {
    pub modulus: Limb,
    /// `n^-1 mod B`.
    pub inverse: Limb,
    /// `B mod n`, the representation of one.
    pub one: Limb,
    /// `B^2 mod n`.
    pub radix_square: Limb,
}

impl LimbMontgomery {
    /// Prepares arithmetic modulo an odd limb greater than one.
    #[must_use]
    pub fn new(modulus: Limb) -> Self {
        debug_assert!(
            modulus > 1 && modulus & 1 != 0,
            "limb Montgomery arithmetic requires an odd modulus above one"
        );
        // SAFETY: n > 1 bounds the high limb of B = (1, 0). The first
        // remainder B mod n is below n and bounds the high limb of B^2 mod n.
        let (one, radix_square) = unsafe {
            let (_, one) = ArchKernels::divrem_1_unchecked(0, 1, modulus);
            let (_, square) = ArchKernels::divrem_1_unchecked(0, one, modulus);
            (one, square)
        };
        Self {
            modulus,
            inverse: Division::modular_inverse_limb(modulus),
            one,
            radix_square,
        }
    }

    /// Returns `left*right/B mod n` for `left*right < n*B`.
    ///
    /// With `m = low*n^-1 mod B`, the product `m*n` has the same low limb as
    /// `left*right`, so their difference is exactly `(high - m*n/B)*B`. Both
    /// high parts are below `n`, and one conditional addition restores the
    /// canonical range.
    #[must_use]
    #[inline]
    pub const fn multiply(&self, left: Limb, right: Limb) -> Limb {
        let (low, high) = ArchKernels::mul_limb_lo_hi(left, right);
        let (_, correction) =
            ArchKernels::mul_limb_lo_hi(low.wrapping_mul(self.inverse), self.modulus);
        let (value, borrow) = high.overflowing_sub(correction);
        if borrow {
            value.wrapping_add(self.modulus)
        } else {
            value
        }
    }

    /// Adds two canonical residues.
    #[must_use]
    #[inline]
    pub const fn add(&self, left: Limb, right: Limb) -> Limb {
        let (sum, carry) = left.overflowing_add(right);
        if carry || sum >= self.modulus {
            sum.wrapping_sub(self.modulus)
        } else {
            sum
        }
    }

    /// Subtracts two canonical residues.
    #[must_use]
    #[inline]
    pub const fn subtract(&self, left: Limb, right: Limb) -> Limb {
        let (difference, borrow) = left.overflowing_sub(right);
        if borrow {
            difference.wrapping_add(self.modulus)
        } else {
            difference
        }
    }
}
