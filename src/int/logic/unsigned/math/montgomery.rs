//! Montgomery reduction domain for modular arithmetic with odd moduli.

use core::{cmp::Ordering, ptr::eq};

use super::{ArchKernels, DivScratch, Division, InternalArbiUint, LIMB_BITS, Limb, MulScratch};

/// Montgomery reduction domain for modular arithmetic with odd moduli.
///
/// Stores precomputed constants for efficient Montgomery multiplication:
/// - `modulus`: the odd modulus M
/// - `m_inv`: `-M^{-1} mod 2^LIMB_BITS`
/// - `r2`: `R^2 mod M` where `R = 2^{LIMB_BITS*n}` and `n` is the number of limbs in M
#[derive(Clone, Debug)]
pub struct MontgomeryDomain {
    pub modulus: InternalArbiUint,
    pub m_inv: Limb,
    pub r2: InternalArbiUint,
}

impl MontgomeryDomain {
    /// Creates a new Montgomery domain for the given odd modulus.
    #[must_use]
    pub fn new(modulus: &InternalArbiUint) -> Self {
        let m_limbs = modulus.limbs();
        let m0 = m_limbs.first().copied().unwrap_or(0);
        debug_assert!(m0 & 1 == 1, "Montgomery modulus must be non-zero and odd");

        // Calculate m_inv = -M^{-1} mod 2^LIMB_BITS
        let mut inv = m0;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2_usize.wrapping_sub(m0.wrapping_mul(inv)));
        }
        let m_inv = 0_usize.wrapping_sub(inv);

        // Calculate R^2 mod M
        let n = m_limbs.len();
        let mut r2_uint = InternalArbiUint::one();
        let shift_amount = n.wrapping_mul(LIMB_BITS).wrapping_mul(2);
        r2_uint.shl_assign(shift_amount);

        let mut rem = InternalArbiUint::zero();
        let mut scratch = DivScratch::default();
        Division::rem_into(&r2_uint, modulus, &mut rem, &mut scratch);

        Self {
            modulus: modulus.clone(),
            m_inv,
            r2: rem,
        }
    }

    /// Performs Montgomery reduction: computes `t * R^{-1} mod M`.
    ///
    /// `t` must have at most `2n + 1` limbs where `n` is the number of limbs in the modulus.
    /// The result is written to `out`.
    #[allow(
        unsafe_code,
        reason = "The t_top branch proves 2*n < t_len; t is then resized to 2*n for i < n rows and n-limb tails, while out is resized to n+1 before accesses through index n."
    )]
    pub fn reduce_into(&self, t: &mut InternalArbiUint, out: &mut InternalArbiUint) {
        let m_limbs = self.modulus.limbs();
        let n = m_limbs.len();

        let t_len = t.limbs().len();
        debug_assert!(
            t_len <= n.wrapping_mul(2).wrapping_add(1),
            "reduce_into: input exceeds 2n+1 limbs"
        );
        let mut t_top = if t_len > n.wrapping_mul(2) {
            // SAFETY: this branch proves `2 * n < t_len = t.limbs().len()`.
            unsafe { *t.limbs().get_unchecked(n.wrapping_mul(2)) }
        } else {
            0
        };
        t.resize(n.wrapping_mul(2));
        let t_limbs = t.limbs_mut();

        // Runtime backend selection is process-stable, so resolve the cached
        // pointer once rather than rechecking its OnceLock for every reduction row.
        let add_mul = ArchKernels::selected_add_mul_limbs_unchecked();
        for i in 0..n {
            // SAFETY: i < n, t.len() is 2n
            let c = unsafe { *t_limbs.get_unchecked(i) }.wrapping_mul(self.m_inv);

            // SAFETY: i + n <= 2n; m_limbs and t_limbs[i..] are valid for n elements
            let carry = unsafe { add_mul(t_limbs.as_mut_ptr().add(i), m_limbs.as_ptr(), n, c) };

            let mut j = i.wrapping_add(n);
            let mut ripple = carry;
            while ripple > 0 {
                if j < t_limbs.len() {
                    // SAFETY: j < t_limbs.len()
                    let t_j = unsafe { *t_limbs.get_unchecked(j) };
                    let (sum, ov) = t_j.overflowing_add(ripple);
                    // SAFETY: j < t_limbs.len()
                    unsafe {
                        *t_limbs.get_unchecked_mut(j) = sum;
                    }
                    ripple = Limb::from(ov);
                    j = j.wrapping_add(1);
                } else {
                    t_top = t_top.wrapping_add(ripple);
                    break;
                }
            }
        }

        out.resize(n.wrapping_add(1));
        let out_limbs = out.limbs_mut();
        // SAFETY: t_limbs length was resized to 2n
        let src_slice = unsafe { t_limbs.get_unchecked(n..n.wrapping_mul(2)) };
        // SAFETY: `out` was resized to `n + 1`, so its `..n` prefix exists,
        // is initialized, and is disjoint from the `t` source.
        unsafe { out_limbs.get_unchecked_mut(..n) }.copy_from_slice(src_slice);
        // SAFETY: `out_limbs.len() = n + 1`, so index `n` is in bounds.
        unsafe {
            *out_limbs.get_unchecked_mut(n) = t_top;
        }
        out.normalize();

        if (*out).cmp(&self.modulus) != Ordering::Less {
            out.sub_assign(&self.modulus);
        }
    }

    pub fn square_into_with_scratch(
        &self,
        a: &InternalArbiUint,
        out: &mut InternalArbiUint,
        t: &mut InternalArbiUint,
        scratch: &mut MulScratch,
    ) {
        t.assign_square_with_scratch(a, scratch);
        self.reduce_into(t, out);
    }

    /// Performs a Montgomery multiplication utilizing pre-allocated scratch space.
    pub fn mul_into_with_scratch(
        &self,
        a: &InternalArbiUint,
        b: &InternalArbiUint,
        out: &mut InternalArbiUint,
        t: &mut InternalArbiUint,
        scratch: &mut MulScratch,
    ) {
        if eq(a, b) {
            self.square_into_with_scratch(a, out, t, scratch);
            return;
        }
        self.mul_into_with_scratch_internal(a, b, out, t, scratch);
    }

    #[allow(
        unsafe_code,
        reason = "In the CIOS branch out has n+1 limbs, b_slice and the modulus have n limbs, i is below n, and the n-1 access occurs only inside that nonempty loop; the final index n is within out."
    )]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::many_single_char_names,
        reason = "CIOS uses conventional a, b, i, and n names; each shifted DoubleLimb carry is reduced to its low Limb by construction."
    )]
    fn mul_into_with_scratch_internal(
        &self,
        a: &InternalArbiUint,
        b: &InternalArbiUint,
        out: &mut InternalArbiUint,
        t: &mut InternalArbiUint,
        scratch: &mut MulScratch,
    ) {
        let m_limbs = self.modulus.limbs();
        let n = m_limbs.len();

        let a_limbs = a.limbs();
        let a_len = a_limbs.len();
        let b_limbs = b.limbs();
        let b_len = b_limbs.len();

        // Use Coarsely Integrated Operand Scanning (CIOS) for small moduli (<= 32 limbs, i.e., 2048 bits).
        if n <= 32 {
            out.resize(n.wrapping_add(1));
            let out_limbs = out.limbs_mut();
            out_limbs.fill(0);

            let b_slice = if b_len >= n {
                // SAFETY: this branch proves `n <= b_len = b_limbs.len()`.
                unsafe { b_limbs.get_unchecked(..n) }
            } else {
                t.resize(n);
                let t_limbs = t.limbs_mut();
                t_limbs.fill(0);
                // SAFETY: `t_limbs.len() = n` after resize and this branch
                // proves `b_len < n`, so the destination prefix has length b_len.
                unsafe { t_limbs.get_unchecked_mut(..b_len) }.copy_from_slice(b_limbs);
                // SAFETY: `t_limbs` was resized to exactly `n` initialized
                // limbs above, so its full `..n` prefix exists.
                unsafe { t_limbs.get_unchecked(..n) }
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
                // SAFETY: out_limbs has length n + 1, b_slice and m_limbs have length >= n.
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
                    // SAFETY: n > 0, so n - 1 is within bounds of out_limbs of length n + 1
                    let out_top = unsafe { out_limbs.get_unchecked_mut(n.wrapping_sub(1)) };
                    let (sum, ov) = out_top.overflowing_add(c_prev);
                    *out_top = sum;
                    c_out = c_out.wrapping_add(Limb::from(ov));
                }
                c_prev = c_out;
            }
            // SAFETY: `out_limbs.len() = n + 1`, so index `n` is in bounds.
            unsafe {
                *out_limbs.get_unchecked_mut(n) = c_prev;
            }

            out.normalize();
            if (*out).cmp(&self.modulus) != Ordering::Less {
                out.sub_assign(&self.modulus);
            }
        } else {
            t.assign_product_with_scratch(a, b, scratch);
            self.reduce_into(t, out);
        }
    }

    #[must_use]
    pub fn transform_into_with_scratch(
        &self,
        a: &InternalArbiUint,
        temp_prod: &mut InternalArbiUint,
        scratch: &mut MulScratch,
    ) -> InternalArbiUint {
        let mut out = InternalArbiUint::zero();
        if a.limbs().len() > self.modulus.limbs().len() {
            // Reduce a modulo the modulus first so a * r2 fits in 2n+1 limbs.
            // (a mod N) * R mod N == a * R mod N, so this is correct.
            let mut div_scratch = DivScratch::default();
            let mut rem = InternalArbiUint::zero();
            Division::rem_into(a, &self.modulus, &mut rem, &mut div_scratch);
            self.mul_into_with_scratch(&rem, &self.r2, &mut out, temp_prod, scratch);
        } else {
            // a fits in the same number of limbs as the modulus, so a * r2
            // is at most 2n limbs, safely within the 2n+1 bound.
            self.mul_into_with_scratch(a, &self.r2, &mut out, temp_prod, scratch);
        }
        out
    }

    #[must_use]
    pub fn transform_out_with_scratch(
        &self,
        a: &InternalArbiUint,
        temp_prod: &mut InternalArbiUint,
        scratch: &mut MulScratch,
    ) -> InternalArbiUint {
        let one = InternalArbiUint::one();
        let mut out = InternalArbiUint::zero();
        self.mul_into_with_scratch(a, &one, &mut out, temp_prod, scratch);
        out
    }
}
