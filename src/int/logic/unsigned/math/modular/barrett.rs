//! Barrett reduction for arbitrary nonzero moduli.
//!
//! Reference: P. Barrett, "Implementing the Rivest Shamir and Adleman Public
//! Key Encryption Algorithm on a Standard Digital Signal Processor",
//! Advances in Cryptology - CRYPTO '86, LNCS 263, 311-323, 1987.
//! DOI: 10.1007/3-540-47721-7_24.

#![expect(
    unsafe_code,
    reason = "the bounded Barrett input supplies a nonempty quotient head, disjoint product spans, and fully initialized residue writes"
)]

use core::{cmp::Ordering, ptr::copy_nonoverlapping};

use super::{
    Addition, DivScratch, Division, HighProduct, InternalMpUint, LIMB_BITS, Limb, LowProduct,
    MulScratch, ScratchBuffer,
};

/// Barrett reduction domain for modular arithmetic.
///
/// Stores the modulus and reciprocal for Barrett reduction:
/// - `modulus`: the modulus M
/// - `mu`: `floor(b^{2k} / M)` where `b = 2^LIMB_BITS` and `k` is the number of limbs in M
#[derive(Clone, Debug)]
pub struct BarrettDomain {
    /// Modulus defining the reduction domain.
    pub modulus: InternalMpUint,
    /// Precomputed `floor(b^(2k) / modulus)` reciprocal.
    pub mu: InternalMpUint,
    /// Limb length of the modulus.
    pub k: usize,
    /// Zero-extended modulus used by bounded correction steps.
    pub modulus_pad: ScratchBuffer,
}

/// Reusable product storage for Barrett reduction and quotient reconstruction.
#[derive(Debug, Clone)]
pub struct BarrettScratch {
    residue_product: ScratchBuffer,
    cross: ScratchBuffer,
    quotient_product: ScratchBuffer,
    carry_product: ScratchBuffer,
}

impl Default for BarrettScratch {
    fn default() -> Self {
        Self {
            residue_product: ScratchBuffer::acquire(0),
            cross: ScratchBuffer::acquire(0),
            quotient_product: ScratchBuffer::acquire(0),
            carry_product: ScratchBuffer::acquire(0),
        }
    }
}

impl BarrettDomain {
    /// Creates a Barrett domain for a nonzero modulus.
    #[must_use]
    pub fn new(modulus: &InternalMpUint) -> Self {
        debug_assert!(!modulus.is_zero(), "Barrett modulus must be non-zero");
        let k = modulus.limbs().len();

        // An addressable k-limb modulus does not prove its doubled bit width
        // fits usize. Validate 2*k*LIMB_BITS before constructing B^(2k).
        let reciprocal_bits = k
            .checked_mul(2)
            .and_then(|limbs| limbs.checked_mul(LIMB_BITS))
            .expect("Barrett reciprocal width exceeds addressable memory");
        // SAFETY: reciprocal_bits proves 2*k*LIMB_BITS <= usize::MAX;
        // LIMB_BITS >= 16 leaves room for this one-limb extension.
        let padded_len = unsafe { k.unchecked_add(1) };

        // Calculate b^{2k}
        let mut b2k = InternalMpUint::one();
        b2k.shl_assign(reciprocal_bits);

        let mut mu = InternalMpUint::zero();
        let mut scratch = DivScratch::default();
        Division::div_into::<true, false>(&b2k, modulus, &mut mu, &mut scratch);
        let mut modulus_pad = ScratchBuffer::acquire(padded_len);
        // SAFETY: the pool reserves k+1 aligned limbs disjoint from modulus.
        // The copy and guard write initialize the complete span before set_len.
        unsafe {
            copy_nonoverlapping(modulus.limbs().as_ptr(), modulus_pad.as_mut_ptr(), k);
            modulus_pad.as_mut_ptr().add(k).write(0);
            modulus_pad.set_len(padded_len);
        }

        Self {
            modulus: modulus.clone(),
            mu,
            k,
            modulus_pad,
        }
    }

    /// Writes `t mod modulus`, reusing quotient and low-product storage.
    ///
    /// Inputs below `B^(2k)` use the reciprocal; wider inputs use division.
    pub fn reduce_into_with_barrett_scratch(
        &self,
        t: &InternalMpUint,
        out: &mut InternalMpUint,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &mut BarrettScratch,
    ) {
        let t_limbs = t.limbs();
        if t_limbs.len() <= self.k && t.cmp(&self.modulus) == Ordering::Less {
            out.clone_from(t);
            return;
        }
        if t_limbs.len() > self.k.saturating_mul(2) {
            let mut div_scratch = DivScratch::default();
            Division::rem_into(t, &self.modulus, out, &mut div_scratch);
            return;
        }

        let _ = self.approximate_remainder(t_limbs, out, mul_scratch, barrett_scratch);

        // The Barrett error bound guarantees at most two corrections
        // when the input is bounded by b^(2k).
        for _ in 0..2 {
            if out.limbs().iter().rev().cmp(self.modulus_pad.iter().rev()) == Ordering::Less {
                break;
            }
            let borrow = Addition::sub_slice_in_place(out.limbs_mut(), &self.modulus_pad);
            debug_assert_eq!(borrow, 0, "the correction subtracts an ordered modulus");
        }
        out.normalize();
        debug_assert_eq!(
            (*out).cmp(&self.modulus),
            Ordering::Less,
            "Barrett error exceeded bound"
        );
    }

    /// Divides a caller-proved input in `modulus..B^(2k)`.
    ///
    /// # Safety
    /// The domain must retain its constructor invariants. The caller must
    /// establish `modulus <= t < B^(2k)`, where `B = 2^LIMB_BITS`.
    /// Output integers and the two scratch owners must be disjoint from `t`.
    pub unsafe fn div_rem_bounded_unchecked(
        &self,
        t: &InternalMpUint,
        q_out: &mut InternalMpUint,
        r_out: &mut InternalMpUint,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &mut BarrettScratch,
    ) {
        debug_assert!(
            t >= &self.modulus,
            "bounded Barrett input is at least the modulus"
        );
        debug_assert!(
            t.limbs().len() <= self.k.saturating_mul(2),
            "bounded Barrett input is below B^(2k)"
        );
        let quotient = self.approximate_remainder(t.limbs(), r_out, mul_scratch, barrett_scratch);
        q_out.clone_from_slice(quotient);

        // The Barrett error bound guarantees at most two corrections
        // when the input is bounded by b^(2k).
        for _ in 0..2 {
            if r_out
                .limbs()
                .iter()
                .rev()
                .cmp(self.modulus_pad.iter().rev())
                == Ordering::Less
            {
                break;
            }
            let borrow = Addition::sub_slice_in_place(r_out.limbs_mut(), &self.modulus_pad);
            debug_assert_eq!(borrow, 0, "the correction subtracts an ordered modulus");
            q_out.increment();
        }
        r_out.normalize();
        debug_assert_eq!(
            (*r_out).cmp(&self.modulus),
            Ordering::Less,
            "Barrett error exceeded bound"
        );
    }

    /// Returns a normalized quotient estimate and writes
    /// `t - q*modulus mod B^(k+1)`, with `floor(t/modulus)-2 <= q
    /// <= floor(t/modulus)`. The entering input lies in `[modulus, B^(2k))`.
    /// The integer difference is nonnegative and below `3*modulus < B^(k+1)`,
    /// so its wrapped subtraction is also its exact integer representation.
    fn approximate_remainder<'scratch>(
        &self,
        t_limbs: &[Limb],
        out: &mut InternalMpUint,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &'scratch mut BarrettScratch,
    ) -> &'scratch [Limb] {
        let residue_len = self.modulus_pad.len();
        debug_assert!(
            t_limbs.len() >= self.k,
            "the entering input is at least the modulus"
        );
        // SAFETY: the constructor's nonzero modulus proves k>=1; t>=M
        // proves t.len()>=k. The borrowed quotient head is therefore nonempty.
        let head = unsafe { t_limbs.get_unchecked(self.k.unchecked_sub(1)..) };
        // mu>=B^k has at least k+1 limbs. Together with the nonempty head,
        // its product has more than residue_len=k+1 limbs. The high-product
        // kernel retains the exact carry across every omitted low diagonal.
        let mut quotient: &[Limb] = HighProduct::mul(
            head,
            self.mu.limbs(),
            residue_len,
            &mut barrett_scratch.quotient_product,
            &mut barrett_scratch.carry_product,
            mul_scratch,
        );
        while let Some((&0, lower)) = quotient.split_last() {
            quotient = lower;
        }

        let low_len = t_limbs.len().min(residue_len);
        let mut write = out.prepare_limb_write(residue_len);
        // SAFETY: the write guard reserves k+1 aligned limbs disjoint from t.
        // low_len<=t.len() bounds the copy; zero extension initializes the tail.
        let residue = unsafe {
            copy_nonoverlapping(t_limbs.as_ptr(), write.as_mut_ptr(), low_len);
            write
                .as_mut_ptr()
                .add(low_len)
                .write_bytes(0, residue_len.unchecked_sub(low_len));
            write.commit()
        };
        if quotient.is_empty() {
            return quotient;
        }
        // q<=floor(t/M)<B^(k+1), since t<B^(2k) and M>=B^(k-1).
        debug_assert!(
            quotient.len() <= residue_len,
            "Barrett's estimate is below B^(k+1)"
        );
        barrett_scratch
            .residue_product
            .reset_with_capacity(residue_len);
        // SAFETY: the estimate has 1..=k+1 initialized limbs and the modulus
        // has k>0 with a zero guard. Their owners are disjoint from the reserved
        // k+1 output limbs and every workspace. Both writers initialize all
        // output limbs before set_len. High-product carry storage is no longer
        // read and can hold the padded cross-product factor.
        unsafe {
            let product = barrett_scratch
                .residue_product
                .spare_capacity_mut()
                .get_unchecked_mut(..residue_len);
            if quotient.len() == residue_len {
                // A k+1-limb estimate already supplies the complete padded span.
                LowProduct::mul(
                    product,
                    quotient,
                    &self.modulus_pad,
                    residue_len,
                    mul_scratch,
                );
            } else {
                let _ = LowProduct::mul_with_guard(
                    self.modulus.limbs(),
                    quotient,
                    product,
                    &mut barrett_scratch.carry_product,
                    &mut barrett_scratch.cross,
                    mul_scratch,
                );
            }
            barrett_scratch.residue_product.set_len(residue_len);
        }
        let _ = Addition::sub_slice_in_place(residue, &barrett_scratch.residue_product);
        quotient
    }
}
