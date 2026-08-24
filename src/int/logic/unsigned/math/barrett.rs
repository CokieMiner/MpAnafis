//! Barrett reduction domain for modular arithmetic with arbitrary moduli.

use core::{
    cmp::{Ordering, min},
    ptr::eq,
};

use super::{
    DivScratch, Division, InternalMpUint, LIMB_BITS, Limb, LowProduct, MulScratch, ScratchBuffer,
};

/// Barrett reduction domain for modular arithmetic.
///
/// Stores precomputed constants for efficient Barrett reduction:
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

#[derive(Debug, Clone)]
pub struct BarrettScratch {
    q1: InternalMpUint,
    q2: InternalMpUint,
    q3: InternalMpUint,
    r2_buf: InternalMpUint,
    r1: InternalMpUint,
    q3_pad: ScratchBuffer,
}

impl Default for BarrettScratch {
    fn default() -> Self {
        Self {
            q1: InternalMpUint::zero(),
            q2: InternalMpUint::zero(),
            q3: InternalMpUint::zero(),
            r2_buf: InternalMpUint::zero(),
            r1: InternalMpUint::zero(),
            q3_pad: ScratchBuffer::acquire(0),
        }
    }
}

impl BarrettDomain {
    /// Creates a new Barrett domain for the given modulus.
    #[must_use]
    #[allow(
        unsafe_code,
        reason = "The modulus pad is resized to k + 1 immediately before taking its first k elements."
    )]
    pub fn new(modulus: &InternalMpUint) -> Self {
        debug_assert!(!modulus.is_zero(), "Barrett modulus must be non-zero");
        let k = modulus.limbs().len();

        // Calculate b^{2k}
        let mut b2k = InternalMpUint::one();
        b2k.shl_assign(k.wrapping_mul(2).wrapping_mul(LIMB_BITS));

        let mut mu = InternalMpUint::zero();
        let mut rem = InternalMpUint::zero();
        let mut scratch = DivScratch::default();
        Division::div_rem_into(&b2k, modulus, &mut mu, &mut rem, &mut scratch);
        let mut modulus_pad = ScratchBuffer::acquire(k.wrapping_add(1));
        modulus_pad.resize(k.wrapping_add(1), 0);
        // SAFETY: resize initialized k + 1 elements, so the first k exist.
        let destination = unsafe { modulus_pad.get_unchecked_mut(..k) };
        destination.copy_from_slice(modulus.limbs());

        Self {
            modulus: modulus.clone(),
            mu,
            k,
            modulus_pad,
        }
    }

    /// Performs Barrett reduction utilizing reusable Barrett and multiplication scratch.
    #[allow(
        clippy::similar_names,
        reason = "shift_limbs and shift2_limbs are the conventional Barrett quotient and residue shifts"
    )]
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
        if t_limbs.len() > self.k.wrapping_mul(2) {
            let mut div_scratch = DivScratch::default();
            Division::rem_into(t, &self.modulus, out, &mut div_scratch);
            return;
        }

        // q1 = t >> (k - 1) limbs
        let shift_limbs = self.k.saturating_sub(1);
        let shift2_limbs = self.k.wrapping_add(1);

        self.prepare_barrett_quotient(
            t_limbs,
            shift_limbs,
            shift2_limbs,
            mul_scratch,
            barrett_scratch,
        );
        self.prepare_barrett_residue(shift2_limbs, mul_scratch, barrett_scratch);
        prepare_barrett_low_part(t_limbs, shift2_limbs, barrett_scratch);
        reduce_barrett_difference(shift2_limbs, barrett_scratch);

        // The Barrett error bound guarantees at most two corrections
        // when the input is bounded by b^(2k).
        for _ in 0..2 {
            if barrett_scratch.r1.cmp(&self.modulus) == Ordering::Less {
                break;
            }
            barrett_scratch.r1.sub_assign(&self.modulus);
        }
        debug_assert_eq!(
            barrett_scratch.r1.cmp(&self.modulus),
            Ordering::Less,
            "Barrett error exceeded bound"
        );

        out.clone_from(&barrett_scratch.r1);
    }

    /// Performs Barrett division utilizing reusable Barrett and multiplication scratch.
    ///
    /// Computes both the quotient and remainder of `t / modulus`, placing them
    /// into `q_out` and `r_out` respectively.
    #[allow(
        clippy::similar_names,
        reason = "shift_limbs and shift2_limbs are the conventional Barrett quotient and residue shifts"
    )]
    pub fn div_rem_into_with_barrett_scratch(
        &self,
        t: &InternalMpUint,
        q_out: &mut InternalMpUint,
        r_out: &mut InternalMpUint,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &mut BarrettScratch,
    ) {
        let t_limbs = t.limbs();
        if t_limbs.len() <= self.k && t.cmp(&self.modulus) == Ordering::Less {
            q_out.clear();
            r_out.clone_from(t);
            return;
        }
        if t_limbs.len() > self.k.wrapping_mul(2) {
            let mut div_scratch = DivScratch::default();
            Division::div_rem_into(t, &self.modulus, q_out, r_out, &mut div_scratch);
            return;
        }

        // q1 = t >> (k - 1) limbs
        let shift_limbs = self.k.saturating_sub(1);
        let shift2_limbs = self.k.wrapping_add(1);

        self.prepare_barrett_quotient(
            t_limbs,
            shift_limbs,
            shift2_limbs,
            mul_scratch,
            barrett_scratch,
        );
        self.prepare_barrett_residue(shift2_limbs, mul_scratch, barrett_scratch);
        prepare_barrett_low_part(t_limbs, shift2_limbs, barrett_scratch);
        reduce_barrett_difference(shift2_limbs, barrett_scratch);

        // The Barrett error bound guarantees at most two corrections
        // when the input is bounded by b^(2k).
        for _ in 0..2 {
            if barrett_scratch.r1.cmp(&self.modulus) == Ordering::Less {
                break;
            }
            barrett_scratch.r1.sub_assign(&self.modulus);
            barrett_scratch.q3.increment();
        }
        debug_assert_eq!(
            barrett_scratch.r1.cmp(&self.modulus),
            Ordering::Less,
            "Barrett error exceeded bound"
        );

        q_out.clone_from(&barrett_scratch.q3);
        r_out.clone_from(&barrett_scratch.r1);
    }

    /// Performs `(a * b) mod M` into `out` with reusable Barrett scratch.
    pub fn mul_into_with_barrett_scratch(
        &self,
        a: &InternalMpUint,
        b: &InternalMpUint,
        out: &mut InternalMpUint,
        temp_prod: &mut InternalMpUint,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &mut BarrettScratch,
    ) {
        if eq(a, b) {
            self.square_into_with_barrett_scratch(a, out, temp_prod, mul_scratch, barrett_scratch);
            return;
        }
        temp_prod.assign_product_with_scratch(a, b, mul_scratch);
        self.reduce_into_with_barrett_scratch(temp_prod, out, mul_scratch, barrett_scratch);
    }

    /// Performs `(a * a) mod M` into `out` with reusable Barrett scratch.
    pub fn square_into_with_barrett_scratch(
        &self,
        a: &InternalMpUint,
        out: &mut InternalMpUint,
        temp_prod: &mut InternalMpUint,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &mut BarrettScratch,
    ) {
        temp_prod.assign_square_with_scratch(a, mul_scratch);
        self.reduce_into_with_barrett_scratch(temp_prod, out, mul_scratch, barrett_scratch);
    }

    #[allow(
        unsafe_code,
        reason = "Each source suffix starts below its slice length by the guarding branch; q1 and q3 are resized to exactly the corresponding suffix length before copying."
    )]
    fn prepare_barrett_quotient(
        &self,
        t_limbs: &[Limb],
        quotient_shift_limbs: usize,
        residue_limbs: usize,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &mut BarrettScratch,
    ) {
        barrett_scratch.q1.clear();
        if t_limbs.len() > quotient_shift_limbs {
            barrett_scratch
                .q1
                .resize(t_limbs.len().wrapping_sub(quotient_shift_limbs));
            // SAFETY: the branch proves `quotient_shift_limbs < t_limbs.len()`,
            // so this suffix contains exactly the number of initialized limbs
            // to which `q1` was resized.
            let quotient_source = unsafe { t_limbs.get_unchecked(quotient_shift_limbs..) };
            barrett_scratch
                .q1
                .limbs_mut()
                .copy_from_slice(quotient_source);
            barrett_scratch.q1.normalize();
        }

        barrett_scratch
            .q2
            .assign_product_with_scratch(&barrett_scratch.q1, &self.mu, mul_scratch);

        // q3 = q2 >> (k + 1) limbs
        barrett_scratch.q3.clear();
        let q2_limbs = barrett_scratch.q2.limbs();
        if q2_limbs.len() > residue_limbs {
            barrett_scratch
                .q3
                .resize(q2_limbs.len().wrapping_sub(residue_limbs));
            // SAFETY: the branch proves `residue_limbs < q2_limbs.len()`, so
            // this suffix has exactly the initialized length assigned to `q3`.
            let quotient_source = unsafe { q2_limbs.get_unchecked(residue_limbs..) };
            barrett_scratch
                .q3
                .limbs_mut()
                .copy_from_slice(quotient_source);
            barrett_scratch.q3.normalize();
        }
    }

    fn prepare_barrett_residue(
        &self,
        shift2_limbs: usize,
        mul_scratch: &mut MulScratch,
        barrett_scratch: &mut BarrettScratch,
    ) {
        // P5 fix: r2 = (q3 * modulus) mod b^{k+1} — only need low k+1 limbs.
        // Use mullo_slice to avoid computing the full product.
        let q3_limbs = barrett_scratch.q3.limbs();
        let mullo_len = shift2_limbs;

        barrett_scratch.r2_buf.clear();
        if !q3_limbs.is_empty() && mullo_len > 0 {
            barrett_scratch.r2_buf.resize(mullo_len);
            let r2_sl = barrett_scratch.r2_buf.limbs_mut();

            // Pad q3 and modulus to mullo_len for mullo_slice
            barrett_scratch.q3_pad.clear();
            barrett_scratch.q3_pad.resize(mullo_len, 0);
            let copy_q3 = min(q3_limbs.len(), mullo_len);
            if let (Some(dst), Some(src)) = (
                barrett_scratch.q3_pad.get_mut(..copy_q3),
                q3_limbs.get(..copy_q3),
            ) {
                dst.copy_from_slice(src);
            }

            LowProduct::mul(
                r2_sl,
                &barrett_scratch.q3_pad,
                &self.modulus_pad,
                mullo_len,
                mul_scratch,
            );
            barrett_scratch.r2_buf.normalize();
        }
    }
}

#[allow(
    unsafe_code,
    reason = "r1_len = min(t_limbs.len(), shift2_limbs), and r1 is resized to exactly r1_len before copying that source prefix."
)]
fn prepare_barrett_low_part(
    t_limbs: &[Limb],
    shift2_limbs: usize,
    barrett_scratch: &mut BarrettScratch,
) {
    let r1_len = min(t_limbs.len(), shift2_limbs);
    barrett_scratch.r1.clear();
    if r1_len > 0 {
        barrett_scratch.r1.resize(r1_len);
        // SAFETY: `r1_len = min(t_limbs.len(), shift2_limbs)`, so the prefix
        // is within the initialized source and matches the resized destination.
        let low_source = unsafe { t_limbs.get_unchecked(..r1_len) };
        barrett_scratch.r1.limbs_mut().copy_from_slice(low_source);
        barrett_scratch.r1.normalize();
    }
}

fn reduce_barrett_difference(shift2_limbs: usize, barrett_scratch: &mut BarrettScratch) {
    if barrett_scratch.r1.cmp(&barrett_scratch.r2_buf) == Ordering::Less {
        wrap_sub_barrett_residue(shift2_limbs, barrett_scratch);
    } else {
        barrett_scratch.r1.sub_assign(&barrett_scratch.r2_buf);
    }
}

#[allow(
    unsafe_code,
    reason = "r1 is resized to shift2_limbs before a loop over i < shift2_limbs; r2 is accessed only under i < r2_limbs.len()."
)]
fn wrap_sub_barrett_residue(shift2_limbs: usize, barrett_scratch: &mut BarrettScratch) {
    // Underflow means r1 - r2 is negative in integers, but the Barrett
    // algorithm needs it modulo b^{k+1}. Because r1 and r2 are both
    // already truncated to k+1 limbs, adding b^{k+1} is exactly equivalent
    // to computing (r1 - r2) mod b^{k+1} via k+1-limb wrapping subtraction.
    let r2_limbs = barrett_scratch.r2_buf.limbs();
    barrett_scratch.r1.resize(shift2_limbs);
    let r1_limbs = barrett_scratch.r1.limbs_mut();
    // Perform multi-limb subtraction (r1 - r2) mod b^{k+1}, discarding any final borrow at shift2_limbs.
    let mut borrow: Limb = 0;
    for i in 0..shift2_limbs {
        let r2_val = if i < r2_limbs.len() {
            // SAFETY: the branch proves `i < r2_limbs.len()`.
            unsafe { *r2_limbs.get_unchecked(i) }
        } else {
            0
        };
        // SAFETY: `r1` was resized to `shift2_limbs`, and the loop proves
        // `i < shift2_limbs = r1_limbs.len()`.
        let r1_val = unsafe { *r1_limbs.get_unchecked(i) };
        // Compute (r1_val - r2_val - borrow) modulo 2^LIMB_BITS
        let (diff, borrow1) = r1_val.overflowing_sub(r2_val);
        let (diff2, borrow2) = diff.overflowing_sub(borrow);
        borrow = Limb::from(borrow1 || borrow2);
        // SAFETY: as above, `i < shift2_limbs = r1_limbs.len()`; this mutable
        // access is exclusive for the current loop iteration.
        unsafe {
            *r1_limbs.get_unchecked_mut(i) = diff2;
        }
    }
    barrett_scratch.r1.normalize();
}
