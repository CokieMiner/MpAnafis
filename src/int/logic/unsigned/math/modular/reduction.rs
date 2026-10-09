//! Product-based Montgomery reduction and radix-inverse refinement.
//!
//! For `R=B^n`, `I=M^-1 mod R` and `q=t_low*I mod R`, `qM` and `t` have
//! identical low halves. Thus `(t-qM)/R=t_high-floor(qM/R)` lies in
//! `(-M,M)` when `0<=t<MR`. One addition of `M` after a borrow is sufficient.

#![expect(
    unsafe_code,
    reason = "Odd-domain construction and materialized modulus widths establish fixed product spans, initialized outputs and disjoint refinement partitions"
)]

use core::cmp::Ordering;

#[cfg(not(target_pointer_width = "16"))]
use super::Multiplication;
use super::{
    Addition, ArchKernels, HighProduct, InternalMpUint, Limb, LowProduct, MontgomeryDomain,
    MulScratch, ScratchBuffer,
};

/// Reusable coefficient, high-product and multiplication storage.
#[derive(Clone, Debug)]
pub struct MontgomeryScratch {
    pub coefficients: ScratchBuffer,
    pub high_product: ScratchBuffer,
    pub carry_product: ScratchBuffer,
    pub multiplication: MulScratch,
}

impl Default for MontgomeryScratch {
    fn default() -> Self {
        Self {
            coefficients: ScratchBuffer::acquire(0),
            high_product: ScratchBuffer::acquire(0),
            carry_product: ScratchBuffer::acquire(0),
            multiplication: MulScratch::default(),
        }
    }
}

impl MontgomeryDomain {
    /// Writes `t/R mod M` for `0<=t<M*R`, where `R=B^n`.
    ///
    /// Small domains use scalar cancellation. Wider domains form q modulo R
    /// and retain only the exact high half of qM. The low halves cancel, so
    /// correction depends solely on the high subtraction's borrow.
    pub fn reduce_into(
        &self,
        t: &mut InternalMpUint,
        out: &mut InternalMpUint,
        scratch: &mut MontgomeryScratch,
    ) {
        if t.is_zero() {
            out.clear();
            return;
        }
        let modulus = self.modulus.limbs();
        let n = modulus.len();
        if self.inverse.is_empty() {
            self.reduce_scalar(t, out);
            return;
        }
        debug_assert_eq!(self.inverse.len(), n, "the radix inverse spans the modulus");
        // SAFETY: the materialized modulus spans at most isize::MAX bytes
        // with at least two bytes per limb, so 2*n fits usize.
        // A bounded input t<M*R<R^2 has no nonzero limb beyond 2n.
        let double_n = unsafe { n.unchecked_mul(2) };
        debug_assert!(
            t.limbs().get(double_n).is_none_or(|&guard| guard == 0),
            "the bounded input has no high radix guard"
        );
        t.resize(double_n);
        // SAFETY: resize supplies 2n initialized limbs, so both disjoint
        // n-limb halves exist. Zero extension preserves the input's value.
        let (low, high) = unsafe { t.limbs_mut().split_at_mut_unchecked(n) };
        scratch.coefficients.reset_with_capacity(n);
        // SAFETY: reservation provides n writable limbs disjoint from the
        // two initialized n-limb factors. LowProduct initializes all n limbs
        // before set_len exposes them as the reduction coefficient.
        unsafe {
            LowProduct::mul(
                scratch
                    .coefficients
                    .spare_capacity_mut()
                    .get_unchecked_mut(..n),
                low,
                &self.inverse,
                n,
                &mut scratch.multiplication,
            );
            scratch.coefficients.set_len(n);
        }
        let correction = self.correction_product(low, scratch);
        debug_assert_eq!(correction.len(), n, "the exact high product spans n limbs");
        // SAFETY: q and M each have n limbs, so retaining their product
        // above position n supplies exactly n initialized correction limbs.
        // The resized input's high half is initialized and disjoint. Passing
        // the established width avoids redispatching on the returned slice.
        let borrow =
            unsafe { ArchKernels::sub_limbs_unchecked(high.as_mut_ptr(), correction.as_ptr(), n) };
        if borrow != 0 {
            // Both high halves are below M. A negative difference lies in
            // (-M,0); its n-limb residue plus M overflows R exactly once.
            let carry = Addition::add_slice_in_place(high, modulus);
            debug_assert_eq!(carry, 1, "the correction consumes the wrapped sign");
        }
        out.clone_from_slice(high);
    }

    /// Reconstructs `floor(qM/R)` from a cyclic product and its known low half.
    ///
    /// Write `P=qM=P0+H*B^w`, `n<=w<2n`, `r=2n-w`, and `L=P mod B^r`.
    /// The cyclic representation is `Y=P0+H-c*(B^w-1)`, `c` in `{0,1}`.
    /// Since `P<=(R-1)^2`, `H+c<B^r`. Thus subtracting `L` from the low
    /// `r` digits of `Y` yields `H+c` and borrow `b`. Appending those digits
    /// and subtracting `b*B^r` gives `P+(Y mod B^r)-L`; all digits above
    /// position `r` equal the exact product. In particular, its high half
    /// is exact because `r<=n`. For a nonzero product, both cyclic zero
    /// representations satisfy the identity. An actual zero product retains
    /// the all-zero state through the CRT leaves, transforms, and merges.
    /// Below cyclic admission, an exact high product
    /// obtains the same result while omitting most unused low digits.
    fn correction_product<'scratch>(
        &self,
        low: &[Limb],
        scratch: &'scratch mut MontgomeryScratch,
    ) -> &'scratch mut [Limb] {
        let n = low.len();
        #[cfg(not(target_pointer_width = "16"))]
        {
            // SAFETY: low is a materialized limb slice of at most isize::MAX
            // bytes; each limb occupies at least two bytes, so 2*n fits usize.
            let double_n = unsafe { n.unchecked_mul(2) };
            if Multiplication::try_mul_mod_bnm1::<false, true>(
                &scratch.coefficients,
                self.modulus.limbs(),
                n,
                &mut scratch.high_product,
                &mut scratch.multiplication,
            ) {
                let width = scratch.high_product.len();
                let product = scratch.high_product.as_mut_ptr();
                // SAFETY: cyclic admission supplies at least n initialized
                // limbs. This prefix is disjoint from the reserved output
                // tail, which begins at width>=n.
                let cyclic_low = unsafe { scratch.high_product.get_unchecked(..n) };
                // SAFETY: cyclic admission gives n<=width<2n, hence
                // 0<tail<=n. Reservation provides a disjoint tail of
                // writable limbs after the initialized cyclic product.
                let borrow = unsafe {
                    let tail = double_n.unchecked_sub(width);
                    ArchKernels::sub_limbs_3_unchecked(
                        product.add(width),
                        cyclic_low.as_ptr(),
                        low.as_ptr(),
                        tail,
                    )
                };
                // SAFETY: the cyclic product initialized [0,width),
                // and the subtraction writer initialized [width,2n).
                unsafe {
                    scratch.high_product.set_len(double_n);
                }
                if borrow != 0 {
                    // A borrow means L+(H+c)>=B^r, hence H+c>0. The
                    // appended r digits therefore contain a nonzero limb
                    // that absorbs the borrow before position 2n. This
                    // proves termination without checking the span again.
                    // SAFETY: 0<r=2n-width<=n gives the first initialized
                    // position; the materialized modulus proves 2n fits usize.
                    let mut index = unsafe { double_n.unchecked_sub(width) };
                    loop {
                        // SAFETY: an absorbing nonzero digit exists in
                        // [width,2n). Every preceding position is initialized;
                        // zero digits retain the borrow and advance toward it.
                        let digit = unsafe { &mut *product.add(index) };
                        let (difference, pending) = digit.overflowing_sub(1);
                        *digit = difference;
                        if !pending {
                            break;
                        }
                        // SAFETY: a zero digit precedes an absorbing one,
                        // so index+1<2n is initialized and cannot overflow.
                        index = unsafe { index.unchecked_add(1) };
                    }
                }
                // SAFETY: all 2n limbs are initialized; the reconstructed
                // high n digits coincide with floor(qM/R).
                return unsafe { scratch.high_product.get_unchecked_mut(n..) };
            }
        }
        HighProduct::mul(
            &scratch.coefficients,
            self.modulus.limbs(),
            n,
            &mut scratch.high_product,
            &mut scratch.carry_product,
            &mut scratch.multiplication,
        )
    }

    /// Cancels the low n digits for a prepared scalar domain and `0<=t<M*R`.
    fn reduce_scalar(&self, t: &mut InternalMpUint, out: &mut InternalMpUint) {
        let modulus = self.modulus.limbs();
        let n = modulus.len();
        // SAFETY: the materialized modulus spans at most isize::MAX bytes
        // with at least two bytes per limb, so its positive 2*n count fits usize.
        let double_n = unsafe { n.unchecked_mul(2) };
        t.resize(double_n);
        let limbs = t.limbs_mut();
        let add_mul = ArchKernels::selected_add_mul_limbs_unchecked();
        let mut pending_carry = 0;
        for index in 0..n {
            // SAFETY: index<n<2n bounds the initialized low input digit.
            // Multiplication computes a coefficient modulo the limb radix.
            let coefficient = unsafe { *limbs.get_unchecked(index) }.wrapping_mul(self.m_inv);
            // SAFETY: index+n<2n bounds the initialized n-limb row and its
            // following carry slot. The immutable modulus is disjoint; the
            // selected architecture backend meets target prerequisites.
            let carry = unsafe {
                add_mul(
                    limbs.as_mut_ptr().add(index),
                    modulus.as_ptr(),
                    n,
                    coefficient,
                )
            };
            // Row index writes [index,index+n). The preceding deferred bit
            // has weight B^(index+n), so merging it cannot change the digit.
            // SAFETY: index<n gives index+n<2n=limbs.len(); the resized
            // initialized span proves that the sum is representable.
            let top = unsafe { limbs.get_unchecked_mut(index.unchecked_add(n)) };
            let (sum, product_overflow) = top.overflowing_add(carry);
            let (total, pending_overflow) = sum.overflowing_add(pending_carry);
            *top = total;
            pending_carry = Limb::from(product_overflow || pending_overflow);
        }
        // SAFETY: the initialized 2n-limb input has an n-limb high half.
        let high = unsafe { limbs.get_unchecked_mut(n..) };
        if pending_carry != 0 || high.iter().rev().cmp(modulus.iter().rev()) != Ordering::Less {
            let borrow = Addition::sub_slice_in_place(high, modulus);
            debug_assert_eq!(
                borrow, pending_carry,
                "REDC correction consumes the top carry"
            );
        }
        out.clone_from_slice(high);
    }
}
