//! Quotients with a known nonzero exact divisor.
//!
//! Scalar divisors use low-to-high cancellation after removing their
//! power-of-two factor. Wider divisors use quotient truncation or the tower.
//!
//! Reference: T. Jebelean, "An algorithm for exact division", Journal of Symbolic
//! Computation, Vol. 15, No. 2, pp. 169–180, 1993. DOI: 10.1006/jsco.1993.1012.

#![expect(
    unsafe_code,
    reason = "known exact factors and validated output spans establish cancellation and initialization contracts"
)]

use super::{ArchKernels, DivScratch, Division, InternalMpUint, Limb};

impl Division {
    /// Divides by a known nonzero exact divisor into reusable output storage.
    ///
    /// A scalar divisor is made odd by shifting the input during its read.
    /// For limb base B, the recurrence `q_i = (a_i - carry)*d^-1 mod B`
    /// cancels each low limb. The high half of `q_i*d` plus the subtraction
    /// borrow is the next carry, bounded by d-1. Exactness makes the final
    /// carry zero. No normalization, reciprocal estimate or correction is needed.
    /// Wider divisors use exact prefix truncation or the general quotient tower.
    pub fn div_exact_into(
        numerator: &InternalMpUint,
        divisor: &InternalMpUint,
        quotient: &mut InternalMpUint,
        scratch: &mut DivScratch,
    ) {
        debug_assert!(
            !divisor.is_zero(),
            "exact division requires a nonzero divisor"
        );
        if numerator.is_zero() {
            quotient.clear();
            return;
        }
        if let [scalar] = divisor.limbs() {
            if scalar.is_power_of_two() {
                let _ = Self::div_rem_1::<true>(numerator.limbs(), *scalar, quotient);
                return;
            }
            let input = numerator.limbs();
            let mut output = quotient.prepare_limb_write(input.len());
            // SAFETY: the nonzero numerator and scalar are disjoint from the
            // output guard, which reserves input.len() aligned writable limbs.
            // Exact divisibility is proved by the caller; the kernel initializes
            // every reserved limb before the length is committed.
            unsafe {
                div_exact_1_raw(input, *scalar, output.as_mut_ptr());
                let _ = output.commit();
            }
            quotient.normalize();
        } else {
            Self::div_into::<true, true>(numerator, divisor, quotient, scratch);
        }
    }
}

/// Cancels a known scalar factor from low to high into raw output storage.
///
/// For odd d, `q_i = (a_i-carry)*d^-1 mod B` cancels one low limb.
/// The next carry is `high(q_i*d)+borrow < d`: equality of the product's
/// high half with d-1 forces its low half at most B-d, whereas a borrow
/// gives residue at least B-d+1. The final exact residue is a nonnegative
/// multiple of d below B, so its quotient needs no high product or carry.
///
/// # Safety
///
/// `input` is nonempty, scalar is nonzero and divides its value exactly.
/// `output` covers `input.len()` aligned writable limbs, disjoint from input.
/// The output need not be initialized; every limb receives its first write.
unsafe fn div_exact_1_raw(input: &[Limb], scalar: Limb, output: *mut Limb) {
    let shift = scalar.trailing_zeros();
    let odd = scalar >> shift;
    let inverse = Division::modular_inverse_limb(odd);
    // SAFETY: the caller supplies a nonempty initialized input span.
    let (&top, prefix) = unsafe { input.split_last().unwrap_unchecked() };
    let mut carry = 0;
    if shift == 0 {
        for (index, &limb) in prefix.iter().enumerate() {
            let (residue, borrow) = limb.overflowing_sub(carry);
            let digit = residue.wrapping_mul(inverse);
            let (_, high) = ArchKernels::mul_limb_lo_hi(digit, odd);
            // SAFETY: the cancellation bound above proves high+borrow<odd.
            // index < prefix.len() < input.len() bounds the writable slot.
            unsafe {
                carry = high.unchecked_add(Limb::from(borrow));
                output.add(index).write(digit);
            }
        }
    } else {
        // SAFETY: scalar>0 gives shift<Limb::BITS; this branch has shift>0.
        // The nonempty input bounds the suffix, whose length matches prefix.
        let (complement, following) =
            unsafe { (Limb::BITS.unchecked_sub(shift), input.get_unchecked(1..)) };
        for (index, (&low, &next)) in prefix.iter().zip(following).enumerate() {
            let limb = (low >> shift) | (next << complement);
            let (residue, borrow) = limb.overflowing_sub(carry);
            let digit = residue.wrapping_mul(inverse);
            let (_, high) = ArchKernels::mul_limb_lo_hi(digit, odd);
            // SAFETY: the cancellation bound proves high+borrow<odd.
            // Equal prefix and suffix lengths bound this writable slot.
            unsafe {
                carry = high.unchecked_add(Limb::from(borrow));
                output.add(index).write(digit);
            }
        }
    }
    // Exactness makes (top>>shift)-carry divisible by odd. A negative
    // value would lie strictly between -odd and zero, which excludes an
    // exact multiple. Its nonnegative quotient is below B.
    // SAFETY: scalar>0 bounds shift, and the exactness argument proves
    // top>>shift>=carry. prefix.len() is the final reserved output slot.
    unsafe {
        let residue = (top >> shift).unchecked_sub(carry);
        let digit = residue.wrapping_mul(inverse);
        debug_assert_eq!(
            digit.checked_mul(odd),
            Some(residue),
            "exactness bounds the final product below the limb base"
        );
        output.add(prefix.len()).write(digit);
    }
}
