//! Fixed-exponent scaling and half-ring factors for Fermat coefficients.
//!
//! Mixed ITFT columns multiply by two or its inverse. These fixed exponents
//! require one data shift and bounded guard corrections. Square-root twists
//! use a fused half-ring rotation and subtraction.

#![expect(
    unsafe_code,
    reason = "Positive limb-aligned rings supply complete coefficients; guard and carry bounds justify every scaling correction"
)]

use super::{ArchKernels, Limb, SsaCarry, SsaRing};

impl SsaRing {
    /// Computes `(2^(n/2) - 1) * src` with a fused rotation and subtraction.
    ///
    /// For R = 2^(n/2), write src = L + R*H + g*R^2. Since R^2 = -1,
    /// the result is (g-L-H) + R*(L-H-g). The low half's signed borrow feeds
    /// the high half; an escaping borrow adds at the low end because -R^2 = 1.
    /// An odd data-limb count requires the general half-limb shift kernel.
    ///
    /// # Safety
    /// `mod_bits` is a positive limb multiple. Both disjoint buffers contain
    /// `coeff_limbs(mod_bits)` initialized limbs; the input guard is at most one.
    pub unsafe fn half_ring_sub_from(dst: &mut [Limb], src: &[Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        if !ml.is_multiple_of(2) {
            // SAFETY: the caller proves complete disjoint coefficients and the
            // half-ring exponent is below the 2n period, even for an odd limb count.
            unsafe {
                Self::shift_from(dst, src, mod_bits >> 1, mod_bits);
                Self::sub_in_place(dst, src, mod_bits);
            }
            return;
        }

        // SAFETY: both complete coefficients contain ml data limbs and a
        // guard; positive even ml supplies two equal nonempty data halves.
        let ((source_low, source_high), source_guard, (data, destination_guard)) = unsafe {
            let (source_data, source_guard) = src.split_at_unchecked(ml);
            (
                source_data.split_at_unchecked(ml >> 1),
                source_guard,
                dst.split_at_mut_unchecked(ml),
            )
        };
        // SAFETY: the complete source includes its guard at ml.
        let guard = unsafe { *source_guard.get_unchecked(0) };
        debug_assert!(guard <= 1, "the source is semi-normalized");
        // SAFETY: positive even ml partitions the complete data into two halves.
        let (low, high) = unsafe { data.split_at_mut_unchecked(ml >> 1) };

        let mut incoming = guard;
        let mut borrow = 0;
        for ((output, &left), &right) in low.iter_mut().zip(source_low).zip(source_high) {
            let (difference, first) = incoming.overflowing_sub(left);
            let (after_right, second) = difference.overflowing_sub(right);
            let (after_borrow, third) = after_right.overflowing_sub(borrow);
            *output = after_borrow;
            // The signed limb value lies between -2B and B-1. Each summed
            // flag is at most one, so this sum fits every supported Limb.
            // SAFETY: each flag is a bit; their sum is at most three.
            borrow = unsafe {
                Limb::from(first)
                    .unchecked_add(Limb::from(second))
                    .unchecked_add(Limb::from(third))
            };
            incoming = 0;
        }

        // A guard bit plus the borrow sum is at most four.
        // SAFETY: borrow<=3 and guard<=1, so the sum is at most four.
        borrow = unsafe { borrow.unchecked_add(guard) };
        for ((output, &left), &right) in high.iter_mut().zip(source_low).zip(source_high) {
            let (difference, first) = left.overflowing_sub(right);
            let (after_borrow, second) = difference.overflowing_sub(borrow);
            *output = after_borrow;
            // SAFETY: both flags are bits; their sum is at most two.
            borrow = unsafe { Limb::from(first).unchecked_add(Limb::from(second)) };
        }

        // SAFETY: dst includes its guard; positive even ml gives a first data
        // limb. The initialized data and guard partitions are disjoint.
        unsafe {
            *destination_guard.get_unchecked_mut(0) = 0;
            let (first, tail) = data.split_first_mut().unwrap_unchecked();
            let (corrected, carry) = first.overflowing_add(borrow);
            *first = corrected;
            if carry && SsaCarry::propagate_carry(tail) {
                *destination_guard.get_unchecked_mut(0) = 1;
            }
            // The carry can leave nonzero data with guard one.
            let _ = Self::normalize(dst, mod_bits);
        }
    }

    /// Multiplies a coefficient by two modulo `2^n + 1`.
    ///
    /// For `x = d + g*2^n`, shifting the data gives `2d = e + c*2^n`.
    /// The combined guard `h = 2g+c` lies in `0..=3`. Subtracting
    /// `max(h-1,0)*(2^n+1)` leaves a semi-normalized coefficient.
    ///
    /// # Safety
    /// `mod_bits` is positive and limb-aligned. The destination contains a
    /// complete initialized coefficient whose guard is at most one.
    pub unsafe fn double_in_place(dst: &mut [Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: the complete coefficient supplies ml data limbs and a guard.
        let (data, high) = unsafe { dst.split_at_mut_unchecked(ml) };
        // SAFETY: the complete coefficient contains its guard immediately
        // after the nonempty data span. The two spans are disjoint.
        let guard = unsafe { high.get_unchecked_mut(0) };
        debug_assert!(*guard <= 1, "a semi-normalized guard is at most one");
        // SAFETY: data is an initialized, aligned, exclusive ml-limb span;
        // ml >= 1 and 1 < Limb::BITS on 16-, 32-, and 64-bit targets. The
        // architecture facade selects a backend supported by the current CPU.
        let carry = unsafe { ArchKernels::lshift_unchecked(data.as_mut_ptr(), ml, 1) };
        // SAFETY: guard<=1 and carry<=1, so 2*guard+carry<=3.
        let combined = unsafe { guard.unchecked_mul(2).unchecked_add(carry) };
        let adjustment = combined.saturating_sub(1);
        *guard = Limb::from(combined != 0);
        // SAFETY: the positive ring width gives at least one data limb, and
        // splitting it preserves exclusive access to the remaining data.
        let (first, tail) = unsafe { data.split_first_mut().unwrap_unchecked() };
        let (low, borrow) = first.overflowing_sub(adjustment);
        *first = low;
        if borrow && SsaCarry::propagate_borrow(tail) {
            // An escaping borrow requires adjustment > 0, hence guard == 1.
            // Subtracting through that guard leaves zero without underflow.
            *guard = 0;
        }
    }

    /// Divides a coefficient by two modulo the odd modulus `2^n + 1`.
    ///
    /// Let `x = d + g*2^n` and `b = d mod 2`. The integer
    /// `(x + b*(2^n+1))/2 = floor(d/2) + (g+b)*2^(n-1) + b`
    /// represents `x/2` in the ring. The sum `g+b` splits into the top
    /// data bit `g XOR b` and guard `g AND b`. Adding b to the low data
    /// cannot escape the complete coefficient: when the guard is one,
    /// the shifted data has its top bit clear before that addition.
    ///
    /// # Safety
    /// `mod_bits` is positive and limb-aligned. The destination contains a
    /// complete initialized coefficient whose guard is at most one.
    pub unsafe fn halve_in_place(dst: &mut [Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: the complete coefficient supplies ml data limbs and a guard.
        let (data, high) = unsafe { dst.split_at_mut_unchecked(ml) };
        // SAFETY: the complete coefficient contains the guard after its
        // nonempty data span. Both mutable partitions are disjoint.
        let guard = unsafe { high.get_unchecked_mut(0) };
        debug_assert!(*guard <= 1, "a semi-normalized guard is at most one");
        // SAFETY: the positive ring width establishes the first data limb.
        let odd = unsafe { *data.get_unchecked(0) } & 1;
        let top = (*guard ^ odd).wrapping_shl(Limb::BITS - 1);
        *guard &= odd;
        // SAFETY: data is an initialized, aligned, exclusive ml-limb span;
        // ml >= 1 and 0 < 1 < Limb::BITS on every supported pointer width.
        // CPU feature selection remains inside the architecture facade.
        let _ = unsafe { ArchKernels::rshift_unchecked(data.as_mut_ptr(), ml, 1) };
        // SAFETY: ml >= 1, so a last data limb exists. Its top bit is clear
        // after the logical shift; inserting top is the exact bounded sum.
        unsafe {
            *data.last_mut().unwrap_unchecked() |= top;
        }
        // SAFETY: the first data limb exists and is disjoint from its tail.
        let (first, tail) = unsafe { data.split_first_mut().unwrap_unchecked() };
        let (low, carry) = first.overflowing_add(odd);
        *first = low;
        if carry && SsaCarry::propagate_carry(tail) {
            // A carry through all data limbs requires the inserted top bit,
            // which proves the previous guard was zero. The result guard is one.
            *guard = 1;
        }
    }
}
