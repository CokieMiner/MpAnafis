//! Shared leading reciprocal and kernel for normalized division windows.
//!
//! Recursive division only removes low divisor limbs. Every leaf therefore
//! shares the same leading pair, reciprocal, and architecture kernel.

#![expect(
    unsafe_code,
    reason = "validated divisor prefixes and normalized window lengths bound reciprocal loads and quotient updates"
)]

use core::num::NonZeroUsize;

use super::{ArchKernels, Division, Limb};

/// Invariant divisor state shared by all windows with the same leading pair.
pub struct PreparedDivisor {
    pub high: Limb,
    pub low: Limb,
    pub inverse: Limb,
    pub sub_mul: unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb),
}

impl PreparedDivisor {
    /// Prepares a normalized divisor of at least two limbs.
    pub fn new(divisor: &[Limb]) -> Self {
        debug_assert!(
            divisor.len() >= 2,
            "the reciprocal requires two divisor limbs"
        );
        // SAFETY: normalized callers supply at least two initialized limbs.
        // The fixed-size leading pair exists; its two loads prepare the
        // reciprocal without independent index or minimum-width assertions.
        let (_, &[low, high]) = unsafe { divisor.split_last_chunk::<2>().unwrap_unchecked() };
        Self {
            high,
            low,
            inverse: Division::invert_pi1(high, low),
            sub_mul: ArchKernels::selected_sub_mul_limbs_unchecked(),
        }
    }

    /// Divides a normalized window in place. The remainder replaces the low
    /// `divisor.len()` numerator limbs. The quotient is empty or contains
    /// exactly `numerator.len()-divisor.len()` initialized output limbs.
    ///
    /// The divisor has at least two limbs and the prepared leading pair. The
    /// numerator has at least one guard limb and its high divisor-width window
    /// is below the divisor, so every quotient digit fits one limb.
    #[expect(
        clippy::inline_always,
        reason = "Pinned small-remainder measurements and emitted assembly expose extra calls and stack materialization between divisor setup and this loop; inlining retains the reciprocal and high remainder while recursive leaves share setup."
    )]
    #[inline(always)]
    pub fn divide(&self, numerator: &mut [Limb], divisor: &[Limb], quotient: &mut [Limb]) {
        debug_assert!(
            divisor.len() >= 2 && numerator.len() > divisor.len(),
            "division requires two divisor limbs and a numerator guard"
        );
        debug_assert_eq!(
            divisor.last(),
            Some(&self.high),
            "prepared high limb must match"
        );
        debug_assert_eq!(
            divisor.iter().rev().nth(1),
            Some(&self.low),
            "prepared second limb must match"
        );
        // SAFETY: every normalized prefix retains the two prepared leading
        // limbs. Its remaining prefix determines the complete divisor width.
        let (lower, _) = unsafe { divisor.split_last_chunk::<2>().unwrap_unchecked() };
        // SAFETY: those two limbs belong to the materialized divisor. The
        // guarded numerator is longer, giving a positive quotient width.
        let (n, digits) = unsafe {
            let n = lower.len().unchecked_add(2);
            (
                n,
                NonZeroUsize::new_unchecked(numerator.len().unchecked_sub(n)).get(),
            )
        };
        if n == 2 {
            // SAFETY: the numerator has a guard, so digits >= 1.
            let last = unsafe { digits.unchecked_sub(1) };
            Division::div_rem_2(numerator, quotient, last, self.high, self.low, self.inverse);
            return;
        }
        let write_quotient = !quotient.is_empty();
        debug_assert!(
            !write_quotient || quotient.len() == digits,
            "the complete quotient span is required"
        );
        let mut active_digits = digits;
        // SAFETY: numerator.len() >= n+1 >= 4 bounds its leading pair.
        // A skipped zero digit leaves the next high window strictly below D.
        // When writing, quotient.len() = digits bounds that omitted slot.
        let mut high = unsafe {
            let guard = numerator.len().unchecked_sub(1);
            let upper = *numerator.get_unchecked(guard);
            let following = *numerator.get_unchecked(guard.unchecked_sub(1));
            if upper == 0 && following < self.high {
                active_digits = digits.unchecked_sub(1);
                if write_quotient {
                    *quotient.get_unchecked_mut(active_digits) = 0;
                }
                following
            } else {
                upper
            }
        };
        if write_quotient {
            // SAFETY: active_digits <= digits = quotient.len(). The omitted
            // leading slot, when present, was initialized to zero above.
            let output = unsafe { quotient.get_unchecked_mut(..active_digits) };
            for (j, digit) in output.iter_mut().enumerate().rev() {
                // SAFETY: j < active_digits <= digits and numerator.len() =
                // n+digits bound the initialized n-limb window. The preceding
                // remainder is below D, including after the zero high digit.
                let window = unsafe { numerator.get_unchecked_mut(j..j.unchecked_add(n)) };
                (*digit, high) = Division::knuth_d_step(
                    window,
                    divisor,
                    high,
                    self.high,
                    self.low,
                    self.inverse,
                    self.sub_mul,
                );
            }
        } else {
            for j in (0..active_digits).rev() {
                // SAFETY: j < active_digits <= digits and numerator.len() =
                // n+digits bound the initialized window. Its high guard is
                // retained in high; no quotient output is accessed.
                let window = unsafe { numerator.get_unchecked_mut(j..j.unchecked_add(n)) };
                (_, high) = Division::knuth_d_step(
                    window,
                    divisor,
                    high,
                    self.high,
                    self.low,
                    self.inverse,
                    self.sub_mul,
                );
            }
        }
        // SAFETY: n >= 3 and numerator.len() > n. All lower remainder limbs
        // were written in the loop; this publishes the retained high limb.
        unsafe {
            *numerator.get_unchecked_mut(n.unchecked_sub(1)) = high;
        }
    }
}
