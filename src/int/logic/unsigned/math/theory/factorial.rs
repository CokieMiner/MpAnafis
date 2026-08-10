//! Factorial through a balanced product of odd factors.

#![allow(
    unsafe_code,
    reason = "The shift conversion is proven bounded by the u32 factorial input."
)]

use super::InternalArbiUint;

impl InternalArbiUint {
    /// Computes the factorial of `n` (`n!`).
    #[must_use]
    pub fn factorial(n: u32) -> Self {
        if n == 0 || n == 1 {
            return Self::one();
        }

        let mut odd_prod = Self::one();
        let mut current_p = Self::one();

        let bits = n.ilog2();
        for j in (0..=bits).rev() {
            let k = n >> j;
            let prev_k = if j == bits { 0 } else { n >> j.wrapping_add(1) };

            let p = product_of_odds(prev_k, k);
            if !p.is_one() {
                current_p = current_p.mul(&p);
            }
            if !current_p.is_one() {
                odd_prod = odd_prod.mul(&current_p);
            }
        }

        let shift = n.wrapping_sub(n.count_ones());
        let mut remaining_shift = shift;
        while remaining_shift > 0 {
            let shift_chunk = usize::try_from(remaining_shift).unwrap_or(usize::MAX);
            odd_prod.shl_assign(shift_chunk);
            // SAFETY: n is u32, shift = n - n.count_ones() ≤ n, so remaining_shift ≤ u32::MAX
            // and the unwrap_or(usize::MAX) at line 204 is unreachable in practice.
            let consumed_shift = unsafe { u32::try_from(shift_chunk).unwrap_unchecked() };
            remaining_shift = remaining_shift.wrapping_sub(consumed_shift);
        }
        odd_prod
    }
}

/// Computes the product of all odd integers in the interval `(a, b]`.
fn product_of_odds(a: u32, b: u32) -> InternalArbiUint {
    let mut first_odd = a.saturating_add(1);
    first_odd |= 1;
    if first_odd > b {
        return InternalArbiUint::one();
    }
    if first_odd.wrapping_add(2) > b {
        return InternalArbiUint::from_u64(u64::from(first_odd));
    }

    let diff = b.wrapping_sub(first_odd);
    if diff < 64 {
        let mut prod = InternalArbiUint::from_u64(u64::from(first_odd));
        let mut i = first_odd.wrapping_add(2);
        while i <= b {
            prod = prod.mul(&InternalArbiUint::from_u64(u64::from(i)));
            i = i.wrapping_add(2);
        }
        return prod;
    }

    let mid = first_odd.wrapping_add(diff.wrapping_div(2));
    let left = product_of_odds(a, mid);
    let right = product_of_odds(mid, b);
    left.mul(&right)
}
