//! Double-limb quotient simulation with the half-GCD stopping rule.
//!
//! The simulation reduces the leading double-limb windows `a, b < B^2` of an
//! ordered operand pair, with `B = 2^LIMB_BITS`, and accumulates a
//! non-negative matrix `M` of determinant one with `(a, b) = M (a', b')`.
//! Reducing `a` by `q*b` adds `q` times the first column of `M` to its second
//! column; reducing `b` by `q*a` adds `q` times the second column to the first.
//!
//! The double-limb phase stops before an exact reduced window falls below
//! `2B`. Once the larger window drops below `B^(3/2)`, both lose their low
//! half limb and the single-limb phase stops below `2*B^(1/2)` on that scale.
//! Its matrix entries stay below `B^(1/2)/2`, so truncation perturbs each
//! exact reduced window by less than `B/2`: the exact windows stay at least
//! `3B/2`. Since `a = m00*a' + m01*b'` and `b = m10*a' + m11*b'` are sums of
//! non-negative terms, every entry of `M` is then below `2B/3`.
//!
//! For full operands `A = a*2^k + x` and `C = b*2^k + y`, `0 <= x, y < 2^k`,
//! `M^-1 (A, C) = (a'*2^k + m11*x - m01*y, b'*2^k + m00*y - m10*x)`. Each
//! correction lies strictly between `-(2B/3)*2^k` and `(2B/3)*2^k`, while each
//! reduced window contributes at least `(3B/2)*2^k`, so both results are
//! positive and the batch needs no validity test.
//!
//! Each loop iteration reduces `a`, then `b`, with fixed operand roles: a
//! quotient of one costs a subtraction and two coefficient additions, detected
//! from the high limbs alone, and larger quotients divide. Coefficient
//! products and sums never wrap, because every accepted matrix is below `2B/3`.
//!
//! Reference: N. Möller, "On Schönhage's algorithm and subquadratic integer
//! GCD computation", Mathematics of Computation, Vol. 77, No. 261,
//! pp. 589-607, 2008. DOI: 10.1090/S0025-5718-07-02017-0.

#![expect(
    unsafe_code,
    reason = "accepted half-GCD windows bound nonzero divisors, exact ordered reductions, and nonnegative matrix entries below one limb"
)]

use core::ops::ControlFlow;

use super::{ArchKernels, DoubleLimb, Gcd, LIMB_BITS, Limb};

/// Matrix column `(m0j, m1j)`.
type Column = (Limb, Limb);

impl Gcd {
    /// Returns the Lehmer transition `(u0, v0, u1, v1)` of a determinant-one
    /// batch, the `even` orientation of the Lehmer matrix update, for ordered
    /// operands `u >= v` of at least two limbs and equal or adjacent widths.
    ///
    /// The batch matrix `M` gives `u0 = m11`, `v0 = m01`, `u1 = m10` and
    /// `v1 = m00`. `None` means the leading windows admit no qualifying
    /// subtraction, because the operands are close or far apart.
    pub fn hgcd2(u: &[Limb], v: &[Limb]) -> Option<(Limb, Limb, Limb, Limb)> {
        let (mut a, mut b) = Self::extract_top_two_limbs(u, v);
        let difference = a.abs_diff(b);
        if a.min(b) >> LIMB_BITS < 2 || difference >> LIMB_BITS < 2 {
            return None;
        }
        let (mut first, mut second) = if a > b {
            a = difference;
            ((1, 0), (1, 1))
        } else {
            b = difference;
            ((1, 1), (0, 1))
        };

        let mut reduce_b_first = a >> LIMB_BITS < b >> LIMB_BITS;
        let reduce_a = loop {
            if !reduce_b_first {
                match reduce_double(&mut a, b, &mut second, first) {
                    ControlFlow::Continue(()) => {}
                    ControlFlow::Break(true) => {
                        return Some((second.1, second.0, first.1, first.0));
                    }
                    ControlFlow::Break(false) => break true,
                }
            }
            reduce_b_first = false;
            match reduce_double(&mut b, a, &mut first, second) {
                ControlFlow::Continue(()) => {}
                ControlFlow::Break(true) => return Some((second.1, second.0, first.1, first.0)),
                ControlFlow::Break(false) => break false,
            }
        };

        // Both windows are below B^(3/2); dropping half a limb keeps them in
        // one limb, and the side due for reduction carries over.
        // SAFETY: the double phase stopped with both windows below B^(3/2).
        // Shifting out half a limb leaves values below B on every target.
        // These infallible conversions need no range check in the scalar loop.
        let (mut narrow_a, mut narrow_b) = unsafe {
            (
                Limb::try_from(a >> (LIMB_BITS >> 1)).unwrap_unchecked(),
                Limb::try_from(b >> (LIMB_BITS >> 1)).unwrap_unchecked(),
            )
        };
        let mut skip_a = !reduce_a;
        loop {
            if !skip_a && reduce_single(&mut narrow_a, narrow_b, &mut second, first) {
                break;
            }
            skip_a = false;
            if reduce_single(&mut narrow_b, narrow_a, &mut first, second) {
                break;
            }
        }
        Some((second.1, second.0, first.1, first.0))
    }
}

/// Reduces a double-limb `window` by `q` times `divisor`, adding `q` times
/// `source` to `column`.
///
/// Requires the window's high limb to be at least the divisor's and both
/// windows to be at least `2B`. Breaks with `true` when the batch is complete
/// and with `false` when the window is below `B^(3/2)`, before any reduction.
/// A window below `B^2` over a divisor of at least `2B` has a one-limb quotient.
fn reduce_double(
    window: &mut DoubleLimb,
    divisor: DoubleLimb,
    column: &mut Column,
    source: Column,
) -> ControlFlow<bool> {
    let window_high = *window >> LIMB_BITS;
    let divisor_high = divisor >> LIMB_BITS;
    if window_high == divisor_high {
        return ControlFlow::Break(true);
    }
    if window_high >> (LIMB_BITS >> 1) == 0 {
        return ControlFlow::Break(false);
    }
    // SAFETY: equal high limbs returned above; the prescribed reduction
    // side has the greater high limb, hence window > divisor.
    let reduced = unsafe { window.unchecked_sub(divisor) };
    let reduced_high = reduced >> LIMB_BITS;
    if reduced_high < 2 {
        return ControlFlow::Break(true);
    }
    if reduced_high <= divisor_high {
        *window = reduced;
        // SAFETY: the accepted remainder stays at least 2B. The module
        // bound keeps each updated nonnegative matrix entry below 2B/3.
        unsafe {
            column.0 = column.0.unchecked_add(source.0);
            column.1 = column.1.unchecked_add(source.1);
        }
        return ControlFlow::Continue(());
    }
    let (partial, remainder) = Gcd::div_rem_wide(reduced, divisor);
    // A remainder below 2B retains one divisor instead, and ends the batch.
    let last = remainder >> LIMB_BITS < 2;
    // SAFETY: reduced < B^2 and divisor >= 2*B, so the exact partial
    // quotient is below B/2 and fits Limb; the wide divider's overflow
    // sentinel is unreachable. No conversion-failure branch is needed.
    let partial_limb = unsafe { Limb::try_from(partial).unwrap_unchecked() };
    // SAFETY: partial < B/2 leaves room for the binary increment.
    let quotient = unsafe { partial_limb.unchecked_add(Limb::from(!last)) };
    *window = if last {
        // SAFETY: partial>=1 gives remainder+divisor<=reduced<B^2.
        unsafe { remainder.unchecked_add(divisor) }
    } else {
        remainder
    };
    // SAFETY: the retained window is at least 2B. Each complete matrix
    // entry is below 2B/3; its nonnegative product and sum also fit.
    unsafe {
        column.0 = column.0.unchecked_add(quotient.unchecked_mul(source.0));
        column.1 = column.1.unchecked_add(quotient.unchecked_mul(source.1));
    }
    if last {
        ControlFlow::Break(true)
    } else {
        ControlFlow::Continue(())
    }
}

/// Reduces a one-limb `window` by `q` times `divisor`, adding `q` times
/// `source` to `column`. Returns `true` when the batch is complete.
///
/// Requires `window >= divisor >= 2^(LIMB_BITS/2 + 1)`.
fn reduce_single(window: &mut Limb, divisor: Limb, column: &mut Column, source: Column) -> bool {
    let minimum: Limb = 2 << (LIMB_BITS >> 1);
    // SAFETY: alternating reductions preserve window >= divisor.
    let reduced = unsafe { window.unchecked_sub(divisor) };
    if reduced < minimum {
        return true;
    }
    if reduced <= divisor {
        *window = reduced;
        // SAFETY: the half-limb stopping rule preserves the module's
        // matrix bound below 2B/3, including both nonnegative sums.
        unsafe {
            column.0 = column.0.unchecked_add(source.0);
            column.1 = column.1.unchecked_add(source.1);
        }
        return false;
    }
    // The double-limb phase retains both windows >= 2*B. Shifting by
    // half a limb gives divisor >= minimum, and every continuing reduction
    // preserves that bound. Use the native primitive directly: on 64-bit
    // targets the divisor exceeds u32::MAX, so a 32-bit divide cannot apply.
    // SAFETY: divisor >= minimum > 0; the numerator's high limb is zero,
    // strictly below divisor, so the exact quotient fits one limb.
    let (partial, remainder) = unsafe { ArchKernels::divrem_1_unchecked(reduced, 0, divisor) };
    let last = remainder < minimum;
    // SAFETY: divisor >= 2*sqrt(B) gives partial < sqrt(B)/2.
    let quotient = unsafe { partial.unchecked_add(Limb::from(!last)) };
    *window = if last {
        // SAFETY: partial>=1 gives remainder+divisor<=reduced<B.
        unsafe { remainder.unchecked_add(divisor) }
    } else {
        remainder
    };
    // SAFETY: the retained half-limb window preserves matrix entries
    // below 2B/3; every nonnegative partial product and sum also fits.
    unsafe {
        column.0 = column.0.unchecked_add(quotient.unchecked_mul(source.0));
        column.1 = column.1.unchecked_add(quotient.unchecked_mul(source.1));
    }
    last
}
