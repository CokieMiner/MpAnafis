//! Fused in-place signed arithmetic APIs.

use super::MpInt;

impl MpInt {
    /// Fused in-place addition. Computes `self = a + b` directly into `self`'s buffer,
    /// avoiding intermediate allocations and extra memory passes.
    ///
    /// # Panics
    /// Panics if the exact sum does not fit the operands' combined bounded precision.
    /// Before panicking, the destination is reset to a valid zero while retaining
    /// its reusable allocation and the combined precision.
    #[track_caller]
    #[allow(
        clippy::inline_always,
        reason = "Hot arithmetic entry point critical for in-place assignment peak performance"
    )]
    #[inline(always)]
    pub fn assign_add(&mut self, a: &Self, b: &Self) {
        self.precision = a.precision.combine_for_binary_op(b.precision);
        self.value.assign_add(&a.value, &b.value);
        assert_fused_assignment_fits(self, "addition");
        self.debug_assert_valid();
    }

    /// Fused in-place multiplication. Computes `self = a * b` directly into
    /// `self`'s buffer, reusing its allocation instead of returning a new value.
    ///
    /// `a * b` has no destination to write into and must allocate one on every
    /// call. This form allocates once: measured at 1.15x on four-limb operands
    /// and 1.07x on eight, falling to parity by about a thousand bits as the
    /// multiplication itself comes to dominate. Use it for repeated products
    /// into the same accumulator -- modular exponentiation, series evaluation,
    /// matrix entries -- and the operator elsewhere, where it buys nothing.
    ///
    /// Squaring is detected: passing the same value as both operands takes the
    /// dedicated squaring tier, exactly as the operator does.
    ///
    /// # Panics
    /// Panics if the exact product does not fit the operands' combined bounded
    /// precision. Before panicking, the destination is reset to a valid zero
    /// while retaining its reusable allocation and the combined precision.
    #[track_caller]
    #[allow(
        clippy::inline_always,
        reason = "Hot arithmetic entry point critical for in-place assignment peak performance"
    )]
    #[inline(always)]
    pub fn assign_mul(&mut self, a: &Self, b: &Self) {
        self.precision = a.precision.combine_for_binary_op(b.precision);
        self.value.assign_mul(&a.value, &b.value);
        assert_fused_assignment_fits(self, "multiplication");
        self.debug_assert_valid();
    }

    /// Fused in-place squaring. Computes `self = a * a` directly into `self`'s
    /// buffer, reusing its allocation.
    ///
    /// The result is never negative regardless of `a`'s sign.
    ///
    /// # Panics
    /// Panics if the exact square does not fit `a`'s bounded precision, leaving
    /// the destination a valid zero as [`Self::assign_mul`] does.
    #[track_caller]
    #[allow(
        clippy::inline_always,
        reason = "Hot arithmetic entry point critical for in-place assignment peak performance"
    )]
    #[inline(always)]
    pub fn assign_square(&mut self, a: &Self) {
        self.precision = a.precision.combine_for_binary_op(a.precision);
        self.value.assign_square(&a.value);
        assert_fused_assignment_fits(self, "squaring");
        self.debug_assert_valid();
    }

    /// Fused in-place subtraction. Computes `self = a - b` directly into `self`'s buffer,
    /// avoiding intermediate allocations and extra memory passes.
    ///
    /// # Panics
    /// Panics if the exact difference does not fit the operands' combined bounded precision.
    /// Before panicking, the destination is reset to a valid zero while retaining
    /// its reusable allocation and the combined precision.
    #[track_caller]
    #[allow(
        clippy::inline_always,
        reason = "Hot arithmetic entry point critical for in-place assignment peak performance"
    )]
    #[inline(always)]
    pub fn assign_sub(&mut self, a: &Self, b: &Self) {
        self.precision = a.precision.combine_for_binary_op(b.precision);
        self.value.assign_sub(&a.value, &b.value);
        assert_fused_assignment_fits(self, "subtraction");
        self.debug_assert_valid();
    }
}

#[allow(
    clippy::inline_always,
    reason = "Inlining allows compile-time elimination of overflow checks when precision is Unlimited"
)]
#[inline(always)]
#[track_caller]
fn assert_fused_assignment_fits(destination: &mut MpInt, operation: &str) {
    if let Some(bits) = destination.precision.significant_bits() {
        let overflow = destination.value.required_signed_bits_for_bounded_storage() > bits;
        if overflow {
            // A caught overflow panic must not expose an invalid bounded value.
            // Clearing retains the destination magnitude's reusable allocation.
            destination.value.abs.clear();
            destination.value.is_positive = true;
        }
        assert!(
            !overflow,
            "MpInt fused {operation} overflow for Bounded({bits})"
        );
    }
}
