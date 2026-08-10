//! Fused in-place unsigned arithmetic APIs.

use super::ArbiUint;

impl ArbiUint {
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
        self.value.assign_sum(&a.value, &b.value);
        if let Some(bits) = self.precision.significant_bits() {
            let overflow = self.value.required_unsigned_bits_for_bounded_storage() > bits;
            if overflow {
                // Preserve the representation invariant even when callers catch
                // the overflow panic. Clearing retains the reusable allocation.
                self.value.clear();
            }
            assert!(
                !overflow,
                "ArbiUint fused addition overflow for Bounded({bits})"
            );
        }
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
        self.value.assign_product(&a.value, &b.value);
        self.assert_assigned_product_fits("multiplication");
    }

    /// Fused in-place squaring. Computes `self = a * a` directly into `self`'s
    /// buffer, reusing its allocation.
    ///
    /// The squaring counterpart of [`Self::assign_mul`], and the reason to
    /// reach for it over `assign_mul(a, a)` is only clarity -- that call
    /// detects the aliasing and takes this path anyway.
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
        self.assert_assigned_product_fits("squaring");
    }

    /// Fused in-place subtraction. Computes `self = a - b` directly into `self`'s buffer,
    /// avoiding intermediate allocations and extra memory passes.
    ///
    /// Returns `true` if the subtraction underflowed. On underflow, `self` is
    /// set to zero; use [`Self::wrapping_sub`] when a modular result is needed.
    #[must_use = "the return value reports whether unsigned subtraction underflowed"]
    #[allow(
        clippy::inline_always,
        reason = "Hot arithmetic entry point critical for in-place assignment peak performance"
    )]
    #[inline(always)]
    pub fn assign_sub(&mut self, a: &Self, b: &Self) -> bool {
        self.precision = a.precision.combine_for_binary_op(b.precision);
        let underflow = self.value.assign_difference(&a.value, &b.value);
        if underflow {
            // The fused limb kernel leaves a machine-limb-width two's-complement
            // residue. That width is a storage detail, not part of ArbiUint's
            // public arithmetic contract, so expose a deterministic failure value.
            self.value.clear();
        }
        self.assert_fits("fused subtraction");
        self.debug_assert_valid();
        underflow
    }

    /// Enforces bounded precision on a fused product, clearing on overflow.
    ///
    /// Shared by [`Self::assign_mul`] and [`Self::assign_square`]: both must
    /// leave a valid value behind even when the caller catches the panic, and
    /// clearing rather than truncating keeps the reusable allocation.
    #[track_caller]
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "Overflow check on the hot fused-assignment path"
    )]
    fn assert_assigned_product_fits(&mut self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            let overflow = self.value.required_unsigned_bits_for_bounded_storage() > bits;
            if overflow {
                self.value.clear();
            }
            assert!(
                !overflow,
                "ArbiUint fused {operation} overflow for Bounded({bits})"
            );
        }
        self.debug_assert_valid();
    }
}
