//! Destination-preserving fused unsigned arithmetic APIs.

use super::{InternalMpUint, MpUint};

impl MpUint {
    /// Computes `self = a + b`, preserving the destination's precision.
    ///
    /// Reuses the destination buffer directly when precision is unlimited or
    /// operand widths prove the sum fits. Other bounded sums are validated in
    /// temporary storage before copying into the reusable buffer. Operand
    /// precision does not constrain the result.
    ///
    /// # Panics
    /// Panics if the sum exceeds the destination's bounded precision, leaving
    /// the destination unchanged.
    #[track_caller]
    #[inline]
    pub fn assign_add(&mut self, a: &Self, b: &Self) {
        if let Some(bits) = self.precision.significant_bits() {
            assign_add_bounded(self, a, b, bits);
        } else {
            self.value.assign_sum(&a.value, &b.value);
        }
        self.debug_assert_valid();
    }

    /// Computes `self = a * b`, preserving the destination's precision.
    ///
    /// Unlimited destinations reuse their buffer directly; bounded results
    /// are validated in temporary storage before copying into that buffer.
    /// Identical operand references select the dedicated squaring path.
    ///
    /// # Panics
    /// Panics if the product exceeds the destination's bounded precision,
    /// leaving the destination unchanged.
    #[track_caller]
    #[inline]
    pub fn assign_mul(&mut self, a: &Self, b: &Self) {
        assign_validated(self, "fused multiplication", |value| {
            value.assign_product(&a.value, &b.value);
        });
    }

    /// Computes `self = a * a`, preserving the destination's precision.
    ///
    /// Unlimited destinations reuse their buffer directly; bounded results
    /// are validated in temporary storage before copying into that buffer.
    ///
    /// # Panics
    /// Panics if the square exceeds the destination's bounded precision,
    /// leaving the destination unchanged.
    #[track_caller]
    #[inline]
    pub fn assign_square(&mut self, a: &Self) {
        assign_validated(self, "fused squaring", |value| {
            value.assign_square(&a.value);
        });
    }

    /// Computes `self = a - b`, preserving the destination's precision.
    ///
    /// Returns `true` when `a < b`, leaving the destination unchanged. Otherwise
    /// returns `false`. Unlimited destinations reuse their buffer directly;
    /// bounded results are validated in temporary storage before copying into
    /// that buffer.
    ///
    /// # Panics
    /// Panics if the nonnegative difference exceeds the destination's bounded
    /// precision, leaving the destination unchanged.
    #[must_use = "the return value reports whether unsigned subtraction underflowed"]
    #[track_caller]
    #[inline]
    pub fn assign_sub(&mut self, a: &Self, b: &Self) -> bool {
        if a.value < b.value {
            return true;
        }
        assign_validated(self, "fused subtraction", |value| {
            // The boundary comparison proves that the difference is nonnegative.
            let underflow = value.assign_difference(&a.value, &b.value);
            debug_assert!(!underflow, "validated unsigned difference is nonnegative");
        });
        false
    }
}

/// Executes bounded addition after the API has decoded the destination width.
#[track_caller]
fn assign_add_bounded(destination: &mut MpUint, a: &MpUint, b: &MpUint, bits: usize) {
    // Each operand is below 2^(bits-1), so their sum is below 2^bits.
    let operand_bits = bits.saturating_sub(1);
    if a.value.fits_in_bits(operand_bits) && b.value.fits_in_bits(operand_bits) {
        destination.value.assign_sum(&a.value, &b.value);
    } else {
        assign_bounded(destination, bits, "fused addition", |value| {
            value.assign_sum(&a.value, &b.value);
        });
    }
}

/// Decodes the destination policy once before direct or transactional execution.
#[track_caller]
#[inline]
fn assign_validated(
    destination: &mut MpUint,
    operation: &str,
    assign: impl FnOnce(&mut InternalMpUint),
) {
    if let Some(bits) = destination.precision.significant_bits() {
        assign_bounded(destination, bits, operation, assign);
    } else {
        assign(&mut destination.value);
    }
    destination.debug_assert_valid();
}

/// Validates a bounded result before committing it to reusable storage.
///
/// The caller has established `destination.precision == Bounded(bits)`.
#[track_caller]
#[inline]
fn assign_bounded(
    destination: &mut MpUint,
    bits: usize,
    operation: &str,
    assign: impl FnOnce(&mut InternalMpUint),
) {
    debug_assert_eq!(
        destination.precision.significant_bits(),
        Some(bits),
        "bounded assignment width must match the destination precision"
    );
    let mut result = InternalMpUint::zero();
    assign(&mut result);
    // The decoded width already establishes the bounded policy. Only the new
    // magnitude needs validation; no precision-bearing temporary is required.
    assert!(
        result.required_unsigned_bits_for_bounded_storage() <= bits,
        "MpUint {operation} overflow for Bounded({bits})"
    );
    destination.value.clone_from(&result);
}
