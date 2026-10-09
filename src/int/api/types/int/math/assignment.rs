//! Destination-preserving fused signed arithmetic APIs.

use super::{InternalMpInt, MpInt};

impl MpInt {
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
            self.value.assign_add(&a.value, &b.value);
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
            value.assign_mul(&a.value, &b.value);
        });
    }

    /// Computes `self = a * a`, preserving the destination's precision.
    ///
    /// The result is nonnegative. Unlimited destinations reuse their buffer
    /// directly; bounded results are validated in temporary storage before
    /// copying into that buffer.
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
    /// Unlimited destinations reuse their buffer directly; bounded results
    /// are validated in temporary storage before copying into that buffer.
    ///
    /// # Panics
    /// Panics if the difference exceeds the destination's bounded precision,
    /// leaving the destination unchanged.
    #[track_caller]
    #[inline]
    pub fn assign_sub(&mut self, a: &Self, b: &Self) {
        assign_validated(self, "fused subtraction", |value| {
            value.assign_sub(&a.value, &b.value);
        });
    }
}

/// Executes bounded addition after the API has decoded the destination width.
#[track_caller]
fn assign_add_bounded(destination: &mut MpInt, a: &MpInt, b: &MpInt, bits: usize) {
    if a.value.sum_fits_by_width(&b.value, bits) {
        destination.value.assign_add(&a.value, &b.value);
    } else {
        assign_bounded(destination, bits, "fused addition", |value| {
            value.assign_add(&a.value, &b.value);
        });
    }
}

/// Decodes the destination policy once before direct or transactional execution.
#[track_caller]
#[inline]
fn assign_validated(
    destination: &mut MpInt,
    operation: &str,
    assign: impl FnOnce(&mut InternalMpInt),
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
    destination: &mut MpInt,
    bits: usize,
    operation: &str,
    assign: impl FnOnce(&mut InternalMpInt),
) {
    debug_assert_eq!(
        destination.precision.significant_bits(),
        Some(bits),
        "bounded assignment width must match the destination precision"
    );
    let mut result = InternalMpInt::zero();
    assign(&mut result);
    // A scratch value carries only magnitude and sign. Its required width is
    // checked once against the already decoded destination policy.
    assert!(
        result.required_signed_bits_for_bounded_storage() <= bits,
        "MpInt {operation} overflow for Bounded({bits})"
    );
    destination.value.abs.clone_from(&result.abs);
    destination.value.is_positive = result.is_positive;
}
