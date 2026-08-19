//! Fused and in-place signed assignment arithmetic.

#![allow(
    unsafe_code,
    reason = "Fixed-width subtraction proofs justify raw limb access and initialization while restoring normalized residues."
)]

use core::cmp::max;

use super::{InternalMpInt, InternalMpUint, Limb};

impl InternalMpInt {
    /// Computes `self = a * b` while reusing this magnitude allocation.
    pub fn assign_mul(&mut self, a: &Self, b: &Self) {
        let is_positive = a.is_positive == b.is_positive;
        self.abs.assign_product(&a.abs, &b.abs);
        self.is_positive = self.abs.is_zero() || is_positive;
    }

    /// Computes `self = a * a` while reusing this magnitude allocation.
    pub fn assign_square(&mut self, a: &Self) {
        self.abs.assign_square(&a.abs);
        self.is_positive = true;
    }

    /// Computes `self = a + b` while reusing this magnitude allocation.
    #[allow(
        clippy::inline_always,
        reason = "Hot fused signed assignment entry point"
    )]
    #[inline(always)]
    pub fn assign_add(&mut self, a: &Self, b: &Self) {
        if a.is_positive == b.is_positive {
            self.abs.assign_sum(&a.abs, &b.abs);
            self.is_positive = a.is_positive;
        } else {
            let underflow = self.abs.assign_difference(&a.abs, &b.abs);
            normalize_sign_after_sub(
                self,
                underflow,
                &a.abs,
                &b.abs,
                a.is_positive,
                b.is_positive,
            );
        }
    }

    /// Computes `self = a - b` while reusing this magnitude allocation.
    #[allow(
        clippy::inline_always,
        reason = "Hot fused signed assignment entry point"
    )]
    #[inline(always)]
    pub fn assign_sub(&mut self, a: &Self, b: &Self) {
        if a.is_positive == b.is_positive {
            let underflow = self.abs.assign_difference(&a.abs, &b.abs);
            normalize_sign_after_sub(
                self,
                underflow,
                &a.abs,
                &b.abs,
                a.is_positive,
                !a.is_positive,
            );
        } else {
            self.abs.assign_sum(&a.abs, &b.abs);
            self.is_positive = a.is_positive;
        }
    }

    /// Adds `other` into this value in place.
    #[allow(clippy::inline_always, reason = "Hot signed assignment entry point")]
    #[inline(always)]
    pub fn add_assign(&mut self, other: &Self) {
        if self.is_positive == other.is_positive {
            self.abs.add_assign(&other.abs);
        } else {
            let underflow = self.abs.sub_assign_with_underflow(&other.abs);
            if self.abs.is_zero() {
                self.is_positive = true;
            } else if underflow {
                // Underflow proves other.abs supplied the fixed subtraction width.
                negate_normalized_inplace(&mut self.abs, other.abs.limbs().len());
                self.is_positive = other.is_positive;
            }
        }
    }

    /// Subtracts `other` from this value in place.
    #[allow(clippy::inline_always, reason = "Hot signed assignment entry point")]
    #[inline(always)]
    pub fn sub_assign(&mut self, other: &Self) {
        if self.is_positive == other.is_positive {
            let underflow = self.abs.sub_assign_with_underflow(&other.abs);
            if self.abs.is_zero() {
                self.is_positive = true;
            } else if underflow {
                // Underflow proves other.abs supplied the fixed subtraction width.
                negate_normalized_inplace(&mut self.abs, other.abs.limbs().len());
                self.is_positive = !self.is_positive;
            }
        } else {
            self.abs.add_assign(&other.abs);
        }
    }
}

/// Negates a normalized fixed-width subtraction residue in place.
#[allow(
    clippy::inline_always,
    reason = "Signed underflow paths need post-normalization width recovery without a magnitude comparison"
)]
#[inline(always)]
pub fn negate_normalized_inplace(value: &mut InternalMpUint, subtraction_width: usize) {
    let active_len = value.limbs().len();
    negate_inplace(value);
    if active_len < subtraction_width {
        restore_negated_width(value, active_len, subtraction_width);
    }
}

/// Negates the active limbs of a two's-complement residue in place.
#[allow(
    clippy::inline_always,
    reason = "Called only on the signed arithmetic underflow path and must remain inlined"
)]
#[inline(always)]
fn negate_inplace(value: &mut InternalMpUint) {
    let (ptr, len) = {
        let limbs = value.limbs_mut();
        if limbs.is_empty() {
            return;
        }
        (limbs.as_mut_ptr(), limbs.len())
    };
    let mut index = 0_usize;
    while index < len {
        // SAFETY: the loop condition proves index < len and ptr addresses the
        // mutable limb allocation for the duration of this function.
        let limb = unsafe { *ptr.add(index) };
        if limb != 0 {
            // SAFETY: the same loop bound proves this slot is initialized and
            // exclusively borrowed through ptr.
            unsafe {
                *ptr.add(index) = limb.wrapping_neg();
            }
            index = index.wrapping_add(1);
            break;
        }
        index = index.wrapping_add(1);
    }
    while index < len {
        // SAFETY: the loop condition proves index < len; every slot remains
        // initialized and ptr retains exclusive access.
        unsafe {
            *ptr.add(index) = !*ptr.add(index);
        }
        index = index.wrapping_add(1);
    }
    value.normalize();
}

/// Restores high sign-extension limbs omitted by residue normalization.
#[cold]
#[inline(never)]
fn restore_negated_width(value: &mut InternalMpUint, low_width: usize, full_width: usize) {
    let result_len = value.limbs().len();
    debug_assert!(
        result_len <= low_width,
        "negation cannot widen beyond the normalized residue width"
    );
    debug_assert!(
        low_width < full_width,
        "width restoration requires at least one omitted high limb"
    );

    // SAFETY: full_width is the original initialized subtraction width and is
    // greater than result_len. Both new suffixes are initialized below before
    // the widened value becomes observable.
    let limbs = unsafe { value.ensure_capacity_set_len_get_limbs(full_width) };
    let (low_limbs, sign_extension) = limbs.split_at_mut(low_width);
    // SAFETY: fixed-width subtraction and negation prove result_len <= low_width.
    unsafe {
        low_limbs.get_unchecked_mut(result_len..).fill(0);
    }
    sign_extension.fill(Limb::MAX);
}

/// Resolves sign and magnitude after fixed-width speculative subtraction.
#[allow(
    clippy::inline_always,
    reason = "Must inline into fused signed assignment to retain convergence-point codegen"
)]
#[inline(always)]
fn normalize_sign_after_sub(
    value: &mut InternalMpInt,
    underflow: bool,
    a: &InternalMpUint,
    b: &InternalMpUint,
    positive_sign: bool,
    negative_sign: bool,
) {
    if value.abs.is_zero() {
        value.is_positive = true;
    } else if underflow {
        negate_normalized_inplace(&mut value.abs, max(a.limbs().len(), b.limbs().len()));
        value.is_positive = negative_sign;
    } else {
        value.is_positive = positive_sign;
    }
}
