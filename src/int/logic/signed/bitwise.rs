//! Signed integer bitwise, shift, and two's-complement logic operations.

#![allow(
    unsafe_code,
    reason = "Inline indices satisfy i < active_len <= INLINE_LIMBS; heap loops use shared_len = min(lhs_len, rhs_len), then i < lhs_len, with a destination allocation of lhs_len slots before set_len."
)]

use core::{
    cmp::min,
    ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr},
};

use alloc::vec::Vec;

use super::{INLINE_LIMBS, InternalMpInt, InternalMpUint, Limb, UintRepr};

#[derive(Clone, Copy)]
enum SignedBitwiseOp {
    And,
    Or,
    Xor,
}

// ---- Macro for BitAnd / BitOr / BitXor traits ----

macro_rules! impl_bitwise {
    ($trait:ident, $method:ident, $op:ident) => {
        impl $trait<Self> for InternalMpInt {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: Self) -> Self::Output {
                $trait::$method(&self, &rhs)
            }
        }

        impl $trait<&Self> for InternalMpInt {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &Self) -> Self::Output {
                $trait::$method(&self, rhs)
            }
        }

        impl $trait<InternalMpInt> for &InternalMpInt {
            type Output = InternalMpInt;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: InternalMpInt) -> Self::Output {
                $trait::$method(self, &rhs)
            }
        }

        impl $trait<&InternalMpInt> for &InternalMpInt {
            type Output = InternalMpInt;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &InternalMpInt) -> InternalMpInt {
                signed_bitwise_op(self, rhs, SignedBitwiseOp::$op)
            }
        }
    };
}

impl_bitwise!(BitAnd, bitand, And);
impl_bitwise!(BitOr, bitor, Or);
impl_bitwise!(BitXor, bitxor, Xor);

// ---- Not trait ----

impl Not for InternalMpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn not(self) -> Self::Output {
        Not::not(&self)
    }
}

impl Not for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    #[track_caller]
    fn not(self) -> InternalMpInt {
        let mut abs = self.abs.clone();
        if self.is_positive {
            abs.increment();
            InternalMpInt {
                abs,
                is_positive: false,
            }
            .normalized()
        } else {
            abs.decrement();
            InternalMpInt {
                abs,
                is_positive: true,
            }
            .normalized()
        }
    }
}

// ---- Shl trait ----

impl Shl<usize> for InternalMpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn shl(self, rhs: usize) -> Self::Output {
        Shl::shl(&self, rhs)
    }
}

impl Shl<usize> for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    #[track_caller]
    fn shl(self, rhs: usize) -> InternalMpInt {
        InternalMpInt {
            abs: self.abs.shl(rhs),
            is_positive: self.is_positive,
        }
        .normalized()
    }
}

// ---- Shr trait ----

impl Shr<usize> for InternalMpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn shr(self, rhs: usize) -> Self::Output {
        Shr::shr(&self, rhs)
    }
}

impl Shr<usize> for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    #[track_caller]
    fn shr(self, rhs: usize) -> InternalMpInt {
        if self.is_positive {
            InternalMpInt {
                abs: self.abs.shr(rhs),
                is_positive: true,
            }
            .normalized()
        } else {
            let mut adjusted = self.abs.shr(rhs);
            if self.abs.has_any_bits_set_below(rhs) {
                adjusted.increment();
            }
            InternalMpInt {
                abs: adjusted,
                is_positive: false,
            }
            .normalized()
        }
    }
}

impl InternalMpInt {
    /// Left-shifts the signed integer in place by `rhs` bits.
    #[inline]
    pub fn shl_assign(&mut self, rhs: usize) {
        self.abs.shl_assign(rhs);
        if self.abs.is_zero() {
            self.is_positive = true;
        }
    }

    /// Right-shifts the signed integer in place by `rhs` bits (arithmetic right shift).
    #[inline]
    pub fn shr_assign(&mut self, rhs: usize) {
        if self.is_positive {
            self.abs.shr_assign(rhs);
        } else {
            let increment_needed = self.abs.has_any_bits_set_below(rhs);
            self.abs.shr_assign(rhs);
            if increment_needed {
                self.abs.increment();
            }
            if self.abs.is_zero() {
                self.is_positive = true;
            }
        }
    }
}

// --- Internal helpers ---

/// Returns `abs - 1` for a strictly negative sign-magnitude value.
#[inline]
#[must_use]
fn negative_predecessor(abs: &InternalMpUint) -> InternalMpUint {
    let mut pred = abs.clone();
    pred.decrement();
    pred
}

/// Builds the negative integer `!(pred)`, equivalent to `-(pred + 1)`.
#[inline]
#[must_use]
fn negative_from_predecessor(mut pred: InternalMpUint) -> InternalMpInt {
    pred.increment();
    InternalMpInt {
        abs: pred,
        is_positive: false,
    }
    .normalized()
}

/// Computes `lhs & !rhs` over the finite width of `lhs`.
#[inline]
#[must_use]
fn bitand_not(lhs: &InternalMpUint, rhs: &InternalMpUint) -> InternalMpUint {
    if let (
        UintRepr::Inline {
            len: lhs_len,
            limbs: lhs_limbs,
        },
        UintRepr::Inline {
            len: rhs_len,
            limbs: rhs_limbs,
        },
    ) = (&lhs.repr, &rhs.repr)
    {
        let active_len = usize::from(*lhs_len);
        let rhs_active_len = usize::from(*rhs_len);
        let mut limbs = [0; INLINE_LIMBS];
        for (i, limb) in limbs.iter_mut().enumerate().take(active_len) {
            let rhs_limb = if i < rhs_active_len {
                // SAFETY: i < rhs_active_len <= INLINE_LIMBS, so the rhs limb exists.
                unsafe { *rhs_limbs.get_unchecked(i) }
            } else {
                0
            };
            // SAFETY: i < active_len <= INLINE_LIMBS, so the lhs limb exists.
            *limb = unsafe { *lhs_limbs.get_unchecked(i) } & !rhs_limb;
        }
        let mut result = InternalMpUint {
            repr: UintRepr::Inline {
                len: *lhs_len,
                limbs,
            },
        };
        result.normalize();
        return result;
    }

    let lhs_limbs = lhs.limbs();
    let rhs_limbs = rhs.limbs();
    let lhs_len = lhs_limbs.len();
    let rhs_len = rhs_limbs.len();
    let shared_len = min(lhs_len, rhs_len);
    let mut limbs: Vec<Limb> = Vec::with_capacity(lhs_len);
    let dst = limbs.as_mut_ptr();

    for i in 0..shared_len {
        // SAFETY: `dst` has capacity lhs_len and i < shared_len <= lhs_len.
        // Both source indices are valid because shared_len <= lhs_len, rhs_len.
        // Each destination slot is written exactly once before set_len.
        unsafe {
            dst.add(i)
                .write(*lhs_limbs.get_unchecked(i) & !*rhs_limbs.get_unchecked(i));
        }
    }
    for i in shared_len..lhs_len {
        // SAFETY: `dst` has capacity lhs_len and i < lhs_len.
        // The lhs source index is valid; rhs is conceptually zero beyond rhs_len.
        // Each destination slot is written exactly once before set_len.
        unsafe {
            dst.add(i).write(*lhs_limbs.get_unchecked(i));
        }
    }
    // SAFETY: The two loops above initialized exactly lhs_len slots.
    unsafe {
        limbs.set_len(lhs_len);
    }
    InternalMpUint::from_limbs(limbs)
}

/// Perform a signed bitwise operation (AND/OR/XOR) via two's complement.
/// When both operands are positive, directly delegates to the unsigned operation.
#[inline]
#[must_use]
fn signed_bitwise_op(
    self_val: &InternalMpInt,
    rhs: &InternalMpInt,
    op: SignedBitwiseOp,
) -> InternalMpInt {
    match (self_val.is_positive, rhs.is_positive) {
        (true, true) => InternalMpInt {
            abs: unsigned_bitwise_op(&self_val.abs, &rhs.abs, op),
            is_positive: true,
        }
        .normalized(),
        (true, false) => positive_negative_bitwise(&self_val.abs, &rhs.abs, op),
        (false, true) => positive_negative_bitwise(&rhs.abs, &self_val.abs, op),
        (false, false) => negative_negative_bitwise(&self_val.abs, &rhs.abs, op),
    }
}

#[inline]
#[must_use]
fn unsigned_bitwise_op(
    lhs: &InternalMpUint,
    rhs: &InternalMpUint,
    op: SignedBitwiseOp,
) -> InternalMpUint {
    match op {
        SignedBitwiseOp::And => lhs.bitand(rhs),
        SignedBitwiseOp::Or => lhs.bitor(rhs),
        SignedBitwiseOp::Xor => lhs.bitxor(rhs),
    }
}

#[inline]
#[must_use]
fn positive_negative_bitwise(
    positive_abs: &InternalMpUint,
    negative_abs: &InternalMpUint,
    op: SignedBitwiseOp,
) -> InternalMpInt {
    let neg_pred = negative_predecessor(negative_abs);
    match op {
        SignedBitwiseOp::And => InternalMpInt {
            abs: bitand_not(positive_abs, &neg_pred),
            is_positive: true,
        }
        .normalized(),
        SignedBitwiseOp::Or => negative_from_predecessor(bitand_not(&neg_pred, positive_abs)),
        SignedBitwiseOp::Xor => negative_from_predecessor(neg_pred.bitxor(positive_abs)),
    }
}

#[inline]
#[must_use]
fn negative_negative_bitwise(
    lhs_abs: &InternalMpUint,
    rhs_abs: &InternalMpUint,
    op: SignedBitwiseOp,
) -> InternalMpInt {
    let lhs_pred = negative_predecessor(lhs_abs);
    let rhs_pred = negative_predecessor(rhs_abs);
    match op {
        SignedBitwiseOp::And => negative_from_predecessor(lhs_pred.bitor(&rhs_pred)),
        SignedBitwiseOp::Or => negative_from_predecessor(lhs_pred.bitand(&rhs_pred)),
        SignedBitwiseOp::Xor => InternalMpInt {
            abs: lhs_pred.bitxor(&rhs_pred),
            is_positive: true,
        }
        .normalized(),
    }
}
