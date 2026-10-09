//! Signed integer bitwise, shift, and two's-complement logic operations.

use core::{
    cmp::min,
    ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr},
};

use alloc::vec::Vec;

use super::{INLINE_LIMBS, InternalMpInt, InternalMpUint, Limb, UintRepr};

impl InternalMpInt {
    /// Left-shifts the signed integer in place by `rhs` bits ($x \leftarrow x \cdot 2^{\text{rhs}}$).
    #[inline]
    pub fn shl_assign(&mut self, rhs: usize) {
        // Exact multiplication by 2^rhs preserves nonzero magnitude. Canonical
        // zero is already positive, so neither case requires sign normalization.
        self.abs.shl_assign(rhs);
    }

    /// Right-shifts the signed integer in place by `rhs` bits (arithmetic right shift $\lfloor x / 2^{\text{rhs}} \rfloor$).
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
            // For negative x <= -1, floor division satisfies floor(x / 2^k) <= -1.
            // Magnitude remains non-zero and sign remains negative.
        }
    }
}

macro_rules! impl_bitwise_forwarding {
    ($trait:ident, $method:ident) => {
        impl $trait<Self> for InternalMpInt {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                $trait::$method(&self, &rhs)
            }
        }

        impl $trait<&Self> for InternalMpInt {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &Self) -> Self::Output {
                $trait::$method(&self, rhs)
            }
        }

        impl $trait<InternalMpInt> for &InternalMpInt {
            type Output = InternalMpInt;
            #[inline]
            fn $method(self, rhs: InternalMpInt) -> Self::Output {
                $trait::$method(self, &rhs)
            }
        }
    };
}

impl_bitwise_forwarding!(BitAnd, bitand);
impl_bitwise_forwarding!(BitOr, bitor);
impl_bitwise_forwarding!(BitXor, bitxor);

// A negative integer x has infinite two's-complement bits ~(|x| - 1).

impl BitAnd<&InternalMpInt> for &InternalMpInt {
    type Output = InternalMpInt;
    /// Evaluates bitwise AND across sign quadrants using negative predecessors:
    /// - $(+, +) \implies |a| \ \& \ |b|$
    /// - $(+, -) \implies a \ \& \ \sim (|b| - 1) = a \ \& \ !(|b| - 1)$
    /// - $(-, +) \implies \sim (|a| - 1) \ \& \ b = b \ \& \ !(|a| - 1)$
    /// - $(-, -) \implies \sim (|a| - 1) \ \& \ \sim (|b| - 1) = \sim ((|a| - 1) \ | \ (|b| - 1))$
    #[inline]
    fn bitand(self, rhs: &InternalMpInt) -> InternalMpInt {
        match (self.is_positive, rhs.is_positive) {
            (true, true) => InternalMpInt {
                abs: self.abs.bitand(&rhs.abs),
                is_positive: true,
            },
            (true, false) => {
                let neg_pred = negative_predecessor(&rhs.abs);
                InternalMpInt {
                    abs: bitand_not(&self.abs, &neg_pred),
                    is_positive: true,
                }
            }
            (false, true) => {
                let neg_pred = negative_predecessor(&self.abs);
                InternalMpInt {
                    abs: bitand_not(&rhs.abs, &neg_pred),
                    is_positive: true,
                }
            }
            (false, false) => {
                let lhs_pred = negative_predecessor(&self.abs);
                let rhs_pred = negative_predecessor(&rhs.abs);
                negative_from_predecessor(lhs_pred.bitor(&rhs_pred))
            }
        }
    }
}

impl BitOr<&InternalMpInt> for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    fn bitor(self, rhs: &InternalMpInt) -> InternalMpInt {
        match (self.is_positive, rhs.is_positive) {
            (true, true) => InternalMpInt {
                abs: self.abs.bitor(&rhs.abs),
                is_positive: true,
            },
            (true, false) => {
                let neg_pred = negative_predecessor(&rhs.abs);
                negative_from_predecessor(bitand_not(&neg_pred, &self.abs))
            }
            (false, true) => {
                let neg_pred = negative_predecessor(&self.abs);
                negative_from_predecessor(bitand_not(&neg_pred, &rhs.abs))
            }
            (false, false) => {
                let lhs_pred = negative_predecessor(&self.abs);
                let rhs_pred = negative_predecessor(&rhs.abs);
                negative_from_predecessor(lhs_pred.bitand(&rhs_pred))
            }
        }
    }
}

impl BitXor<&InternalMpInt> for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    fn bitxor(self, rhs: &InternalMpInt) -> InternalMpInt {
        match (self.is_positive, rhs.is_positive) {
            (true, true) => InternalMpInt {
                abs: self.abs.bitxor(&rhs.abs),
                is_positive: true,
            },
            (true, false) => {
                let neg_pred = negative_predecessor(&rhs.abs);
                negative_from_predecessor(neg_pred.bitxor(&self.abs))
            }
            (false, true) => {
                let neg_pred = negative_predecessor(&self.abs);
                negative_from_predecessor(neg_pred.bitxor(&rhs.abs))
            }
            (false, false) => {
                let lhs_pred = negative_predecessor(&self.abs);
                let rhs_pred = negative_predecessor(&rhs.abs);
                InternalMpInt {
                    abs: lhs_pred.bitxor(&rhs_pred),
                    is_positive: true,
                }
            }
        }
    }
}

// Bitwise complement satisfies ~x = -x - 1.

impl Not for InternalMpInt {
    type Output = Self;
    #[inline]
    fn not(mut self) -> Self::Output {
        if self.is_positive {
            self.abs.increment();
            self.is_positive = false;
        } else {
            self.abs.decrement();
            self.is_positive = true;
        }
        self
    }
}

impl Not for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    fn not(self) -> InternalMpInt {
        let mut abs = self.abs.clone();
        if self.is_positive {
            abs.increment();
            InternalMpInt {
                abs,
                is_positive: false,
            }
        } else {
            abs.decrement();
            InternalMpInt {
                abs,
                is_positive: true,
            }
        }
    }
}

impl Shl<usize> for InternalMpInt {
    type Output = Self;
    #[inline]
    fn shl(mut self, rhs: usize) -> Self::Output {
        self.shl_assign(rhs);
        self
    }
}

impl Shl<usize> for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    fn shl(self, rhs: usize) -> InternalMpInt {
        InternalMpInt {
            abs: self.abs.shl(rhs),
            is_positive: self.is_positive,
        }
    }
}

impl Shr<usize> for InternalMpInt {
    type Output = Self;
    #[inline]
    fn shr(mut self, rhs: usize) -> Self::Output {
        self.shr_assign(rhs);
        self
    }
}

impl Shr<usize> for &InternalMpInt {
    type Output = InternalMpInt;
    #[inline]
    fn shr(self, rhs: usize) -> InternalMpInt {
        if self.is_positive {
            InternalMpInt {
                abs: self.abs.shr(rhs),
                is_positive: true,
            }
        } else {
            let mut adjusted = self.abs.shr(rhs);
            if self.abs.has_any_bits_set_below(rhs) {
                adjusted.increment();
            }
            InternalMpInt {
                abs: adjusted,
                is_positive: false,
            }
        }
    }
}

/// Computes `lhs & !rhs` over the finite width of `lhs`.
#[expect(
    unsafe_code,
    reason = "Source lengths bound initialized reads; disjoint aligned destination slots are initialized before publishing the vector length"
)]
#[inline]
#[must_use]
fn bitand_not(lhs: &InternalMpUint, rhs: &InternalMpUint) -> InternalMpUint {
    // 0 <= lhs & !rhs <= lhs, so an inline lhs guarantees an inline result.
    if let UintRepr::Inline {
        len: lhs_len,
        limbs: lhs_limbs,
    } = lhs.repr
    {
        let active_len = usize::from(lhs_len);
        let rhs_limbs = rhs.limbs();
        let rhs_len = rhs_limbs.len();
        let mut limbs = [0; INLINE_LIMBS];
        for (i, slot) in limbs.iter_mut().enumerate().take(active_len) {
            let rhs_limb = if i < rhs_len {
                // SAFETY: i < rhs_len == rhs_limbs.len().
                unsafe { *rhs_limbs.get_unchecked(i) }
            } else {
                0
            };
            // SAFETY: i < active_len <= INLINE_LIMBS == lhs_limbs.len().
            *slot = unsafe { *lhs_limbs.get_unchecked(i) } & !rhs_limb;
        }
        let mut result = InternalMpUint {
            repr: UintRepr::Inline {
                len: lhs_len,
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
        // SAFETY: Vec<Limb> supplies aligned, exclusive storage disjoint from
        // both borrowed inputs. i < shared_len bounds initialized reads from
        // both sources and a destination slot below capacity lhs_len. Each
        // slot is initialized once before set_len.
        unsafe {
            dst.add(i)
                .write(*lhs_limbs.get_unchecked(i) & !*rhs_limbs.get_unchecked(i));
        }
    }
    for i in shared_len..lhs_len {
        // SAFETY: the aligned destination remains exclusive and disjoint from
        // lhs. i < lhs_len bounds the initialized source read and destination
        // slot below capacity. These slots follow the initialized shared prefix.
        unsafe {
            dst.add(i).write(*lhs_limbs.get_unchecked(i));
        }
    }
    // SAFETY: The loops over 0..shared_len and shared_len..lhs_len initialize every
    // element in 0..lhs_len exactly once. The allocation has capacity lhs_len.
    unsafe {
        limbs.set_len(lhs_len);
    }
    InternalMpUint::from_limbs(limbs)
}

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
}
