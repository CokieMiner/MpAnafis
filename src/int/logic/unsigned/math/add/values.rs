//! Owned addition and subtraction results.

#![expect(
    unsafe_code,
    reason = "Normalized operand lengths prove infallible sizing; disjoint output spans are initialized before exposing their logical length."
)]

use core::{
    cmp::{max, min},
    ptr::{copy_nonoverlapping, write_bytes},
};

use alloc::vec::Vec;

use super::{Addition, ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};

impl InternalMpUint {
    /// Computes `self + rhs`, returning a new value.
    #[expect(
        clippy::inline_always,
        reason = "Inlining constructor arithmetic allows the compiler to optimize small-size stack buffers directly."
    )]
    #[inline(always)]
    #[must_use]
    pub fn add(&self, rhs: &Self) -> Self {
        let a_limbs = self.limbs();
        let b_limbs = rhs.limbs();
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();

        if a_len == 0 {
            return rhs.clone();
        }
        if b_len == 0 {
            return self.clone();
        }

        let max_len = max(a_len, b_len);
        let short_len = min(a_len, b_len);

        if max_len <= INLINE_LIMBS {
            let mut arr = [0_usize; INLINE_LIMBS];
            let dst = arr.as_mut_ptr();

            // SAFETY: both aligned sources contain short_len initialized limbs.
            // The fresh array is aligned, disjoint from both inputs, and covers
            // max_len <= INLINE_LIMBS; only its output prefix is written.
            let mut carry = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    dst,
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                    short_len,
                )
            };

            let (long_limbs, long_len) = if a_len >= b_len {
                (a_limbs, a_len)
            } else {
                (b_limbs, b_len)
            };
            if long_len > short_len {
                // SAFETY: this branch proves short_len < long_len.
                let rem = unsafe { long_len.unchecked_sub(short_len) };
                // SAFETY: short_len + rem == long_len <= INLINE_LIMBS bounds
                // the fresh array tail and initialized source tail; their
                // aligned spans are disjoint and the full output tail is written.
                unsafe {
                    carry = Addition::copy_tail_with_carry(
                        dst.add(short_len),
                        long_limbs.as_ptr().add(short_len),
                        rem,
                        carry,
                    );
                }
            }

            if carry == 0 {
                // SAFETY: `max_len <= INLINE_LIMBS <= u8::MAX`.
                let len = unsafe { u8::try_from(max_len).unwrap_unchecked() };
                return Self {
                    repr: UintRepr::Inline { len, limbs: arr },
                };
            }
            if max_len < INLINE_LIMBS {
                // SAFETY: `max_len < INLINE_LIMBS`, so the slot is in bounds.
                unsafe {
                    *dst.add(max_len) = carry;
                }
                // SAFETY: `max_len + 1 <= INLINE_LIMBS <= u8::MAX`.
                let len = unsafe { u8::try_from(max_len.unchecked_add(1)).unwrap_unchecked() };
                return Self {
                    repr: UintRepr::Inline { len, limbs: arr },
                };
            }

            let mut limbs = Vec::with_capacity(INLINE_LIMBS + 1);
            // SAFETY: capacity is `INLINE_LIMBS + 1`; the copy initializes the
            // first `INLINE_LIMBS` slots and the next write initializes the carry.
            unsafe {
                copy_nonoverlapping(arr.as_ptr(), limbs.as_mut_ptr(), INLINE_LIMBS);
                *limbs.as_mut_ptr().add(INLINE_LIMBS) = carry;
                limbs.set_len(INLINE_LIMBS + 1);
            }
            return Self {
                repr: UintRepr::Heap(limbs),
            };
        }

        // SAFETY: max_len is the length of an existing Limb slice. Its byte
        // span is at most isize::MAX, so one additional element fits usize
        // on 16-, 32-, and 64-bit targets. Vec validates the allocation layout.
        let capacity = unsafe { max_len.unchecked_add(1) };
        let mut limbs: Vec<Limb> = Vec::with_capacity(capacity);
        let dst = limbs.spare_capacity_mut().as_mut_ptr().cast::<Limb>();

        // SAFETY: both aligned sources contain short_len initialized limbs.
        // The fresh Vec owns max_len + 1 aligned writable slots, disjoint from
        // both inputs; the kernel initializes its prefix without reading it.
        let mut carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(dst, a_limbs.as_ptr(), b_limbs.as_ptr(), short_len)
        };

        let (long_limbs, long_len) = if a_len >= b_len {
            (a_limbs, a_len)
        } else {
            (b_limbs, b_len)
        };
        if long_len > short_len {
            // SAFETY: this branch proves short_len < long_len.
            let rem = unsafe { long_len.unchecked_sub(short_len) };
            // SAFETY: short_len + rem == long_len == max_len bounds the
            // initialized source tail and fresh, aligned output tail. They are
            // disjoint, and the kernel initializes every output slot in the tail.
            unsafe {
                carry = Addition::copy_tail_with_carry(
                    dst.add(short_len),
                    long_limbs.as_ptr().add(short_len),
                    rem,
                    carry,
                );
            }
        }

        // SAFETY: the kernel and optional tail copy initialized all first
        // `max_len` slots. A nonzero carry initializes the one additional slot
        // before the logical length is committed.
        unsafe {
            if carry != 0 {
                *dst.add(max_len) = carry;
                limbs.set_len(capacity);
            } else {
                limbs.set_len(max_len);
            }
        }

        // The sum has at least max_len > INLINE_LIMBS limbs. The longer
        // operand is normalized and an appended carry is nonzero, so neither
        // inline conversion nor a high-zero scan is needed.
        Self {
            repr: UintRepr::Heap(limbs),
        }
    }

    /// Computes `self - rhs`.
    ///
    /// `self` must be greater than or equal to `rhs`.
    #[expect(
        clippy::inline_always,
        reason = "Inlining constructor subtraction allows the compiler to optimize small-size stack buffers directly."
    )]
    #[inline(always)]
    #[must_use]
    pub fn sub(&self, rhs: &Self) -> Self {
        let a = self;
        let b = rhs;
        debug_assert!(a >= b, "internal unsigned subtraction requires self >= rhs");
        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();

        if b_len == 0 {
            // a - 0 = a; cloning a normalized operand needs no subtraction
            // kernel or high-zero scan, including short heap-backed values.
            return a.clone();
        }

        if a_len <= INLINE_LIMBS {
            let mut arr = [0_usize; INLINE_LIMBS];
            let dst = arr.as_mut_ptr();
            // SAFETY: a >= b and normalization give b_len <= a_len <= INLINE_LIMBS.
            // Both aligned sources contain b_len initialized limbs; the fresh,
            // disjoint array accepts their write-only prefix, including b_len = 0.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(dst, a_limbs.as_ptr(), b_limbs.as_ptr(), b_len)
            };
            if a_len > b_len {
                // SAFETY: this branch proves b_len < a_len.
                let rem = unsafe { a_len.unchecked_sub(b_len) };
                // SAFETY: b_len + rem == a_len <= INLINE_LIMBS bounds both
                // aligned tails; the source is initialized and the fresh array
                // is disjoint. Every output tail slot is written.
                unsafe {
                    borrow = Addition::copy_tail_with_borrow(
                        dst.add(b_len),
                        a_limbs.as_ptr().add(b_len),
                        rem,
                        borrow,
                    );
                }
            }
            debug_assert_eq!(borrow, 0, "the subtraction precondition prevents borrow");
            let mut final_len = a_len;
            while final_len > 0 {
                // SAFETY: 0 < final_len <= a_len <= INLINE_LIMBS; the prefix
                // and tail kernels initialized every slot below a_len.
                let last = unsafe { final_len.unchecked_sub(1) };
                // SAFETY: last < a_len bounds this initialized output read.
                if unsafe { *dst.add(last) != 0 } {
                    break;
                }
                final_len = last;
            }
            // SAFETY: `final_len <= INLINE_LIMBS <= u8::MAX`.
            let len = unsafe { u8::try_from(final_len).unwrap_unchecked() };
            return Self {
                repr: UintRepr::Inline { len, limbs: arr },
            };
        }

        let mut result = Self::with_capacity(a_len);
        let underflowed = result.assign_difference(a, b);
        debug_assert!(
            !underflowed,
            "the subtraction precondition prevents underflow"
        );
        result
    }

    /// Computes `self - rhs` and reports unsigned underflow.
    ///
    /// The returned value is the fixed-width residue when underflow occurs.
    /// Public checked and panicking boundaries consume the flag before exposing
    /// the value.
    #[inline]
    #[must_use]
    pub fn sub_with_underflow(&self, rhs: &Self) -> (Self, bool) {
        if rhs.is_zero() {
            return (self.clone(), false);
        }
        let max_len = max(self.limbs().len(), rhs.limbs().len());
        let mut result = Self::with_capacity(max_len);
        let underflowed = result.assign_difference(self, rhs);
        (result, underflowed)
    }

    /// Computes `(self - rhs) mod 2^bits` and reports unsigned underflow.
    ///
    /// Both operands must fit in the non-zero `bits`-wide destination. The
    /// subtraction kernel first produces a residue modulo `B^n`, where
    /// `B = 2^LIMB_BITS` and `n` is the wider operand length. On underflow,
    /// extending that negative two's-complement residue from `n` limbs to the
    /// destination width requires filling every new high limb with ones.
    #[inline]
    #[must_use]
    pub fn wrapping_sub_with_underflow(&self, rhs: &Self, bits: usize) -> (Self, bool) {
        debug_assert!(bits != 0, "a bounded precision is non-zero");
        debug_assert!(
            self.significant_bits() <= bits && rhs.significant_bits() <= bits,
            "both operands must fit the wrapping destination"
        );

        if rhs.is_zero() {
            // self - 0 = self already fits the validated destination width.
            return (self.clone(), false);
        }

        let residue_limbs = max(self.limbs().len(), rhs.limbs().len());
        let mut result = Self::with_capacity(residue_limbs);
        let underflowed = result.assign_difference(self, rhs);
        let output_limbs = bits.div_ceil(LIMB_BITS);

        if !underflowed {
            // `self - rhs <= self < 2^bits`; the proved destination bound makes
            // a post-subtraction significant-bit scan redundant.
            return (result, false);
        }
        debug_assert!(
            output_limbs >= residue_limbs,
            "operands that fit the destination cannot use more destination limbs"
        );
        let rem = bits % LIMB_BITS;
        if output_limbs == residue_limbs {
            if rem == 0 {
                return (result, true);
            }
            return (result.apply_wrapping(bits), true);
        }

        // The normalized residue may have discarded zero high limbs within its
        // original `residue_limbs`-limb width. Restore those zeros before sign
        // extending the negative residue with ones through `output_limbs`.
        result.resize(residue_limbs);
        let mut pending = result.prepare_limb_write(output_limbs);
        // SAFETY: the restored residue prefix is initialized and the guard
        // reserves output_limbs slots. This branch proves output_limbs exceeds
        // residue_limbs, so the suffix length is representable and every new
        // limb is written as all ones without reading spare capacity.
        unsafe {
            write_bytes(
                pending.as_mut_ptr().add(residue_limbs),
                0xff,
                output_limbs.unchecked_sub(residue_limbs),
            );
        }
        // SAFETY: the restored residue and sign-extension suffix initialize
        // the entire output_limbs-limb allocation before its length is exposed.
        let limbs = unsafe { pending.commit() };

        if rem != 0 {
            // SAFETY: 0 < rem < LIMB_BITS, so the difference is in 1..LIMB_BITS
            // on every target. Shifting MAX gives exactly rem low one bits.
            let mask = Limb::MAX >> unsafe { LIMB_BITS.unchecked_sub(rem) };
            // SAFETY: `output_limbs > residue_limbs` proves `limbs` is non-empty.
            unsafe {
                *limbs.last_mut().unwrap_unchecked() &= mask;
            }
        }

        (result, true)
    }
}
