//! Owned addition and subtraction results.

use core::{
    cmp::{max, min},
    ptr::copy_nonoverlapping,
};

use alloc::vec::Vec;

use super::{Addition, ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};

impl InternalMpUint {
    /// Computes `self + rhs`, returning a new value.
    #[allow(
        unsafe_code,
        reason = "Vec and stack array arithmetic bypasses enum dispatch for fewer branches"
    )]
    #[allow(
        clippy::many_single_char_names,
        reason = "a_len, b_len are descriptive; i/c are traditional loop/carry names"
    )]
    #[allow(
        clippy::uninit_vec,
        reason = "All max_len slots are immediately filled by the assembly kernel or copy below"
    )]
    #[allow(
        clippy::inline_always,
        reason = "Inlining constructor arithmetic allows the compiler to optimize small-size stack buffers directly."
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "The inline fast path and heap fallback share similar fused carry-propagation structure"
    )]
    #[inline(always)]
    #[must_use]
    pub fn add(&self, rhs: &Self) -> Self {
        let a = self;
        let b = rhs;
        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();

        if a_len == 0 {
            return b.clone();
        }
        if b_len == 0 {
            return a.clone();
        }

        let max_len = max(a_len, b_len);
        let short_len = min(a_len, b_len);

        if max_len <= INLINE_LIMBS {
            let mut arr = [0_usize; INLINE_LIMBS];
            let dst = arr.as_mut_ptr();

            // SAFETY: `dst`, `a_limbs`, and `b_limbs` have at least `short_len`
            // elements.
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
                let rem = long_len.wrapping_sub(short_len);
                // SAFETY: both pointers are valid for `rem` elements.
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
                let len = unsafe { u8::try_from(max_len.wrapping_add(1)).unwrap_unchecked() };
                return Self {
                    repr: UintRepr::Inline { len, limbs: arr },
                };
            }

            let mut limbs = Vec::with_capacity(INLINE_LIMBS.wrapping_add(1));
            // SAFETY: capacity is `INLINE_LIMBS + 1`; the copy initializes the
            // first `INLINE_LIMBS` slots and the next write initializes the carry.
            unsafe {
                copy_nonoverlapping(arr.as_ptr(), limbs.as_mut_ptr(), INLINE_LIMBS);
                *limbs.as_mut_ptr().add(INLINE_LIMBS) = carry;
                limbs.set_len(INLINE_LIMBS.wrapping_add(1));
            }
            return Self {
                repr: UintRepr::Heap(limbs),
            };
        }

        // SAFETY: capacity is `max_len + 1`; the kernel and tail copy initialize
        // all first `max_len` slots before any read.
        let mut limbs: Vec<Limb> = unsafe {
            let mut buffer = Vec::with_capacity(max_len.wrapping_add(1));
            buffer.set_len(max_len);
            buffer
        };
        let dst = limbs.as_mut_ptr();

        // SAFETY: all three pointers have at least `short_len` elements.
        let mut carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(dst, a_limbs.as_ptr(), b_limbs.as_ptr(), short_len)
        };

        let (long_limbs, long_len) = if a_len >= b_len {
            (a_limbs, a_len)
        } else {
            (b_limbs, b_len)
        };
        if long_len > short_len {
            let rem = long_len.wrapping_sub(short_len);
            // SAFETY: both pointers are valid for `rem` elements.
            unsafe {
                carry = Addition::copy_tail_with_carry(
                    dst.add(short_len),
                    long_limbs.as_ptr().add(short_len),
                    rem,
                    carry,
                );
            }
        }

        if carry != 0 {
            // SAFETY: the allocation has capacity for `max_len + 1`.
            unsafe {
                *dst.add(max_len) = carry;
                limbs.set_len(max_len.wrapping_add(1));
            }
        }

        // SAFETY: the longer operand is normalized when no carry is appended, and
        // an appended carry is non-zero.
        unsafe { Self::from_limbs_normalized(limbs) }
    }

    /// Computes `self - rhs` and reports unsigned underflow.
    ///
    /// The returned value is the fixed-width residue when underflow occurs.
    /// Public checked and panicking boundaries consume the flag before exposing
    /// the value.
    #[inline]
    #[must_use]
    pub fn sub_with_underflow(&self, rhs: &Self) -> (Self, bool) {
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
    #[allow(
        unsafe_code,
        reason = "The proved destination and residue widths make the sign-extension tail and top-limb access in bounds"
    )]
    #[inline]
    #[must_use]
    pub fn wrapping_sub_with_underflow(&self, rhs: &Self, bits: usize) -> (Self, bool) {
        debug_assert!(bits != 0, "a bounded precision is non-zero");
        debug_assert!(
            self.significant_bits() <= bits && rhs.significant_bits() <= bits,
            "both operands must fit the wrapping destination"
        );

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
        let rem = bits.wrapping_rem(LIMB_BITS);
        if output_limbs == residue_limbs {
            if rem == 0 {
                return (result, true);
            }
            return (result.apply_wrapping_with_mask(output_limbs, rem), true);
        }

        // The normalized residue may have discarded zero high limbs within its
        // original `residue_limbs`-limb width. Restore those zeros before sign
        // extending the negative residue with ones through `output_limbs`.
        result.resize(residue_limbs);
        result.resize(output_limbs);
        let limbs = result.limbs_mut();
        // SAFETY: both resizes completed, and this branch proves
        // `residue_limbs < output_limbs = limbs.len()`.
        unsafe {
            limbs
                .get_unchecked_mut(residue_limbs..output_limbs)
                .fill(Limb::MAX);
        }

        if rem != 0 {
            // SAFETY: `rem < LIMB_BITS <= u32::MAX` on every supported limb
            // width, and `output_limbs > residue_limbs`, so the last limb exists.
            let shift = unsafe { u32::try_from(rem).unwrap_unchecked() };
            let mask = 1_usize.wrapping_shl(shift).wrapping_sub(1);
            // SAFETY: `output_limbs > residue_limbs` proves `limbs` is non-empty.
            unsafe {
                *limbs.last_mut().unwrap_unchecked() &= mask;
            }
        }

        (result, true)
    }

    /// Computes `self - rhs`.
    ///
    /// `self` must be greater than or equal to `rhs`.
    #[allow(
        clippy::inline_always,
        reason = "Inlining constructor subtraction allows the compiler to optimize small-size stack buffers directly."
    )]
    #[allow(
        unsafe_code,
        reason = "Bypasses enum dispatch for inline stack arithmetic"
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

        if a_len <= INLINE_LIMBS {
            let mut arr = [0_usize; INLINE_LIMBS];
            let dst = arr.as_mut_ptr();
            // SAFETY: `b_len <= a_len <= INLINE_LIMBS`, so all pointers cover
            // `b_len` elements.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(dst, a_limbs.as_ptr(), b_limbs.as_ptr(), b_len)
            };
            if a_len > b_len {
                let rem = a_len.wrapping_sub(b_len);
                // SAFETY: both tail pointers are valid for `rem` elements.
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
                // SAFETY: `0 < final_len <= a_len <= INLINE_LIMBS`.
                if unsafe { *dst.add(final_len.wrapping_sub(1)) != 0 } {
                    break;
                }
                final_len = final_len.wrapping_sub(1);
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
}
