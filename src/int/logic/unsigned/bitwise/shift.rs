//! Unsigned logical shift operations and in-place shift helpers.

#![allow(
    unsafe_code,
    reason = "Bypassing bounds checks on pre-sized result arrays to ensure branchless execution and prevent insertion of runtime panic paths in hot loops."
)]

use core::{
    ptr::{copy, copy_nonoverlapping, write_bytes},
    slice::from_raw_parts,
};

use alloc::vec::Vec;

use super::{ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};
impl InternalMpUint {
    /// Left-shifts the integer by `shift` bits (padded with zero limbs).
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo Limb::BITS fits in u32 even on 16-bit targets (where Limb::BITS is 16): avoids branchy range checks and compiles to zero-cost branchless register truncation."
    )]
    pub fn shl(&self, shift: usize) -> Self {
        if shift == 0 || self.is_zero() {
            return self.clone();
        }
        let word_shift = shift.wrapping_div(LIMB_BITS);

        let bit_shift = shift.wrapping_rem(LIMB_BITS);
        let src = self.limbs();
        let src_len = src.len();

        if bit_shift == 0 {
            // No overflow: `word_shift <= shift >> log2(LIMB_BITS)` and every
            // valid Vec (or inline array) satisfies `src_len <= isize::MAX`;
            // on every supported width
            // `isize::MAX + (usize::MAX >> log2(LIMB_BITS)) < usize::MAX`
            // (64-bit: 2^63-1 + 2^58-1, 32-bit: 2^31-1 + 2^27-1, 16-bit:
            // 2^15-1 + 2^11-1), so `src_len + word_shift` cannot wrap.
            let result_len = src_len.wrapping_add(word_shift);
            let mut limbs: Vec<Limb> = Vec::with_capacity(result_len);
            limbs.resize(word_shift, 0);
            limbs.extend_from_slice(src);
            // SAFETY: src is normalized, so the top limb of the aligned
            // result is src[src_len - 1] != 0.
            return unsafe { Self::from_limbs_normalized(limbs) };
        }

        let drop = LIMB_BITS.wrapping_sub(bit_shift);
        // SAFETY: `src` is normalized and non-zero (early return above), so
        // `src_len >= 1`, `src_len - 1` cannot underflow, and the index is in
        // bounds. The last limb is read to compute the carry limb.
        let carry = unsafe { *src.get_unchecked(src_len.wrapping_sub(1)) } >> drop;
        // No overflow: same bound as the aligned path above.
        let result_len = src_len
            .wrapping_add(word_shift)
            .wrapping_add(1)
            .wrapping_sub(usize::from(carry == 0));

        if result_len <= INLINE_LIMBS {
            let mut out = [0; INLINE_LIMBS];
            // SAFETY: the kernel writes src_len limbs at offset word_shift;
            // word_shift + src_len <= result_len <= INLINE_LIMBS, and out
            // holds INLINE_LIMBS limbs. The spans are disjoint by construction.
            let kernel_carry = unsafe {
                ArchKernels::lshift_into_unchecked(
                    out.as_mut_ptr().add(word_shift),
                    src.as_ptr(),
                    src_len,
                    bit_shift as u32,
                )
            };
            debug_assert_eq!(
                kernel_carry, carry,
                "kernel carry must equal the scalar-computed top-limb carry"
            );
            if carry != 0 {
                // SAFETY: `carry != 0` implies `result_len >= src_len >= 1`,
                // so `result_len - 1` is a valid in-bounds index of `out`.
                unsafe {
                    *out.get_unchecked_mut(result_len.wrapping_sub(1)) = carry;
                }
            }
            return Self {
                repr: UintRepr::Inline {
                    len: result_len as u8,
                    limbs: out,
                },
            };
        }

        let mut limbs: Vec<Limb> = Vec::with_capacity(result_len);
        // SAFETY: capacity == result_len covers the kernel's src_len writes
        // plus the optional carry write at result_len - 1; the word_shift low
        // limbs are zeroed first because the kernel only writes the shifted
        // limbs at offset word_shift and the result's low limbs are zero by
        // construction. Both happen before any read of `limbs`.
        let kernel_carry = unsafe {
            limbs.set_len(result_len);
            write_bytes(limbs.as_mut_ptr(), 0, word_shift);
            ArchKernels::lshift_into_unchecked(
                limbs.as_mut_ptr().add(word_shift),
                src.as_ptr(),
                src_len,
                bit_shift as u32,
            )
        };
        debug_assert_eq!(
            kernel_carry, carry,
            "kernel carry must equal the scalar-computed top-limb carry"
        );
        if carry != 0 {
            // SAFETY: `carry != 0` implies `result_len >= src_len >= 1`, so
            // `result_len - 1` is a valid index of `limbs`.
            unsafe {
                *limbs.get_unchecked_mut(result_len.wrapping_sub(1)) = carry;
            }
        }
        // SAFETY: the top limb is either `carry` (non-zero by the branch) or
        // the merged top `(src[top] << bit_shift) | (src[top-1] >> drop)`,
        // which is non-zero because carry == 0 implies `src[top] << bit_shift
        // != 0` — src[top] != 0 and shifting it left by bit_shift cannot
        // overflow when carry == 0.
        unsafe { Self::from_limbs_normalized(limbs) }
    }

    /// Right-shifts the integer by `shift` bits (logical shift).
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo Limb::BITS fits in u32 even on 16-bit targets (where Limb::BITS is 16): avoids branchy range checks and compiles to zero-cost branchless register truncation."
    )]
    pub fn shr(&self, shift: usize) -> Self {
        if shift == 0 || self.is_zero() {
            return self.clone();
        }
        let word_shift = shift.wrapping_div(LIMB_BITS);

        let bit_shift = shift.wrapping_rem(LIMB_BITS);
        let src = self.limbs();
        let src_len = src.len();

        if word_shift >= src_len {
            return Self::zero();
        }

        let result_len = src_len.wrapping_sub(word_shift);

        if bit_shift == 0 {
            if result_len <= INLINE_LIMBS {
                let mut out = [0; INLINE_LIMBS];
                // SAFETY: `word_shift + result_len == src_len`, so
                // `src[word_shift..src_len]` is in bounds, and `result_len <=
                // INLINE_LIMBS == out.len()`; the buffers are disjoint.
                unsafe {
                    copy_nonoverlapping(src.as_ptr().add(word_shift), out.as_mut_ptr(), result_len);
                }
                return Self {
                    repr: UintRepr::Inline {
                        len: result_len as u8,
                        limbs: out,
                    },
                };
            }
            let mut limbs: Vec<Limb> = Vec::with_capacity(result_len);
            // SAFETY: `word_shift + result_len == src_len`, so the source
            // suffix is in bounds; `limbs` has capacity `result_len`, which
            // covers the full append before `from_limbs_normalized` reads it.
            unsafe {
                limbs.extend_from_slice(from_raw_parts(src.as_ptr().add(word_shift), result_len));
            }
            // SAFETY: src is normalized, so the copied suffix has a non-zero
            // top limb (src[src_len - 1]).
            return unsafe { Self::from_limbs_normalized(limbs) };
        }

        // SAFETY: `src_len >= 1` (normalized, non-zero), so `src_len - 1`
        // cannot underflow and indexes the top limb, which decides whether
        // the result shrinks by one limb.
        let top_is_shifted_out =
            unsafe { *src.get_unchecked(src_len.wrapping_sub(1)) >> bit_shift == 0 };
        let exact_len = result_len.wrapping_sub(usize::from(top_is_shifted_out));
        if exact_len == 0 {
            return Self::zero();
        }

        if result_len <= INLINE_LIMBS {
            let mut out = [0; INLINE_LIMBS];
            // SAFETY: the kernel writes result_len limbs at out[0..];
            // result_len <= INLINE_LIMBS, and the spans are disjoint.
            unsafe {
                let _ = ArchKernels::rshift_into_unchecked(
                    out.as_mut_ptr(),
                    src.as_ptr().add(word_shift),
                    result_len,
                    bit_shift as u32,
                );
            }
            return Self {
                repr: UintRepr::Inline {
                    len: exact_len as u8,
                    limbs: out,
                },
            };
        }

        let mut limbs: Vec<Limb> = Vec::with_capacity(result_len);
        // SAFETY: capacity == result_len covers every limb the kernel writes
        // before any read of `limbs`; the normalized length exact_len is then
        // applied below. The kernel's carry limb is dropped: for a logical
        // right shift it is always zero (the low limbs below `word_shift` are
        // discarded by the caller), so ignoring it cannot lose data.
        let _kernel_carry = unsafe {
            limbs.set_len(result_len);
            ArchKernels::rshift_into_unchecked(
                limbs.as_mut_ptr(),
                src.as_ptr().add(word_shift),
                result_len,
                bit_shift as u32,
            )
        };
        // SAFETY: the top limb is provably non-zero: when the overflow
        // limb `src[src_len-1] >> bit_shift` is zero, src[src_len-1] <
        // 2^bit_shift, so the merged limb below contains
        // `src[src_len-1] << (LIMB_BITS - bit_shift) != 0`.
        unsafe {
            limbs.set_len(exact_len);
            Self::from_limbs_normalized(limbs)
        }
    }

    /// Left-shifts the integer by `shift` bits in-place.
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo Limb::BITS fits in u32 even on 16-bit targets (where Limb::BITS is 16): avoids branchy range checks and compiles to zero-cost branchless register truncation."
    )]
    pub fn shl_assign(&mut self, shift: usize) {
        if shift == 0 || self.is_zero() {
            return;
        }
        let word_shift = shift.wrapping_div(LIMB_BITS);

        let bit_shift: u32 = shift.wrapping_rem(LIMB_BITS) as u32;

        if word_shift > 0 {
            let src_len = self.limbs().len();
            // No overflow: same bound as `InternalMpUint::shl`; `word_shift
            // <= shift >> log2(LIMB_BITS)` and `src_len <= isize::MAX`, so
            // `src_len + word_shift` cannot wrap on any supported width.
            let new_len = src_len.wrapping_add(word_shift);
            self.reserve(word_shift);
            // SAFETY: reserved sufficient capacity, new_len = src_len + word_shift.
            unsafe {
                self.set_len(new_len);
            }
            let ptr = self.limbs_mut().as_mut_ptr();
            // SAFETY: shifting in-place with memmove, then zeroing lower limbs.
            unsafe {
                copy(ptr, ptr.add(word_shift), src_len);
                write_bytes(ptr, 0, word_shift);
            }
        }

        if bit_shift > 0 {
            let len = self.limbs().len();
            let limbs_ptr = self.limbs_mut().as_mut_ptr();
            // SAFETY: `bit_shift` is in (0, LIMB_BITS) by construction. `limbs_ptr` points to
            // `len` valid Limb elements.
            let carry = unsafe { ArchKernels::lshift_unchecked(limbs_ptr, len, bit_shift) };
            if carry != 0 {
                self.reserve(1);
                // SAFETY: reserved 1 capacity, so index `len` is valid
                unsafe {
                    self.set_len(len.wrapping_add(1));
                    *self.limbs_mut().get_unchecked_mut(len) = carry;
                }
            }
        }
        self.normalize();
    }

    /// Right-shifts the integer by `shift` bits in-place.
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo Limb::BITS fits in u32 even on 16-bit targets (where Limb::BITS is 16): avoids branchy range checks and compiles to zero-cost branchless register truncation."
    )]
    pub fn shr_assign(&mut self, shift: usize) {
        if shift == 0 || self.is_zero() {
            return;
        }
        let word_shift = shift.wrapping_div(LIMB_BITS);

        let bit_shift: u32 = shift.wrapping_rem(LIMB_BITS) as u32;

        let src_len = self.limbs().len();
        if word_shift >= src_len {
            // SAFETY: setting len to 0 is safe
            unsafe {
                self.set_len(0);
            }
            self.normalize();
            return;
        }

        if word_shift > 0 {
            let new_len = src_len.wrapping_sub(word_shift);
            let ptr = self.limbs_mut().as_mut_ptr();
            // SAFETY: memory regions may overlap, so use ptr::copy (memmove).
            unsafe {
                copy(ptr.add(word_shift), ptr, new_len);
                self.set_len(new_len);
            }
        }

        if bit_shift > 0 {
            let len = self.limbs().len();
            let limbs_ptr = self.limbs_mut().as_mut_ptr();
            // SAFETY: `bit_shift` is in (0, LIMB_BITS) by construction. `limbs_ptr` points to
            // `len` valid Limb elements.
            unsafe {
                let _ = ArchKernels::rshift_unchecked(limbs_ptr, len, bit_shift);
            }
        }
        self.normalize();
    }
}

#[cfg(test)]
#[path = "tests/shift.rs"]
mod tests;
