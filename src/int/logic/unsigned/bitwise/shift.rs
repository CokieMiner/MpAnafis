//! Unsigned logical shift operations and in-place shift helpers.

#![expect(
    unsafe_code,
    reason = "Exact result widths bound shift kernels and overlapping assignment copies; raw writes initialize every limb before length commitment"
)]

use core::ptr::{copy, copy_nonoverlapping, write_bytes};

use alloc::vec::Vec;

use super::{ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};

impl InternalMpUint {
    /// Left-shifts the integer by `shift` bits (padded with zero limbs).
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Shift remainders are below LIMB_BITS <= 64 and inline lengths are at most four, fitting u32 and u8 on every supported target"
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
            // SAFETY: an addressable limb slice has src_len <= isize::MAX,
            // while word_shift <= usize::MAX / LIMB_BITS <= usize::MAX / 16.
            // Their sum fits usize on 16-, 32-, and 64-bit targets, so the
            // checked size has no failure branch at this kernel boundary.
            let result_len = unsafe { src_len.checked_add(word_shift).unwrap_unchecked() };
            if result_len <= INLINE_LIMBS {
                let mut out = [0; INLINE_LIMBS];
                // SAFETY: src is initialized and disjoint from out. The
                // shifted span ends at result_len <= INLINE_LIMBS; the low
                // word_shift slots are zero and the source top stays nonzero.
                unsafe {
                    copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr().add(word_shift), src_len);
                }
                return Self {
                    repr: UintRepr::Inline {
                        len: result_len as u8,
                        limbs: out,
                    },
                };
            }
            let mut limbs: Vec<Limb> = Vec::with_capacity(result_len);
            // SAFETY: the fresh aligned allocation reserves result_len limbs.
            // Zeroing the low word_shift limbs and copying the disjoint src_len
            // initialized source limbs fill that exact span before committing.
            unsafe {
                let dst = limbs.as_mut_ptr();
                write_bytes(dst, 0, word_shift);
                copy_nonoverlapping(src.as_ptr(), dst.add(word_shift), src_len);
                limbs.set_len(result_len);
            }
            // SAFETY: src is normalized, so the top limb of the aligned
            // result is src[src_len - 1] != 0.
            return unsafe { Self::from_limbs_normalized(limbs) };
        }

        // SAFETY: this arm has 0 < bit_shift < LIMB_BITS, so drop lies in 1..W.
        let drop = unsafe { LIMB_BITS.unchecked_sub(bit_shift) };
        // SAFETY: `src` is normalized and non-zero (early return above), so
        // `src_len >= 1`, `src_len - 1` cannot underflow, and the index is in
        // bounds. The last limb is read to compute the carry limb.
        let carry = unsafe { *src.get_unchecked(src_len.unchecked_sub(1)) } >> drop;
        // SAFETY: src_len <= isize::MAX and word_shift <= usize::MAX / 16.
        // Their sum plus at most one carry remains below usize::MAX on every
        // supported target. Infallible extraction removes the impossible size error.
        let result_len = unsafe {
            src_len
                .checked_add(word_shift)
                .and_then(|width| width.checked_add(usize::from(carry != 0)))
                .unwrap_unchecked()
        };

        if result_len <= INLINE_LIMBS {
            let mut out = [0; INLINE_LIMBS];
            // SAFETY: the kernel writes src_len limbs at offset word_shift;
            // word_shift + src_len <= result_len <= INLINE_LIMBS, and out
            // holds INLINE_LIMBS limbs. The spans are disjoint by construction.
            let kernel_carry = unsafe {
                ArchKernels::lshift_into_small_unchecked(
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
                    *out.get_unchecked_mut(result_len.unchecked_sub(1)) = carry;
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
        let dst = limbs.spare_capacity_mut().as_mut_ptr().cast::<Limb>();
        // SAFETY: the reserved result_len covers the kernel's src_len writes
        // plus the optional carry write at result_len - 1; the word_shift low
        // limbs are zeroed first because the kernel only writes the shifted
        // limbs at offset word_shift and the result's low limbs are zero by
        // construction. Length is committed only after all writes complete.
        let kernel_carry = unsafe {
            write_bytes(dst, 0, word_shift);
            if src_len <= INLINE_LIMBS {
                ArchKernels::lshift_into_small_unchecked(
                    dst.add(word_shift),
                    src.as_ptr(),
                    src_len,
                    bit_shift as u32,
                )
            } else {
                ArchKernels::lshift_into_unchecked(
                    dst.add(word_shift),
                    src.as_ptr(),
                    src_len,
                    bit_shift as u32,
                )
            }
        };
        debug_assert_eq!(
            kernel_carry, carry,
            "kernel carry must equal the scalar-computed top-limb carry"
        );
        if carry != 0 {
            // SAFETY: `carry != 0` implies `result_len >= src_len >= 1`, so
            // `result_len - 1` is a valid destination slot.
            unsafe {
                *dst.add(result_len.unchecked_sub(1)) = carry;
            }
        }
        // SAFETY: the zeroing, shift kernel, and optional carry write initialized
        // all `result_len` slots in the exact-capacity allocation.
        unsafe {
            limbs.set_len(result_len);
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
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Shift remainders are below LIMB_BITS <= 64 and inline lengths are at most four, fitting u32 and u8 on every supported target"
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

        // SAFETY: the preceding guard proves word_shift < src_len.
        let result_len = unsafe { src_len.unchecked_sub(word_shift) };

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
            // suffix is initialized; the fresh aligned allocation reserves
            // result_len disjoint slots. Copying fills its committed prefix.
            unsafe {
                copy_nonoverlapping(src.as_ptr().add(word_shift), limbs.as_mut_ptr(), result_len);
                limbs.set_len(result_len);
            }
            // SAFETY: src is normalized, so the copied suffix has a non-zero
            // top limb (src[src_len - 1]).
            return unsafe { Self::from_limbs_normalized(limbs) };
        }

        // SAFETY: `src_len >= 1` (normalized, non-zero), so `src_len - 1`
        // cannot underflow and indexes the top limb, which decides whether
        // the result shrinks by one limb.
        let top_is_shifted_out =
            unsafe { *src.get_unchecked(src_len.unchecked_sub(1)) >> bit_shift == 0 };
        // SAFETY: result_len > 0 and at most the highest limb is removed.
        let exact_len = unsafe { result_len.unchecked_sub(usize::from(top_is_shifted_out)) };
        if exact_len == 0 {
            return Self::zero();
        }

        if result_len <= INLINE_LIMBS {
            let mut out = [0; INLINE_LIMBS];
            // SAFETY: the kernel writes result_len limbs at out[0..];
            // result_len <= INLINE_LIMBS, and the spans are disjoint.
            unsafe {
                let _ = ArchKernels::rshift_into_small_unchecked(
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
        let dst = limbs.spare_capacity_mut().as_mut_ptr().cast::<Limb>();
        // SAFETY: the reserved result_len covers every limb the kernel writes
        // before the length is committed. Its returned low-bit residue is
        // discarded by logical right shift; all retained bits are written here.
        let _kernel_carry = unsafe {
            ArchKernels::rshift_into_unchecked(
                dst,
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
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Shift remainders are below LIMB_BITS <= 64, fitting u32 on every supported target"
    )]
    pub fn shl_assign(&mut self, shift: usize) {
        if shift == 0 || self.is_zero() {
            return;
        }
        let word_shift = shift.wrapping_div(LIMB_BITS);

        let bit_shift: u32 = shift.wrapping_rem(LIMB_BITS) as u32;

        if word_shift > 0 {
            let src_len = self.limbs().len();
            // SAFETY: src_len <= isize::MAX and word_shift <= usize::MAX / 16
            // bound the exact checked sum below usize::MAX on every supported
            // pointer width; the reservation size therefore has no failure path.
            let new_len = unsafe { src_len.checked_add(word_shift).unwrap_unchecked() };
            let mut pending = self.prepare_limb_write(new_len);
            let ptr = pending.as_mut_ptr();
            // SAFETY: preparation reserves new_len = src_len + word_shift;
            // the old initialized prefix moves within that allocation. copy
            // permits the overlap, and the low padding is then initialized.
            unsafe {
                copy(ptr, ptr.add(word_shift), src_len);
                write_bytes(ptr, 0, word_shift);
            }
            // SAFETY: the old prefix was initialized, and the memmove plus
            // zeroing initialized the complete expanded span.
            let _ = unsafe { pending.commit() };
        }

        if bit_shift > 0 {
            let len = self.limbs().len();
            let limbs_ptr = self.limbs_mut().as_mut_ptr();
            // SAFETY: `bit_shift` is in (0, LIMB_BITS) by construction. `limbs_ptr` points to
            // `len` valid Limb elements.
            let carry = unsafe { ArchKernels::lshift_unchecked(limbs_ptr, len, bit_shift) };
            if carry != 0 {
                self.push_limb(carry);
            }
        }
        // A nonzero source retains a nonzero top limb or produces a nonzero
        // carry. Whole-limb padding cannot change this normalization invariant.
    }

    /// Right-shifts the integer by `shift` bits in-place.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Shift remainders are below LIMB_BITS <= 64, fitting u32 on every supported target"
    )]
    pub fn shr_assign(&mut self, shift: usize) {
        if shift == 0 || self.is_zero() {
            return;
        }
        let word_shift = shift.wrapping_div(LIMB_BITS);

        let bit_shift: u32 = shift.wrapping_rem(LIMB_BITS) as u32;

        let src_len = self.limbs().len();
        if word_shift >= src_len {
            self.clear();
            return;
        }

        if word_shift > 0 {
            // SAFETY: the entry guard proves word_shift < src_len.
            let new_len = unsafe { src_len.unchecked_sub(word_shift) };
            let ptr = self.limbs_mut().as_mut_ptr();
            // SAFETY: word_shift < src_len and new_len = src_len - word_shift
            // bound both initialized spans. The exclusive owner permits overlap
            // through copy; every committed destination limb is initialized.
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
            self.normalize();
        }
    }
}
