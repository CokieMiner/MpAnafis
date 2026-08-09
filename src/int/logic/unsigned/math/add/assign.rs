//! In-place addition and subtraction.

use core::{
    cmp::{max, min},
    slice::{from_raw_parts, from_raw_parts_mut},
};

use super::{Addition, ArchKernels, INLINE_LIMBS, InternalMpUint, UintRepr};

impl InternalMpUint {
    /// Adds `src` directly into `self`.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this arithmetic loop eliminates call overhead and exposes loop invariants to optimizer branch pruning."
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "The two paths (Heap / Inline) share the same logic but target different storage"
    )]
    #[allow(
        unsafe_code,
        reason = "Bypasses InternalMpUint enum dispatch by extracting the inner Vec<Limb> and working with raw pointers."
    )]
    #[allow(
        clippy::many_single_char_names,
        reason = "len/carry/c are conventional names"
    )]
    #[inline(always)]
    pub fn add_assign(&mut self, src: &Self) {
        let dst = self;
        let src_limbs = src.limbs();
        let src_len = src_limbs.len();
        if src_len == 0 {
            return;
        }

        if let UintRepr::Heap(ref mut limbs) = dst.repr {
            let old_dst_len = limbs.len();
            if old_dst_len == 0 {
                limbs.extend_from_slice(src_limbs);
                return;
            }

            let max_len = max(old_dst_len, src_len);
            if limbs.capacity() < max_len {
                limbs.reserve(max_len.wrapping_sub(limbs.len()));
            }
            #[allow(
                clippy::uninit_vec,
                reason = "add_limbs_unchecked plus fused copy and propagation fill all max_len slots"
            )]
            // SAFETY: capacity is at least `max_len`; all extended slots are
            // initialized below before any read.
            unsafe {
                limbs.set_len(max_len);
            }

            let dst_ptr = limbs.as_mut_ptr();
            let short_len = min(old_dst_len, src_len);
            // SAFETY: both pointers cover `short_len` limbs.
            let mut carry =
                unsafe { ArchKernels::add_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) };

            if src_len > old_dst_len {
                let rem = src_len.wrapping_sub(old_dst_len);
                // SAFETY: both tails cover `rem` limbs.
                unsafe {
                    carry = Addition::copy_tail_with_carry(
                        dst_ptr.add(old_dst_len),
                        src_limbs.as_ptr().add(old_dst_len),
                        rem,
                        carry,
                    );
                }
            } else if carry != 0 && old_dst_len > src_len {
                let rem = old_dst_len.wrapping_sub(src_len);
                // SAFETY: the destination tail covers `rem` limbs.
                unsafe {
                    carry = Addition::propagate_carry(
                        from_raw_parts_mut(dst_ptr.add(src_len), rem),
                        carry,
                    );
                }
            }

            if carry != 0 {
                limbs.push(carry);
            }
            return;
        }

        let old_dst_len = dst.limbs().len();
        if old_dst_len == 0 {
            dst.clone_from(src);
            return;
        }

        let max_len = max(old_dst_len, src_len);
        // SAFETY: the method grows storage to `max_len` and returns that many
        // writable limbs.
        let dst_limbs = unsafe { dst.ensure_capacity_set_len_get_limbs(max_len) };
        let dst_ptr = dst_limbs.as_mut_ptr();
        let short_len = min(old_dst_len, src_len);
        // SAFETY: both pointers cover `short_len` limbs.
        let mut carry =
            unsafe { ArchKernels::add_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) };

        if src_len > old_dst_len {
            let rem = src_len.wrapping_sub(old_dst_len);
            // SAFETY: both tails cover `rem` limbs.
            unsafe {
                carry = Addition::copy_tail_with_carry(
                    dst_ptr.add(old_dst_len),
                    src_limbs.as_ptr().add(old_dst_len),
                    rem,
                    carry,
                );
            }
        } else if carry != 0 && old_dst_len > src_len {
            let rem = old_dst_len.wrapping_sub(src_len);
            // SAFETY: the destination tail covers `rem` limbs.
            unsafe {
                carry =
                    Addition::propagate_carry(from_raw_parts_mut(dst_ptr.add(src_len), rem), carry);
            }
        }

        if carry != 0 {
            if max_len < INLINE_LIMBS {
                // SAFETY: `max_len < INLINE_LIMBS`; `dst` remains inline and the
                // next slot and encoded length are representable.
                unsafe {
                    if let UintRepr::Inline {
                        ref mut len,
                        ref mut limbs,
                    } = dst.repr
                    {
                        *limbs.as_mut_ptr().add(max_len) = carry;
                        *len = u8::try_from(max_len.wrapping_add(1)).unwrap_unchecked();
                    }
                }
            } else {
                // SAFETY: `carry` is the non-zero normalized next limb.
                unsafe {
                    Addition::append_carry(dst, carry);
                }
            }
        }
    }

    /// Subtracts `src` directly from `self`.
    ///
    /// `self` must be greater than or equal to `src`.
    #[allow(
        clippy::inline_always,
        reason = "The invariant boundary disappears in release builds while preserving the in-place subtraction hot path."
    )]
    #[inline(always)]
    pub fn sub_assign(&mut self, src: &Self) {
        let underflowed = self.sub_assign_with_underflow(src);
        debug_assert!(
            !underflowed,
            "the subtraction precondition prevents underflow"
        );
    }

    /// Subtracts `src` from `self` as a fixed-width residue.
    ///
    /// Returns `true` when the residue represents a negative mathematical
    /// result. Signed magnitude arithmetic consumes that state directly.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this arithmetic loop eliminates call overhead and exposes loop invariants to optimizer branch pruning."
    )]
    #[allow(
        clippy::many_single_char_names,
        reason = "len variables have explicit names; c/i are traditional carry/loop names"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "The two paths (Heap / Inline) share the same logic but target different storage"
    )]
    #[allow(
        unsafe_code,
        reason = "Bypasses InternalMpUint enum dispatch by extracting the inner Vec<Limb> and working with raw pointers."
    )]
    #[inline(always)]
    pub fn sub_assign_with_underflow(&mut self, src: &Self) -> bool {
        let dst = self;
        let src_limbs = src.limbs();
        let src_len = src_limbs.len();
        if src_len == 0 {
            return false;
        }

        if let UintRepr::Heap(ref mut limbs) = dst.repr {
            let old_dst_len = limbs.len();
            let max_len = max(old_dst_len, src_len);
            if limbs.capacity() < max_len {
                limbs.reserve(max_len.wrapping_sub(limbs.len()));
            }
            #[allow(
                clippy::uninit_vec,
                reason = "sub_limbs_unchecked plus negate and borrow propagation fill all slots"
            )]
            // SAFETY: capacity is at least `max_len`; every slot is initialized
            // below before it is read.
            unsafe {
                limbs.set_len(max_len);
            }

            let dst_ptr = limbs.as_mut_ptr();
            let short_len = min(old_dst_len, src_len);
            let mut borrow = if short_len == 0 {
                0
            } else {
                // SAFETY: both pointers cover `short_len` limbs.
                unsafe { ArchKernels::sub_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) }
            };

            if src_len > old_dst_len {
                let rem = src_len.wrapping_sub(old_dst_len);
                // SAFETY: both tails cover `rem` limbs.
                unsafe {
                    borrow = Addition::negate_with_borrow(
                        from_raw_parts_mut(dst_ptr.add(old_dst_len), rem),
                        from_raw_parts(src_limbs.as_ptr().add(old_dst_len), rem),
                        borrow,
                    );
                }
            } else if borrow != 0 && old_dst_len > src_len {
                let rem = old_dst_len.wrapping_sub(src_len);
                // SAFETY: the destination tail covers `rem` limbs.
                unsafe {
                    borrow = Addition::propagate_borrow(
                        from_raw_parts_mut(dst_ptr.add(src_len), rem),
                        borrow,
                    );
                }
            }

            if let Some(last_nonzero) = limbs.iter().rposition(|&limb| limb != 0) {
                limbs.truncate(last_nonzero.wrapping_add(1));
            } else {
                limbs.clear();
            }
            return borrow != 0;
        }

        let old_dst_len = dst.limbs().len();
        let max_len = max(old_dst_len, src_len);
        // SAFETY: the method grows storage to `max_len` and returns that many
        // writable limbs.
        let dst_limbs = unsafe { dst.ensure_capacity_set_len_get_limbs(max_len) };
        let short_len = min(old_dst_len, src_len);
        let dst_ptr = dst_limbs.as_mut_ptr();
        let mut borrow = if short_len == 0 {
            0
        } else {
            // SAFETY: both pointers cover `short_len` limbs.
            unsafe { ArchKernels::sub_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) }
        };

        if src_len > old_dst_len {
            let rem = src_len.wrapping_sub(old_dst_len);
            // SAFETY: both tails cover `rem` limbs.
            unsafe {
                borrow = Addition::negate_with_borrow(
                    from_raw_parts_mut(dst_ptr.add(old_dst_len), rem),
                    from_raw_parts(src_limbs.as_ptr().add(old_dst_len), rem),
                    borrow,
                );
            }
        } else if borrow != 0 && old_dst_len > src_len {
            let rem = old_dst_len.wrapping_sub(src_len);
            // SAFETY: the destination tail covers `rem` limbs.
            unsafe {
                borrow = Addition::propagate_borrow(
                    from_raw_parts_mut(dst_ptr.add(src_len), rem),
                    borrow,
                );
            }
        }

        dst.normalize();
        borrow != 0
    }
}
