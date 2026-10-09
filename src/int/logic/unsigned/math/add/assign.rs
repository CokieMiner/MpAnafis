//! In-place addition and subtraction.

#![expect(
    unsafe_code,
    reason = "Representation bounds and reserved capacity justify raw in-place arithmetic and infallible length calculations."
)]

use core::{
    cmp::{max, min},
    slice::from_raw_parts_mut,
};

use super::{Addition, ArchKernels, INLINE_LIMBS, InternalMpUint, UintRepr};

impl InternalMpUint {
    /// Adds `src` directly into `self`.
    #[expect(
        clippy::too_many_lines,
        reason = "Carry-capacity proofs, heap reuse, and inline transitions share one in-place arithmetic boundary."
    )]
    #[expect(
        clippy::inline_always,
        reason = "Inlining this arithmetic loop eliminates call overhead and exposes loop invariants to optimizer branch pruning."
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
            let capacity = limbs.capacity();
            let needs_growth = if capacity == max_len {
                // The lower columns contribute at most one carry. A high
                // column sum below B-1 proves no additional limb is possible.
                // SAFETY: both operands are nonempty; each last-limb read is
                // within its initialized, immutable span on every target.
                let (left, right) = unsafe {
                    (
                        if old_dst_len == max_len {
                            *limbs.last().unwrap_unchecked()
                        } else {
                            0
                        },
                        if src_len == max_len {
                            *src_limbs.last().unwrap_unchecked()
                        } else {
                            0
                        },
                    )
                };
                // left + right >= B-1 iff left >= (B-1)-right.
                // The complement expresses that exact bound in one compare.
                left >= !right
            } else {
                capacity < max_len
            };
            if needs_growth {
                // Reserve before mutation when growth is required or a carry
                // is possible, avoiding allocation on the carry dependency.
                // SAFETY: max_len is the length of an existing Limb slice.
                // Its byte span is at most isize::MAX, so one additional
                // element fits usize on 16-, 32-, and 64-bit targets.
                let required_capacity = unsafe { max_len.unchecked_add(1) };
                // SAFETY: capacity < required_capacity and len <= capacity
                // prove the subtraction is defined on every pointer width.
                let additional = unsafe { required_capacity.unchecked_sub(limbs.len()) };
                limbs.reserve(additional);
            }

            let dst_ptr = limbs.as_mut_ptr();
            let short_len = min(old_dst_len, src_len);
            // SAFETY: short_len is the smaller initialized operand length.
            // Vec storage and the source slice are aligned and disjoint under
            // the exclusive receiver borrow; the kernel reads and updates only
            // the old initialized destination prefix.
            let mut carry =
                unsafe { ArchKernels::add_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) };

            if src_len > old_dst_len {
                // SAFETY: this branch proves old_dst_len < src_len.
                let rem = unsafe { src_len.unchecked_sub(old_dst_len) };
                // SAFETY: the reserved destination has writable, aligned space
                // through max_len. The source tail contains rem initialized
                // limbs and cannot alias dst. This kernel initializes the
                // entire new tail without reading its previous contents.
                unsafe {
                    carry = Addition::copy_tail_with_carry(
                        dst_ptr.add(old_dst_len),
                        src_limbs.as_ptr().add(old_dst_len),
                        rem,
                        carry,
                    );
                }
            } else if carry != 0 && old_dst_len > src_len {
                // SAFETY: this branch proves src_len < old_dst_len.
                let rem = unsafe { old_dst_len.unchecked_sub(src_len) };
                // SAFETY: src_len + rem == old_dst_len bounds this aligned,
                // initialized, exclusive destination tail; carry is binary.
                unsafe {
                    carry = Addition::propagate_carry(
                        from_raw_parts_mut(dst_ptr.add(src_len), rem),
                        carry,
                    );
                }
            }

            let result_len = if carry != 0 {
                // SAFETY: a carry is impossible when the high-column bound
                // leaves capacity == max_len. Every other path reserves or
                // already owns at least max_len+1 slots. The aligned carry
                // write initializes that final slot before committing length.
                unsafe {
                    *dst_ptr.add(max_len) = carry;
                    max_len.unchecked_add(1)
                }
            } else {
                max_len
            };
            // SAFETY: the old prefix and growing tail are initialized; the
            // optional carry slot is written above. Capacity covers result_len.
            unsafe {
                limbs.set_len(result_len);
            }
            return;
        }

        let old_dst_len = dst.limbs().len();
        if old_dst_len == 0 {
            dst.clone_from(src);
            return;
        }

        if src_len > INLINE_LIMBS {
            // The destination is inline and the sum must be heap-backed.
            // Construct it directly in one allocation with a carry slot;
            // preserving the inline prefix in a growing buffer would copy
            // those limbs before overwriting them and may grow twice.
            *dst = dst.add(src);
            return;
        }

        let max_len = max(old_dst_len, src_len);
        let mut write = dst.prepare_limb_write(max_len);
        let dst_ptr = write.as_mut_ptr();
        let short_len = min(old_dst_len, src_len);
        // SAFETY: short_len is the smaller initialized operand length. The
        // prepared inline storage is aligned, exclusively borrowed, and
        // disjoint from the source; no newly exposed tail is read here.
        let mut carry =
            unsafe { ArchKernels::add_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) };

        if src_len > old_dst_len {
            // SAFETY: this branch proves old_dst_len < src_len.
            let rem = unsafe { src_len.unchecked_sub(old_dst_len) };
            // SAFETY: old_dst_len + rem == src_len <= max_len bounds the
            // initialized source tail and prepared, aligned output tail. The
            // exclusive receiver borrow makes them disjoint; every slot is written.
            unsafe {
                carry = Addition::copy_tail_with_carry(
                    dst_ptr.add(old_dst_len),
                    src_limbs.as_ptr().add(old_dst_len),
                    rem,
                    carry,
                );
            }
        } else if carry != 0 && old_dst_len > src_len {
            // SAFETY: this branch proves src_len < old_dst_len.
            let rem = unsafe { old_dst_len.unchecked_sub(src_len) };
            // SAFETY: src_len + rem == old_dst_len bounds this aligned,
            // initialized, exclusive destination tail; carry is binary.
            unsafe {
                carry =
                    Addition::propagate_carry(from_raw_parts_mut(dst_ptr.add(src_len), rem), carry);
            }
        }

        // SAFETY: the shared prefix was initialized before preparation; the
        // growing tail, if any, is completely written by copy_tail_with_carry.
        // The prepared capacity covers all max_len initialized limbs.
        let _initialized = unsafe { write.commit() };
        if carry != 0 {
            if max_len < INLINE_LIMBS {
                // SAFETY: `max_len < INLINE_LIMBS`; `dst` remains inline and the
                // next slot and max_len + 1 <= 4 <= u8::MAX are representable.
                unsafe {
                    if let UintRepr::Inline {
                        ref mut len,
                        ref mut limbs,
                    } = dst.repr
                    {
                        *limbs.as_mut_ptr().add(max_len) = carry;
                        *len = u8::try_from(max_len.unchecked_add(1)).unwrap_unchecked();
                    }
                }
            } else {
                dst.push_limb(carry);
            }
        }
    }

    /// Subtracts `src` directly from `self`.
    ///
    /// `self` must be greater than or equal to `src`. Normalized operand order
    /// proves that the source has no more initialized limbs than the destination,
    /// so this path never grows storage or
    /// initializes a negative residue tail. The general underflow-reporting
    /// entry handles those states separately.
    #[expect(
        clippy::inline_always,
        reason = "The invariant boundary disappears in release builds while preserving the in-place subtraction hot path."
    )]
    #[inline(always)]
    pub fn sub_assign(&mut self, src: &Self) {
        let source = src.limbs();
        if source.is_empty() {
            return;
        }
        let destination = self.limbs_mut();
        let mut borrow = Addition::sub_slice_in_place(destination, source);
        if borrow != 0 {
            // SAFETY: normalized self >= src proves source.len() <= destination.len().
            // The initialized, aligned tail belongs to the exclusive receiver;
            // only the source-length prefix was changed by the first kernel.
            let tail = unsafe { destination.get_unchecked_mut(source.len()..) };
            borrow = Addition::propagate_borrow(tail, borrow);
        }
        debug_assert_eq!(borrow, 0, "the subtraction precondition prevents underflow");
        self.normalize();
    }

    /// Subtracts `src` from `self` as a fixed-width residue.
    ///
    /// Returns `true` when the residue represents a negative mathematical
    /// result. Signed magnitude arithmetic consumes that state directly.
    #[expect(
        clippy::inline_always,
        reason = "Inlining this arithmetic loop eliminates call overhead and exposes loop invariants to optimizer branch pruning."
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
                // SAFETY: len <= capacity < max_len proves this difference
                // is positive on every pointer width. Vec checks byte capacity.
                let additional = unsafe { max_len.unchecked_sub(old_dst_len) };
                limbs.reserve(additional);
            }

            let dst_ptr = limbs.as_mut_ptr();
            let short_len = min(old_dst_len, src_len);
            let mut borrow = if short_len == 0 {
                0
            } else {
                // SAFETY: short_len bounds both initialized operand prefixes;
                // Vec and slice storage are aligned and disjoint under the
                // exclusive receiver borrow. Newly exposed slots are not read.
                unsafe { ArchKernels::sub_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) }
            };

            if src_len > old_dst_len {
                // SAFETY: this branch proves old_dst_len < src_len.
                let rem = unsafe { src_len.unchecked_sub(old_dst_len) };
                // SAFETY: reservation covers max_len = src_len. The aligned
                // writable destination tail is disjoint from the initialized
                // source tail; each covers rem limbs. Borrow is zero or one.
                unsafe {
                    borrow = Addition::negate_with_borrow(
                        dst_ptr.add(old_dst_len),
                        src_limbs.as_ptr().add(old_dst_len),
                        rem,
                        borrow,
                    );
                }
            } else if borrow != 0 && old_dst_len > src_len {
                // SAFETY: this branch proves src_len < old_dst_len.
                let rem = unsafe { old_dst_len.unchecked_sub(src_len) };
                // SAFETY: src_len + rem == old_dst_len bounds this aligned,
                // initialized, exclusive destination tail; borrow is binary.
                unsafe {
                    borrow = Addition::propagate_borrow(
                        from_raw_parts_mut(dst_ptr.add(src_len), rem),
                        borrow,
                    );
                }
            }

            // Determine the normalized length before committing the output;
            // the raw kernels initialized max_len slots, including any tail.
            let mut final_len = max_len;
            while final_len > 0 {
                // SAFETY: 0 < final_len <= max_len, so the last index exists.
                let last = unsafe { final_len.unchecked_sub(1) };
                // SAFETY: last < max_len <= capacity; the old prefix and
                // write-only growing tail are initialized, aligned, and owned.
                if unsafe { *dst_ptr.add(last) != 0 } {
                    break;
                }
                final_len = last;
            }
            // SAFETY: final_len <= max_len <= capacity retains only initialized
            // limbs. Limb = usize has no Drop; discarded high limbs are zero.
            unsafe {
                limbs.set_len(final_len);
            }
            return borrow != 0;
        }

        let old_dst_len = dst.limbs().len();
        let max_len = max(old_dst_len, src_len);
        let mut write = dst.prepare_limb_write(max_len);
        let short_len = min(old_dst_len, src_len);
        let dst_ptr = write.as_mut_ptr();
        let mut borrow = if short_len == 0 {
            0
        } else {
            // SAFETY: short_len bounds both initialized operand prefixes;
            // prepared storage and the source slice are aligned and disjoint
            // under the exclusive receiver borrow. New tail slots are not read.
            unsafe { ArchKernels::sub_limbs_unchecked(dst_ptr, src_limbs.as_ptr(), short_len) }
        };

        if src_len > old_dst_len {
            // SAFETY: this branch proves old_dst_len < src_len.
            let rem = unsafe { src_len.unchecked_sub(old_dst_len) };
            // SAFETY: preparation covers max_len = src_len. The aligned
            // writable destination tail is disjoint from the initialized
            // source tail; each covers rem limbs. Borrow is zero or one.
            unsafe {
                borrow = Addition::negate_with_borrow(
                    dst_ptr.add(old_dst_len),
                    src_limbs.as_ptr().add(old_dst_len),
                    rem,
                    borrow,
                );
            }
        } else if borrow != 0 && old_dst_len > src_len {
            // SAFETY: this branch proves src_len < old_dst_len.
            let rem = unsafe { old_dst_len.unchecked_sub(src_len) };
            // SAFETY: src_len + rem == old_dst_len bounds this aligned,
            // initialized, exclusive destination tail; borrow is binary.
            unsafe {
                borrow = Addition::propagate_borrow(
                    from_raw_parts_mut(dst_ptr.add(src_len), rem),
                    borrow,
                );
            }
        }

        // SAFETY: the shared prefix is initialized before preparation, and
        // the write-only negation kernel initializes any newly exposed tail.
        // Preparation guarantees capacity for all max_len limbs.
        let _initialized = unsafe { write.commit() };
        dst.normalize();
        borrow != 0
    }
}
