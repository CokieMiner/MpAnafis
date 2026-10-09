//! Fused addition and subtraction into caller-owned storage.

#![expect(
    unsafe_code,
    reason = "Initialized source spans and prepared destination capacity justify infallible lengths, raw output writes, and normalization."
)]

use core::cmp::{max, min};

use super::{Addition, ArchKernels, INLINE_LIMBS, InternalMpUint, UintRepr};

impl InternalMpUint {
    /// Computes `self = a + b`, overwriting the current value.
    #[expect(
        clippy::too_many_lines,
        reason = "The inline, heap, and storage-transition paths share one fused arithmetic operation"
    )]
    #[inline]
    pub fn assign_sum(&mut self, a: &Self, b: &Self) {
        let dst = self;
        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();

        if a_len == 0 {
            dst.clone_from(b);
            return;
        }
        if b_len == 0 {
            dst.clone_from(a);
            return;
        }

        let max_len = max(a_len, b_len);
        let short_len = min(a_len, b_len);

        if let UintRepr::Inline {
            ref mut len,
            ref mut limbs,
        } = dst.repr
            && max_len <= INLINE_LIMBS
        {
            let dst_ptr = limbs.as_mut_ptr();
            // SAFETY: both aligned sources contain short_len initialized limbs.
            // The exclusive inline destination is disjoint from both inputs
            // and covers max_len <= INLINE_LIMBS; its previous values are not read.
            let mut carry = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    dst_ptr,
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                    short_len,
                )
            };
            if a_len != b_len {
                let (long_limbs, long_len) = if a_len > b_len {
                    (a_limbs, a_len)
                } else {
                    (b_limbs, b_len)
                };
                // SAFETY: unequal operand lengths give short_len < long_len.
                let rem = unsafe { long_len.unchecked_sub(short_len) };
                // SAFETY: short_len + rem == long_len == max_len <= INLINE_LIMBS.
                // The aligned source tail is initialized and disjoint from the
                // exclusive output; the kernel initializes every new tail slot.
                unsafe {
                    carry = Addition::copy_tail_with_carry(
                        dst_ptr.add(short_len),
                        long_limbs.as_ptr().add(short_len),
                        rem,
                        carry,
                    );
                }
            }

            if carry != 0 {
                if max_len < INLINE_LIMBS {
                    // SAFETY: max_len < INLINE_LIMBS == 4 gives an available
                    // carry slot and max_len + 1 <= 4 <= u8::MAX on every target.
                    unsafe {
                        *dst_ptr.add(max_len) = carry;
                        *len = u8::try_from(max_len.unchecked_add(1)).unwrap_unchecked();
                    }
                } else {
                    // SAFETY: all inline slots are initialized and the encoded
                    // length fits in u8.
                    unsafe {
                        *len = u8::try_from(max_len).unwrap_unchecked();
                    }
                    dst.push_limb(carry);
                }
            } else {
                // SAFETY: `max_len <= INLINE_LIMBS <= u8::MAX`.
                unsafe {
                    *len = u8::try_from(max_len).unwrap_unchecked();
                }
            }
            return;
        }

        if let UintRepr::Heap(ref mut limbs) = dst.repr {
            let capacity = limbs.capacity();
            let needs_growth = if capacity == max_len {
                // Lower columns contribute at most one carry; a high-column
                // sum below B-1 proves the existing capacity is sufficient.
                // SAFETY: the zero operands return above; both last-limb
                // reads are within their initialized immutable source spans.
                let (left, right) = unsafe {
                    (
                        if a_len == max_len {
                            *a_limbs.last().unwrap_unchecked()
                        } else {
                            0
                        },
                        if b_len == max_len {
                            *b_limbs.last().unwrap_unchecked()
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
                // Possible carries reserve their space before arithmetic.
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
            // SAFETY: both sources contain short_len initialized limbs and
            // are disjoint from dst. The reserved destination is aligned and
            // writable for max_len limbs; the kernel writes every prefix
            // limb without reading the destination's previous contents.
            let mut carry = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    dst_ptr,
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                    short_len,
                )
            };
            if a_len != b_len {
                let (long_limbs, long_len) = if a_len > b_len {
                    (a_limbs, a_len)
                } else {
                    (b_limbs, b_len)
                };
                // SAFETY: unequal operand lengths give short_len < long_len.
                let rem = unsafe { long_len.unchecked_sub(short_len) };
                // SAFETY: short_len + rem == long_len == max_len <= capacity.
                // The aligned, initialized source tail is disjoint from the
                // exclusive output; the kernel initializes every tail slot.
                unsafe {
                    carry = Addition::copy_tail_with_carry(
                        dst_ptr.add(short_len),
                        long_limbs.as_ptr().add(short_len),
                        rem,
                        carry,
                    );
                }
            }
            let result_len = if carry != 0 {
                // SAFETY: a carry is impossible when the high-column bound
                // leaves capacity == max_len. Every other path reserves or
                // already owns max_len+1 slots, so this aligned carry write
                // initializes the last slot before committing its length.
                unsafe {
                    *dst_ptr.add(max_len) = carry;
                    max_len.unchecked_add(1)
                }
            } else {
                max_len
            };
            // SAFETY: the prefix and tail kernels initialize max_len limbs;
            // any carry slot is written above. Capacity covers result_len,
            // including when overwriting a longer previous value.
            unsafe {
                limbs.set_len(result_len);
            }
            return;
        }

        // Only an inline destination with a heap-sized result reaches here.
        // No allocation is reusable and none of the previous value is an
        // input: construct the sum directly without preserving that prefix.
        *dst = a.add(b);
    }

    /// Computes `dst = a - b`, overwriting `dst`.
    ///
    /// Returns `true` when the operation underflows.
    #[expect(
        clippy::too_many_lines,
        reason = "Native, inline, and heap subtraction share the same output-initialization and borrow contract."
    )]
    #[inline]
    pub fn assign_difference(&mut self, a: &Self, b: &Self) -> bool {
        let dst = self;
        let a_limbs = a.limbs();
        let b_limbs = b.limbs();
        let a_len = a_limbs.len();
        let b_len = b_limbs.len();

        if b_len == 0 {
            dst.clone_from(a);
            return false;
        }

        let max_len = max(a_len, b_len);
        let short_len = min(a_len, b_len);

        if let UintRepr::Inline {
            ref mut len,
            ref mut limbs,
        } = dst.repr
            && max_len <= INLINE_LIMBS
        {
            let dst_ptr = limbs.as_mut_ptr();
            if max_len == 1 {
                // One native subtraction determines both the residue and
                // borrow; its nonzero bit determines the normalized length.
                let left = a_limbs.first().copied().unwrap_or(0);
                // SAFETY: b is nonzero and b_len <= max_len == 1, so its
                // first initialized limb exists. The inline destination is
                // aligned, exclusive, and disjoint from both source slices.
                unsafe {
                    let (difference, borrow) = left.overflowing_sub(*b_limbs.get_unchecked(0));
                    *dst_ptr = difference;
                    *len = u8::from(difference != 0);
                    return borrow;
                }
            }
            // SAFETY: both aligned sources contain short_len initialized limbs;
            // the exclusive, disjoint inline output covers max_len <= INLINE_LIMBS.
            // The kernel accepts an empty shared prefix and writes without
            // reading any old destination value.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    dst_ptr,
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                    short_len,
                )
            };
            if a_len > b_len {
                // SAFETY: this branch proves b_len < a_len.
                let rem = unsafe { a_len.unchecked_sub(b_len) };
                // SAFETY: b_len + rem == a_len <= INLINE_LIMBS bounds both
                // aligned, disjoint tails; the source is initialized and the
                // kernel writes the output without reading its previous values.
                unsafe {
                    borrow = Addition::copy_tail_with_borrow(
                        dst_ptr.add(b_len),
                        a_limbs.as_ptr().add(b_len),
                        rem,
                        borrow,
                    );
                }
            } else if b_len > a_len {
                // SAFETY: this branch proves a_len < b_len.
                let rem = unsafe { b_len.unchecked_sub(a_len) };
                // SAFETY: b_len <= INLINE_LIMBS bounds the aligned destination
                // tail and initialized source tail. Their rem-limb spans are
                // disjoint, and the prefix kernel returns a zero-or-one borrow.
                unsafe {
                    borrow = Addition::negate_with_borrow(
                        dst_ptr.add(a_len),
                        b_limbs.as_ptr().add(a_len),
                        rem,
                        borrow,
                    );
                }
            }
            let mut final_len = max_len;
            while final_len > 0 {
                // SAFETY: 0 < final_len <= max_len <= INLINE_LIMBS; the prefix
                // and tail kernels initialized every slot below max_len.
                let last = unsafe { final_len.unchecked_sub(1) };
                // SAFETY: last < max_len bounds this initialized output read.
                if unsafe { *dst_ptr.add(last) != 0 } {
                    break;
                }
                final_len = last;
            }
            // SAFETY: `final_len <= INLINE_LIMBS <= u8::MAX`.
            unsafe {
                *len = u8::try_from(final_len).unwrap_unchecked();
            }
            return borrow != 0;
        }

        let mut write = dst.prepare_limb_write(max_len);
        let dst_ptr = write.as_mut_ptr();
        // SAFETY: both initialized sources cover short_len; the prepared,
        // aligned destination covers max_len >= short_len and is disjoint.
        // The kernel initializes the prefix without reading the destination.
        let mut borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                dst_ptr,
                a_limbs.as_ptr(),
                b_limbs.as_ptr(),
                short_len,
            )
        };
        if a_len > b_len {
            // SAFETY: this branch proves b_len < a_len.
            let rem_len = unsafe { a_len.unchecked_sub(b_len) };
            // SAFETY: b_len + rem_len == a_len == max_len bounds both aligned,
            // disjoint tails. The source is initialized; preparation reserves
            // every output slot, and the kernel initializes the complete tail.
            unsafe {
                borrow = Addition::copy_tail_with_borrow(
                    dst_ptr.add(b_len),
                    a_limbs.as_ptr().add(b_len),
                    rem_len,
                    borrow,
                );
            }
        } else if b_len > a_len {
            // SAFETY: a_len < b_len = max_len. The initialized immutable
            // source tail and aligned writable destination tail are disjoint
            // and cover b_len-a_len limbs. Borrow is zero or one.
            unsafe {
                borrow = Addition::negate_with_borrow(
                    dst_ptr.add(a_len),
                    b_limbs.as_ptr().add(a_len),
                    b_len.unchecked_sub(a_len),
                    borrow,
                );
            }
        }
        // SAFETY: the prefix and applicable tail kernel initialize every
        // limb through max_len. Preparation guarantees that capacity, and no
        // typed reference to an uninitialized destination is constructed.
        let _initialized = unsafe { write.commit() };
        dst.normalize();
        borrow != 0
    }
}
