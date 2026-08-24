//! Fused addition and subtraction into caller-owned storage.

use core::{
    cmp::{max, min},
    ptr::copy_nonoverlapping,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use alloc::vec::Vec;

use super::{Addition, ArchKernels, INLINE_LIMBS, InternalMpUint, UintRepr};

impl InternalMpUint {
    /// Computes `self = a + b`, overwriting the current value.
    #[allow(
        clippy::too_many_lines,
        reason = "The inline, heap, and storage-transition paths share one fused arithmetic operation"
    )]
    #[allow(
        unsafe_code,
        reason = "Pointer spans use short_len = min(a_len, b_len) or max_len - short_len; destinations are INLINE_LIMBS arrays or buffers sized to max_len, with a carry written only to an existing spare slot or storage transition."
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
            // SAFETY: all pointers cover `short_len <= max_len <= INLINE_LIMBS`.
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
                let rem = long_len.wrapping_sub(short_len);
                // SAFETY: both tails cover `rem` elements.
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
                    // SAFETY: the next inline slot exists and the length fits u8.
                    unsafe {
                        *dst_ptr.add(max_len) = carry;
                        *len = u8::try_from(max_len.wrapping_add(1)).unwrap_unchecked();
                    }
                } else {
                    // SAFETY: all inline slots are initialized, and `carry` is the
                    // normalized next limb.
                    unsafe {
                        *len = u8::try_from(max_len).unwrap_unchecked();
                        Addition::append_carry(dst, carry);
                    }
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
            let required_capacity = max_len.wrapping_add(1);
            if limbs.capacity() < required_capacity {
                limbs.reserve(required_capacity.wrapping_sub(limbs.len()));
            }
            #[allow(
                clippy::uninit_vec,
                reason = "add_limbs_3_unchecked plus fused tail copy fill all max_len slots"
            )]
            // SAFETY: capacity covers `required_capacity >= max_len + 1`; all slots are initialized below.
            unsafe {
                limbs.set_len(max_len);
            }
            let dst_ptr = limbs.as_mut_ptr();
            // SAFETY: all pointers cover `short_len`.
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
                let rem = long_len.wrapping_sub(short_len);
                // SAFETY: both tails cover `rem` elements.
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
                // SAFETY: capacity covers `max_len + 1` as ensured above.
                unsafe {
                    *dst_ptr.add(max_len) = carry;
                    limbs.set_len(max_len.wrapping_add(1));
                }
            }
            return;
        }

        let required_capacity = max_len.wrapping_add(1);
        // SAFETY: storage is grown to `required_capacity`, and all `max_len` slots
        // are overwritten below.
        let dst_limbs = unsafe { dst.ensure_capacity_set_len_get_limbs(required_capacity) };
        let dst_ptr = dst_limbs.as_mut_ptr();
        // SAFETY: all pointers cover `short_len`.
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
            let rem = long_len.wrapping_sub(short_len);
            // SAFETY: both tails cover `rem` elements.
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
            // SAFETY: capacity is at least `max_len + 1`.
            unsafe {
                *dst_ptr.add(max_len) = carry;
                dst.set_len(max_len.wrapping_add(1));
            }
        } else {
            // SAFETY: `max_len` is the exact sum length when carry is zero.
            unsafe {
                dst.set_len(max_len);
            }
        }
    }

    /// Computes `dst = a - b`, overwriting `dst`.
    ///
    /// Returns `true` when the operation underflows.
    #[allow(
        clippy::too_many_lines,
        reason = "The inline, heap, and storage-transition paths share one fused arithmetic operation"
    )]
    #[allow(
        unsafe_code,
        reason = "Pointer spans use short_len = min(a_len, b_len) or the remaining max_len - short_len limbs; destinations are INLINE_LIMBS arrays or buffers sized to max_len before subtraction and normalization."
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
        if a_len == 0 {
            let mut limbs = Vec::with_capacity(b_len);
            #[allow(
                clippy::uninit_vec,
                reason = "negate_with_borrow writes all b_len elements"
            )]
            // SAFETY: the following kernel initializes every slot.
            unsafe {
                limbs.set_len(b_len);
            }
            // SAFETY: both slices cover `b_len` elements.
            let borrow = unsafe {
                Addition::negate_with_borrow(
                    from_raw_parts_mut(limbs.as_mut_ptr(), b_len),
                    from_raw_parts(b_limbs.as_ptr(), b_len),
                    0,
                )
            };
            *dst = Self::from_limbs(limbs);
            return borrow != 0;
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
            // SAFETY: all pointers cover `short_len <= INLINE_LIMBS`.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    dst_ptr,
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                    short_len,
                )
            };
            if a_len > b_len {
                let rem = a_len.wrapping_sub(b_len);
                // SAFETY: both tails cover `rem` elements.
                unsafe {
                    borrow = Addition::copy_tail_with_borrow(
                        dst_ptr.add(b_len),
                        a_limbs.as_ptr().add(b_len),
                        rem,
                        borrow,
                    );
                }
            } else if b_len > a_len {
                let rem = b_len.wrapping_sub(a_len);
                // SAFETY: both tails cover `rem` elements.
                unsafe {
                    borrow = Addition::negate_with_borrow(
                        from_raw_parts_mut(dst_ptr.add(a_len), rem),
                        from_raw_parts(b_limbs.as_ptr().add(a_len), rem),
                        borrow,
                    );
                }
            }
            let mut final_len = max_len;
            while final_len > 0 {
                // SAFETY: `0 < final_len <= max_len <= INLINE_LIMBS`.
                if unsafe { *dst_ptr.add(final_len.wrapping_sub(1)) != 0 } {
                    break;
                }
                final_len = final_len.wrapping_sub(1);
            }
            // SAFETY: `final_len <= INLINE_LIMBS <= u8::MAX`.
            unsafe {
                *len = u8::try_from(final_len).unwrap_unchecked();
            }
            return borrow != 0;
        }

        if let UintRepr::Heap(ref mut limbs) = dst.repr {
            if limbs.capacity() < max_len {
                limbs.reserve(max_len.wrapping_sub(limbs.len()));
            }
            #[allow(
                clippy::uninit_vec,
                reason = "sub_limbs_3_unchecked plus fused tail handling fill all max_len slots"
            )]
            // SAFETY: capacity covers `max_len`; all slots are initialized below.
            unsafe {
                limbs.set_len(max_len);
            }
            let dst_ptr = limbs.as_mut_ptr();
            // SAFETY: all pointers cover `short_len`.
            let mut borrow = unsafe {
                ArchKernels::sub_limbs_3_unchecked(
                    dst_ptr,
                    a_limbs.as_ptr(),
                    b_limbs.as_ptr(),
                    short_len,
                )
            };
            if a_len > b_len {
                let rem = a_len.wrapping_sub(b_len);
                // SAFETY: both tails cover `rem` elements.
                unsafe {
                    borrow = Addition::copy_tail_with_borrow(
                        dst_ptr.add(b_len),
                        a_limbs.as_ptr().add(b_len),
                        rem,
                        borrow,
                    );
                }
            } else if b_len > a_len {
                let rem = b_len.wrapping_sub(a_len);
                // SAFETY: both tails cover `rem` elements.
                unsafe {
                    borrow = Addition::negate_with_borrow(
                        from_raw_parts_mut(dst_ptr.add(a_len), rem),
                        from_raw_parts(b_limbs.as_ptr().add(a_len), rem),
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

        // SAFETY: storage is grown to `max_len`, and every returned slot is
        // overwritten below.
        let dst_limbs = unsafe { dst.ensure_capacity_set_len_get_limbs(max_len) };
        // SAFETY: all pointers cover `short_len`.
        let mut borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                dst_limbs.as_mut_ptr(),
                a_limbs.as_ptr(),
                b_limbs.as_ptr(),
                short_len,
            )
        };
        if a_len > b_len {
            let rem_len = a_len.wrapping_sub(b_len);
            // SAFETY: both tails cover `rem_len` elements.
            unsafe {
                copy_nonoverlapping(
                    a_limbs.as_ptr().add(b_len),
                    dst_limbs.as_mut_ptr().add(b_len),
                    rem_len,
                );
            }
            if borrow != 0 {
                // SAFETY: this branch has `a_len > b_len`, `max_len = a_len`,
                // and `dst_limbs.len() = max_len`, so `b_len..` is in bounds.
                let dst_tail = unsafe { dst_limbs.get_unchecked_mut(b_len..) };
                borrow = Addition::propagate_borrow(dst_tail, borrow);
            }
        } else if b_len > a_len {
            // SAFETY: this branch has `b_len > a_len`, `max_len = b_len`,
            // and `dst_limbs.len() = max_len`, so `a_len..` is in bounds.
            let dst_tail = unsafe { dst_limbs.get_unchecked_mut(a_len..) };
            // SAFETY: `a_len < b_len = b_limbs.len()`, so `a_len..` is within
            // the initialized immutable source slice and does not alias `dst`.
            let b_tail = unsafe { b_limbs.get_unchecked(a_len..) };
            borrow = Addition::negate_with_borrow(dst_tail, b_tail, borrow);
        }
        dst.normalize();
        borrow != 0
    }
}
