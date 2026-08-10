//! Limb-slice addition and subtraction primitives.

use core::{cmp::min, ptr::copy_nonoverlapping};

use super::{ArchKernels, InternalArbiUint, Limb};

/// Namespace for shared addition and subtraction limb primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addition;

impl Addition {
    /// Adds `src` into `dst`, returning the carry out.
    ///
    /// `dst` must have at least `src.len()` elements.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this helper eliminates call overhead and exposes slice invariants to the optimizer."
    )]
    #[allow(
        unsafe_code,
        reason = "Bypassing bounds checks on slice indexing to ensure branchless execution and prevent runtime panic checks in loop bodies."
    )]
    #[inline(always)]
    pub fn add_slice_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        let src_len = src.len();
        if src_len == 0 {
            return 0;
        }
        if src_len == 1 {
            // SAFETY: `src_len == 1`, and the caller guarantees `dst.len() >= 1`.
            unsafe {
                let (sum, carry) = (*dst.get_unchecked(0)).overflowing_add(*src.get_unchecked(0));
                *dst.get_unchecked_mut(0) = sum;
                return Limb::from(carry);
            }
        }
        // SAFETY: the caller guarantees `dst` has at least `src_len` elements.
        unsafe { ArchKernels::add_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), src_len) }
    }

    /// Subtracts `src` from `dst`, returning the borrow out.
    ///
    /// `dst` must have at least `src.len()` elements.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this helper eliminates call overhead and exposes slice invariants to the optimizer."
    )]
    #[allow(
        unsafe_code,
        reason = "Bypassing bounds checks on slice indexing to ensure branchless execution and prevent runtime panic checks in loop bodies."
    )]
    #[inline(always)]
    pub fn sub_slice_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        let src_len = src.len();
        if src_len == 0 {
            return 0;
        }
        if src_len == 1 {
            // SAFETY: `src_len == 1`, and the caller guarantees `dst.len() >= 1`.
            unsafe {
                let (diff, borrow) = (*dst.get_unchecked(0)).overflowing_sub(*src.get_unchecked(0));
                *dst.get_unchecked_mut(0) = diff;
                return Limb::from(borrow);
            }
        }
        // SAFETY: the caller guarantees `dst` has at least `src_len` elements.
        unsafe { ArchKernels::sub_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), src_len) }
    }

    /// Copies `rem` limbs from `src` to `dst`, propagating an initial `carry`.
    ///
    /// Returns the remaining carry out.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must be valid for reading and writing `rem` limbs,
    /// respectively.
    #[allow(
        clippy::inline_always,
        reason = "Core carry-propagation primitive used across all addition paths"
    )]
    #[allow(
        unsafe_code,
        reason = "Raw pointer indexing for maximum loop performance"
    )]
    #[inline(always)]
    pub unsafe fn copy_tail_with_carry(
        dst: *mut Limb,
        src: *const Limb,
        rem: usize,
        mut carry: Limb,
    ) -> Limb {
        if carry != 0 {
            let mut i = 0_usize;
            while i < rem {
                // SAFETY: the caller guarantees both pointers are valid for `rem`
                // limbs, and the loop maintains `i < rem`.
                unsafe {
                    let (sum, overflowed) = (*src.add(i)).overflowing_add(carry);
                    *dst.add(i) = sum;
                    carry = Limb::from(overflowed);
                    i = i.wrapping_add(1);
                    if carry == 0 {
                        if i < rem {
                            copy_nonoverlapping(src.add(i), dst.add(i), rem.wrapping_sub(i));
                        }
                        break;
                    }
                }
            }
        } else {
            // SAFETY: the caller guarantees both pointers are valid for `rem`
            // elements.
            unsafe {
                copy_nonoverlapping(src, dst, rem);
            }
        }
        carry
    }

    /// Copies `rem` limbs from `src` to `dst`, propagating an initial `borrow`.
    ///
    /// Returns the remaining borrow out.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must be valid for reading and writing `rem` limbs,
    /// respectively.
    #[allow(
        clippy::inline_always,
        reason = "Core borrow-propagation primitive used across subtraction paths"
    )]
    #[allow(
        unsafe_code,
        reason = "Raw pointer indexing for maximum loop performance"
    )]
    #[inline(always)]
    pub unsafe fn copy_tail_with_borrow(
        dst: *mut Limb,
        src: *const Limb,
        rem: usize,
        mut borrow: Limb,
    ) -> Limb {
        if borrow != 0 {
            let mut i = 0_usize;
            while i < rem {
                // SAFETY: the caller guarantees both pointers are valid for `rem`
                // limbs, and the loop maintains `i < rem`.
                unsafe {
                    let (diff, underflowed) = (*src.add(i)).overflowing_sub(borrow);
                    *dst.add(i) = diff;
                    borrow = Limb::from(underflowed);
                    i = i.wrapping_add(1);
                    if borrow == 0 {
                        if i < rem {
                            copy_nonoverlapping(src.add(i), dst.add(i), rem.wrapping_sub(i));
                        }
                        break;
                    }
                }
            }
        } else {
            // SAFETY: the caller guarantees both pointers are valid for `rem`
            // elements.
            unsafe {
                copy_nonoverlapping(src, dst, rem);
            }
        }
        borrow
    }

    /// Propagates a carry through `limbs`, returning the carry out.
    #[allow(
        unsafe_code,
        reason = "Calling unchecked architecture kernel requires unsafe block"
    )]
    #[allow(clippy::inline_always, reason = "Critical for peak performance")]
    #[inline(always)]
    pub fn propagate_carry(limbs: &mut [Limb], carry: Limb) -> Limb {
        if carry == 0 || limbs.is_empty() {
            return carry;
        }
        // SAFETY: `limbs` is non-empty, `carry != 0`, and the slice provides a
        // valid pointer for its full length.
        unsafe { ArchKernels::propagate_carry_unchecked(limbs.as_mut_ptr(), limbs.len(), carry) }
    }

    /// Propagates a borrow through `limbs`, returning the borrow out.
    #[allow(
        unsafe_code,
        reason = "Calling unchecked architecture kernel requires unsafe block"
    )]
    #[allow(clippy::inline_always, reason = "Critical for peak performance")]
    #[inline(always)]
    pub fn propagate_borrow(limbs: &mut [Limb], borrow: Limb) -> Limb {
        if borrow == 0 || limbs.is_empty() {
            return borrow;
        }
        // SAFETY: `limbs` is non-empty, `borrow != 0`, and the slice provides a
        // valid pointer for its full length.
        unsafe { ArchKernels::propagate_borrow_unchecked(limbs.as_mut_ptr(), limbs.len(), borrow) }
    }

    /// Subtracts `src` and `borrow` from zero into `dst`.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this arithmetic loop eliminates call overhead and exposes loop invariants to optimizer branch pruning."
    )]
    #[allow(
        unsafe_code,
        reason = "Raw pointer iteration eliminates bounds checks and iterator overhead"
    )]
    #[inline(always)]
    pub fn negate_with_borrow(dst: &mut [Limb], src: &[Limb], mut borrow: Limb) -> Limb {
        let len = min(dst.len(), src.len());
        let mut i = 0_usize;
        let dst_ptr = dst.as_mut_ptr();
        let src_ptr = src.as_ptr();
        while i < len {
            // SAFETY: `i < len <= min(dst.len(), src.len())`.
            unsafe {
                let (diff1, underflowed1) = (0_usize).overflowing_sub(*src_ptr.add(i));
                let (diff, underflowed2) = diff1.overflowing_sub(borrow);
                let underflowed = underflowed1 || underflowed2;
                *dst_ptr.add(i) = diff;
                borrow = Limb::from(underflowed);
            }
            i = i.wrapping_add(1);
        }
        borrow
    }

    /// Appends a non-zero carry limb to `dst`.
    ///
    /// # Safety
    ///
    /// The caller must have proved that `carry` is the next normalized limb.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this arithmetic loop eliminates call overhead and exposes loop invariants to optimizer branch pruning."
    )]
    #[allow(unsafe_code, reason = "Manages vector length directly for performance")]
    #[inline(always)]
    pub unsafe fn append_carry(dst: &mut InternalArbiUint, carry: Limb) {
        if carry != 0 {
            dst.push_limb(carry);
        }
    }
}
