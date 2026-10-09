//! Limb-slice addition and subtraction primitives.
//!
//! Columns compute residues modulo the limb base and propagate binary flags.
//!
//! References:
//! - D. E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*,
//!   3rd ed., Addison-Wesley, 1997, Section 4.3.1, Algorithms A and S, pp. 265–268.
//! - R. P. Brent and P. Zimmermann, *Modern Computer Arithmetic*, Cambridge University
//!   Press, 2011, Section 1.2. DOI: 10.1017/CBO9780511921698.

#![expect(
    unsafe_code,
    reason = "Validated slice spans and raw-pointer contracts establish bounds, initialization, and disjointness for infallible carry and borrow kernels."
)]

use core::ptr::copy_nonoverlapping;

use super::{ArchKernels, Limb};

/// Namespace for shared addition and subtraction limb primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addition;

impl Addition {
    /// Adds `src` into `dst`, returning the carry out.
    ///
    /// The caller establishes `dst.len() >= src.len()` when sizing or
    /// partitioning the initialized destination.
    #[expect(
        clippy::inline_always,
        reason = "Inlining this helper eliminates call overhead and exposes slice invariants to the optimizer."
    )]
    #[inline(always)]
    pub fn add_slice_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        let src_len = src.len();
        debug_assert!(
            dst.len() >= src_len,
            "addition destination is shorter than source"
        );
        if src_len == 0 {
            return 0;
        }
        if src_len == 1 {
            // SAFETY: src_len == 1 and the caller's destination-length
            // invariant establishes an initialized element in each slice.
            unsafe {
                let (sum, carry) = (*dst.get_unchecked(0)).overflowing_add(*src.get_unchecked(0));
                *dst.get_unchecked_mut(0) = sum;
                return Limb::from(carry);
            }
        }
        // SAFETY: the caller establishes dst.len() >= src_len before entering
        // this kernel; both active spans are initialized.
        // As `dst` is `&mut [Limb]` and `src` is `&[Limb]`, the buffers are valid,
        // properly aligned, and guaranteed disjoint (non-overlapping).
        unsafe { ArchKernels::add_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), src_len) }
    }

    /// Subtracts `src` from `dst`, returning the borrow out.
    ///
    /// The caller establishes `dst.len() >= src.len()` when sizing or
    /// partitioning the initialized destination.
    #[expect(
        clippy::inline_always,
        reason = "Inlining this helper eliminates call overhead and exposes slice invariants to the optimizer."
    )]
    #[inline(always)]
    pub fn sub_slice_in_place(dst: &mut [Limb], src: &[Limb]) -> Limb {
        let src_len = src.len();
        debug_assert!(
            dst.len() >= src_len,
            "subtraction destination is shorter than source"
        );
        if src_len == 0 {
            return 0;
        }
        if src_len == 1 {
            // SAFETY: src_len == 1 and the caller's destination-length
            // invariant establishes an initialized element in each slice.
            unsafe {
                let (diff, borrow) = (*dst.get_unchecked(0)).overflowing_sub(*src.get_unchecked(0));
                *dst.get_unchecked_mut(0) = diff;
                return Limb::from(borrow);
            }
        }
        // SAFETY: the caller establishes dst.len() >= src_len before entering
        // this kernel; both active spans are initialized.
        // As `dst` is `&mut [Limb]` and `src` is `&[Limb]`, the buffers are valid,
        // properly aligned, and guaranteed disjoint (non-overlapping).
        unsafe { ArchKernels::sub_limbs_unchecked(dst.as_mut_ptr(), src.as_ptr(), src_len) }
    }

    /// Copies `rem` limbs from `src` to `dst`, propagating an initial `carry`.
    ///
    /// Returns the remaining carry out.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must be valid for reading and writing `rem` limbs,
    /// respectively, and must not point to overlapping memory regions. Both
    /// pointers remain aligned and non-null even when `rem == 0`. The incoming
    /// carry is zero or one; only limb values wrap modulo the limb base.
    #[expect(
        clippy::inline_always,
        reason = "Core carry-propagation primitive used across all addition paths"
    )]
    #[inline(always)]
    pub unsafe fn copy_tail_with_carry(
        dst: *mut Limb,
        src: *const Limb,
        rem: usize,
        carry: Limb,
    ) -> Limb {
        debug_assert!(carry <= 1, "tail copying requires a binary carry");
        if carry == 0 {
            // SAFETY: the caller supplies aligned, disjoint spans covering rem
            // limbs; the source is initialized and the output is writable.
            unsafe {
                copy_nonoverlapping(src, dst, rem);
            }
            return 0;
        }

        // The nonzero binary carry is one. A MAX limb writes zero and retains
        // it; any smaller limb absorbs it, leaving an unchanged suffix.
        let mut i = 0_usize;
        while i < rem {
            // SAFETY: i < rem bounds the caller's aligned initialized source.
            let limb = unsafe { *src.add(i) };
            if limb != Limb::MAX {
                // SAFETY: limb < MAX proves limb + 1 fits. The caller's aligned,
                // disjoint output covers i < rem; i + 1 <= rem fits usize.
                // After incrementing, i < rem bounds the initialized source
                // suffix and writable destination suffix of rem - i limbs.
                unsafe {
                    *dst.add(i) = limb.unchecked_add(1);
                    i = i.unchecked_add(1);
                    if i < rem {
                        copy_nonoverlapping(src.add(i), dst.add(i), rem.unchecked_sub(i));
                    }
                }
                return 0;
            }
            // SAFETY: i < rem bounds the aligned writable output, and
            // i + 1 <= rem fits usize. MAX + 1 has residue zero and carry one.
            unsafe {
                *dst.add(i) = 0;
                i = i.unchecked_add(1);
            }
        }
        1
    }

    /// Copies `rem` limbs from `src` to `dst`, propagating an initial `borrow`.
    ///
    /// Returns the remaining borrow out.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must be valid for reading and writing `rem` limbs,
    /// respectively, and must not point to overlapping memory regions. Both
    /// pointers remain aligned and non-null even when `rem == 0`. The incoming
    /// borrow is zero or one; only limb values wrap modulo the limb base.
    #[expect(
        clippy::inline_always,
        reason = "Core borrow-propagation primitive used across subtraction paths"
    )]
    #[inline(always)]
    pub unsafe fn copy_tail_with_borrow(
        dst: *mut Limb,
        src: *const Limb,
        rem: usize,
        borrow: Limb,
    ) -> Limb {
        debug_assert!(borrow <= 1, "tail copying requires a binary borrow");
        if borrow == 0 {
            // SAFETY: the caller supplies aligned, disjoint spans covering rem
            // limbs; the source is initialized and the output is writable.
            unsafe {
                copy_nonoverlapping(src, dst, rem);
            }
            return 0;
        }

        // The nonzero binary borrow is one. A zero limb writes MAX and retains
        // it; any positive limb absorbs it, leaving an unchanged suffix.
        let mut i = 0_usize;
        while i < rem {
            // SAFETY: i < rem bounds the caller's aligned initialized source.
            let limb = unsafe { *src.add(i) };
            if limb != 0 {
                // SAFETY: limb > 0 proves limb - 1 fits. The caller's aligned,
                // disjoint output covers i < rem; i + 1 <= rem fits usize.
                // After incrementing, i < rem bounds the initialized source
                // suffix and writable destination suffix of rem - i limbs.
                unsafe {
                    *dst.add(i) = limb.unchecked_sub(1);
                    i = i.unchecked_add(1);
                    if i < rem {
                        copy_nonoverlapping(src.add(i), dst.add(i), rem.unchecked_sub(i));
                    }
                }
                return 0;
            }
            // SAFETY: i < rem bounds the aligned writable output, and
            // i + 1 <= rem fits usize. Zero - 1 has residue MAX and borrow one.
            unsafe {
                *dst.add(i) = Limb::MAX;
                i = i.unchecked_add(1);
            }
        }
        1
    }

    /// Propagates a binary carry through `limbs`, returning the carry out.
    #[expect(
        clippy::inline_always,
        reason = "Callers supply binary flags and known tail lengths; inlining exposes those predicates to eliminate redundant dispatch."
    )]
    #[inline(always)]
    pub fn propagate_carry(limbs: &mut [Limb], carry: Limb) -> Limb {
        debug_assert!(carry <= 1, "carry propagation requires a binary carry");
        if carry == 0 || limbs.is_empty() {
            return carry;
        }
        // SAFETY: the early return excludes zero and the incoming carry is
        // binary, so exactly one is propagated. The nonempty slice provides
        // initialized writable storage for its full length.
        unsafe { ArchKernels::propagate_carry_unchecked(limbs.as_mut_ptr(), limbs.len(), 1) }
    }

    /// Propagates a binary borrow through `limbs`, returning the borrow out.
    #[expect(
        clippy::inline_always,
        reason = "Callers supply binary flags and known tail lengths; inlining exposes those predicates to eliminate redundant dispatch."
    )]
    #[inline(always)]
    pub fn propagate_borrow(limbs: &mut [Limb], borrow: Limb) -> Limb {
        debug_assert!(borrow <= 1, "borrow propagation requires a binary borrow");
        if borrow == 0 || limbs.is_empty() {
            return borrow;
        }
        // SAFETY: the early return excludes zero and the incoming borrow is
        // binary, so exactly one is propagated. The nonempty slice provides
        // initialized writable storage for its full length.
        unsafe { ArchKernels::propagate_borrow_unchecked(limbs.as_mut_ptr(), limbs.len(), 1) }
    }

    /// Subtracts `src` and `borrow` from zero into `dst`.
    ///
    /// # Safety
    ///
    /// `src` contains `len` initialized limbs. `dst` is aligned and writable
    /// for `len` limbs, but need not be initialized. The spans do not overlap,
    /// remain valid throughout the call, and `borrow` is zero or one.
    #[expect(
        clippy::inline_always,
        reason = "Inlining this arithmetic loop eliminates call overhead and exposes loop invariants to optimizer branch pruning."
    )]
    #[inline(always)]
    pub unsafe fn negate_with_borrow(
        dst: *mut Limb,
        src: *const Limb,
        len: usize,
        mut borrow: Limb,
    ) -> Limb {
        debug_assert!(borrow <= 1, "negation requires a binary borrow");
        let mut i = 0_usize;
        if borrow == 0 {
            // With no incoming borrow, zero source limbs leave the borrow at
            // zero. The first nonzero limb writes -src modulo B and establishes
            // borrow = 1; no later column can clear that borrow.
            while i < len {
                // SAFETY: the caller supplies aligned, disjoint spans for len
                // limbs; i < len bounds the initialized read and output write,
                // and i + 1 <= len fits usize on every pointer width.
                let limb = unsafe {
                    let limb = *src.add(i);
                    *dst.add(i) = limb.wrapping_neg();
                    i = i.unchecked_add(1);
                    limb
                };
                if limb != 0 {
                    borrow = 1;
                    break;
                }
            }
        }

        // Here i < len implies borrow = 1: an initial one skips the first
        // loop, and an initial zero leaves it early only at a nonzero limb.
        // For 0 <= src < B, 0-src-1 is in [-B, -1], so its borrow remains one
        // and its residue is B-1-src = !src. Columns are now independent.
        while i < len {
            // SAFETY: the caller's aligned spans cover len limbs; i < len
            // bounds the initialized read and disjoint, write-only output.
            unsafe {
                *dst.add(i) = !*src.add(i);
            }
            // SAFETY: the loop has i < len, hence i + 1 <= len <= usize::MAX.
            i = unsafe { i.unchecked_add(1) };
        }
        borrow
    }
}
