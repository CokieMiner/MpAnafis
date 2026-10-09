//! Fixed-width significance, copying, and zero-extended comparison.

#![expect(
    unsafe_code,
    reason = "Monotone scans and validated destination widths bound initialized nonoverlapping spans"
)]

use core::{cmp::Ordering, ptr::copy_nonoverlapping};

use super::{Limb, SharedEval};

impl SharedEval {
    /// Returns one past the highest nonzero limb.
    #[expect(
        clippy::inline_always,
        reason = "Reconstruction callers reuse the significance scan without a separate call frame"
    )]
    #[inline(always)]
    pub fn active_len(limbs: &[Limb]) -> usize {
        let mut len = limbs.len();
        while len > 0 {
            // SAFETY: the loop condition proves len > 0.
            let index = unsafe { len.unchecked_sub(1) };
            // SAFETY: len is positive and never exceeds limbs.len().
            if unsafe { *limbs.get_unchecked(index) } != 0 {
                break;
            }
            len = index;
        }
        len
    }

    /// Copies a polynomial part and zeroes the retained guard suffix.
    pub fn copy_part(dst: &mut [Limb], src: &[Limb]) {
        debug_assert!(src.len() < dst.len(), "the evaluation retains a guard limb");
        // SAFETY: the evaluation reserves the initialized source width followed
        // by a guard. Rust's exclusive destination borrow proves disjointness.
        let guard = unsafe {
            let (body, guard) = dst.split_at_mut_unchecked(src.len());
            copy_nonoverlapping(src.as_ptr(), body.as_mut_ptr(), src.len());
            guard
        };
        guard.fill(0);
    }

    /// Compares equal-width magnitudes after zero-extending the narrower input.
    pub fn compare_with_zero_extension(wide: &[Limb], narrow: &[Limb]) -> Ordering {
        debug_assert!(
            narrow.len() <= wide.len(),
            "the narrow operand fits the shared prefix"
        );
        // SAFETY: the comparison contract supplies narrow.len() <= wide.len().
        let (shared, extension) = unsafe { wide.split_at_unchecked(narrow.len()) };
        if extension.iter().any(|limb| *limb != 0) {
            Ordering::Greater
        } else {
            shared.iter().rev().cmp(narrow.iter().rev())
        }
    }
}
