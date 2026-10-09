//! Harvey's matrix TFT with weighted strided columns and contiguous row leaves.
//!
//! For L=R*C, column u has root exponent root*C and weight w+root*u;
//! rows have root root*R and weight w*R. The invariant 0 <= w < root
//! keeps every exponent reduced without modular division. Splitting the depth
//! approximately equally makes each subtransform visit O(sqrt(L)) slots.

#![expect(
    unsafe_code,
    reason = "Matrix views own disjoint complete slots; power-of-two factors and reduced weights bound all roots and supports"
)]

use core::{num::NonZeroUsize, ptr::from_mut};

use crate::parallel::ParallelExecutor;

use super::{CoefficientView, Limb, SsaRing, TruncatedTransform};

impl TruncatedTransform {
    /// Evaluates a weighted frequency prefix by column transforms then rows.
    ///
    /// # Safety
    /// The view owns len complete initialized ring slots, len is a power of
    /// two, root*len=period, weight<root, and 1<=z,n<=len. Only slots below z
    /// need semi-normalized guards. Scratch is disjoint and contains at least
    /// one coefficient. Ring dimensions and kernel satisfy forward's contract.
    #[expect(
        clippy::too_many_arguments,
        reason = "matrix TFT keeps the dependent column and row passes beside their shared dimension proofs"
    )]
    pub unsafe fn matrix_forward<E: ParallelExecutor>(
        &self,
        mut view: CoefficientView<'_>,
        root: usize,
        weight: usize,
        z: usize,
        n: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let len = view.len();
        if len == 1 {
            return;
        }
        if weight == 0
            && len <= self.max_resident
            && let Some(matrix) = view.contiguous()
        {
            // SAFETY: contiguous view owns every limb; the unweighted
            // resident child satisfies the ordinary TFT support contract.
            unsafe {
                self.forward(matrix, len, root, z, n, executor, scratch);
            }
            return;
        }
        if len == 2 {
            // SAFETY: the view supplies two disjoint initialized slots and
            // scratch is private. Only z established inputs are read; weight
            // is reduced and all ring kernels preserve semi-normalization.
            unsafe {
                let (left, right) = view.pair();
                if n == 1 {
                    if z == 2 {
                        SsaRing::add_in_place(left, right, self.bits);
                    }
                } else if z == 1 {
                    SsaRing::shift_from(right, left, weight, self.bits);
                } else {
                    let difference = from_mut::<[Limb]>(right);
                    SsaRing::add_sub(
                        left,
                        difference,
                        difference.cast::<Limb>(),
                        self.bits,
                        self.kernel,
                    );
                    if weight != 0 {
                        SsaRing::shift_in_place(right, weight, self.bits, scratch);
                    }
                }
            }
            return;
        }
        // SAFETY: view owns len complete slots, len is a power of two,
        // root*len=period, weight<root, and 1<=z,n<=len. Disjoint scratch
        // satisfies matrix_forward's invariant.
        let (width, row_root, row_weight, support, n1, n2) = unsafe {
            self.matrix_forward_columns(&mut view, root, weight, z, n, executor, scratch)
        };
        // SAFETY: n1<=rows, and columns establish support inputs in each
        // requested complete row; row_root*width=period, row_weight<row_root.
        unsafe {
            view.axis(width, false, 0..n1, executor, scratch, &|row, _, work| {
                self.matrix_forward(
                    row,
                    row_root,
                    row_weight,
                    support,
                    width.get(),
                    executor,
                    work,
                );
            });
        }
        if n2 != 0 {
            // SAFETY: n2>0 implies n1<rows and column_outputs=n1+1. The
            // established final row supplies the requested nonempty prefix.
            unsafe {
                view.axis(
                    width,
                    false,
                    n1..n1.unchecked_add(1),
                    executor,
                    scratch,
                    &|row, _, work| {
                        self.matrix_forward(row, row_root, row_weight, support, n2, executor, work);
                    },
                );
            }
        }
    }

    /// Evaluates column transforms for the active support of a matrix.
    ///
    /// # Safety
    /// View owns len>=4 complete initialized slots, len is a power of two,
    /// root*len=period, weight<root, and 1<=z,n<=len.
    /// Disjoint scratch contains at least one coefficient.
    #[expect(
        clippy::too_many_arguments,
        reason = "Column transform decomposition requires ring parameters, support bounds, and execution scratch"
    )]
    pub unsafe fn matrix_forward_columns<E: ParallelExecutor>(
        &self,
        view: &mut CoefficientView<'_>,
        root: usize,
        weight: usize,
        z: usize,
        n: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) -> (NonZeroUsize, usize, usize, usize, usize, usize) {
        let len = view.len();
        debug_assert!(
            len >= 4,
            "a column decomposition requires at least four slots"
        );
        let row_log = len.trailing_zeros() >> 1;
        // SAFETY: row_log<=log2(len)<usize::BITS and len>=4. Both powers
        // are representable, their product is len, and width is positive.
        let (rows, width, column_log, mask) = unsafe {
            let column_log = len.trailing_zeros().unchecked_sub(row_log);
            let rows = 1_usize.unchecked_shl(row_log);
            let width = 1_usize.unchecked_shl(column_log);
            (
                rows,
                NonZeroUsize::new_unchecked(width),
                column_log,
                width.unchecked_sub(1),
            )
        };
        let z1 = z >> column_log;
        let z2 = z & mask;
        let support = if z1 == 0 { z2 } else { width.get() };
        let n1 = n >> column_log;
        let n2 = n & mask;
        // SAFETY: ceil(n/width)<=rows since 1<=n<=len.
        let column_outputs = unsafe { n1.unchecked_add(usize::from(n2 != 0)) };
        // SAFETY: rows,width divide len; root*len=period and weight<root
        // bound each product by period on every supported pointer width.
        // z1<=rows<=len/2 also makes z1+1 representable.
        let (column_root, row_root, row_weight, z_next) = unsafe {
            (
                root.unchecked_mul(width.get()),
                root.unchecked_mul(rows),
                weight.unchecked_mul(rows),
                z1.unchecked_add(1),
            )
        };
        for (range, inputs) in [(0..z2, z_next), (z2..support, z1)] {
            // SAFETY: ranges partition the supported columns within width;
            // each owns rows complete slots. u<width and weight<root imply
            // shift<column_root. The pass joins before row slots are borrowed.
            unsafe {
                view.axis(width, true, range, executor, scratch, &|column, u, work| {
                    let shift = weight.unchecked_add(root.unchecked_mul(u));
                    self.matrix_forward(
                        column,
                        column_root,
                        shift,
                        inputs,
                        column_outputs,
                        executor,
                        work,
                    );
                });
            }
        }
        (width, row_root, row_weight, support, n1, n2)
    }
}
