//! Matrix ITFT in complete-row, right-column, partial-row, left-column order.
//!
//! Right columns establish the scaled time tail needed by the partial row.
//! Every axis pass joins before the next dependency phase. Weighted two-point
//! leaves share the binary ITFT's ring identities and scaling primitives.

#![expect(
    unsafe_code,
    reason = "Joined axis passes establish each mixed coordinate before it is read; matrix ownership bounds slots and reduced roots"
)]

use core::{num::NonZeroUsize, ptr::from_mut};

use crate::parallel::ParallelExecutor;

use super::{CoefficientView, Limb, SsaRing, TruncatedTransform};

impl TruncatedTransform {
    /// Recovers len-scaled time coordinates and optionally frequency n.
    ///
    /// # Safety
    /// The view owns len initialized complete slots with 0<=n<=z<=len and
    /// 1<=n+extra<=len. Slots below n hold weighted frequencies; n..z holds
    /// len-scaled time coordinates with semi-normalized guards. The rest is
    /// implicit zero. len is a power of two, root*len=period, weight<root;
    /// ring/kernel and disjoint scratch satisfy the ordinary inverse contract.
    #[expect(
        clippy::too_many_arguments,
        reason = "matrix ITFT evaluates scaled multi-axis passes across rows and columns in dependency order"
    )]
    pub unsafe fn matrix_inverse<E: ParallelExecutor>(
        &self,
        mut view: CoefficientView<'_>,
        root: usize,
        weight: usize,
        z: usize,
        n: usize,
        extra: bool,
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
            // SAFETY: a contiguous unweighted resident child carries the
            // ordinary ITFT's exact mixed-coordinate and scaling contract.
            unsafe {
                self.inverse(matrix, len, root, z, n, extra, executor, scratch);
            }
            return;
        }
        if len == 2 {
            // SAFETY: the two complete slots and scratch are disjoint; these
            // cases read the right slot exactly when its coordinate is known.
            // Here weight<root=period/2: a positive weight has a positive
            // reduced inverse exponent. Every ring operation preserves guards.
            unsafe {
                let (left, right) = view.pair();
                match (n, extra, z == 2) {
                    (2, _, _) => {
                        // Weighted strided leaves admit both zero and positive
                        // weights. Their identity remains a valid leaf state.
                        if weight != 0 {
                            SsaRing::shift_in_place(
                                right,
                                self.period.get().unchecked_sub(weight),
                                self.bits,
                                scratch,
                            );
                        }
                        let difference = from_mut::<[Limb]>(right);
                        SsaRing::add_sub(
                            left,
                            difference,
                            difference.cast::<Limb>(),
                            self.bits,
                            self.kernel,
                        );
                    }
                    (0, _, true) => self.column::<0, false, true>(left, right, weight, scratch),
                    (0, _, false) => self.column::<0, false, false>(left, right, weight, scratch),
                    (_, true, true) => self.column::<1, true, true>(left, right, weight, scratch),
                    (_, true, false) => self.column::<1, true, false>(left, right, weight, scratch),
                    (_, false, true) => self.column::<1, false, true>(left, right, weight, scratch),
                    (_, false, false) => {
                        self.column::<1, false, false>(left, right, weight, scratch);
                    }
                }
            }
            return;
        }
        let row_log = len.trailing_zeros() >> 1;
        // SAFETY: row_log<=log2(len)<usize::BITS and len>=4. The child
        // powers are representable, multiply to len, and width is positive.
        let (rows, width, column_log) = unsafe {
            let column_log = len.trailing_zeros().unchecked_sub(row_log);
            let rows = 1_usize.unchecked_shl(row_log);
            let width = 1_usize.unchecked_shl(column_log);
            (rows, NonZeroUsize::new_unchecked(width), column_log)
        };
        let n1 = n >> column_log;
        // SAFETY: rows divides len, root*len=period, weight<root.
        let (row_root, row_weight) =
            unsafe { (root.unchecked_mul(rows), weight.unchecked_mul(rows)) };
        // SAFETY: n1<=rows; each selected complete row owns width frequencies.
        // Their inverses establish width-scaled columns before the next pass.
        unsafe {
            view.axis(width, false, 0..n1, executor, scratch, &|row, _, work| {
                self.matrix_inverse(
                    row,
                    row_root,
                    row_weight,
                    width.get(),
                    width.get(),
                    false,
                    executor,
                    work,
                );
            });
            self.matrix_inverse_columns_and_partial(
                &mut view, root, weight, z, n, extra, executor, scratch,
            );
        }
    }

    /// Inverts right columns, the partial row (if any), and left columns.
    ///
    /// # Safety
    /// View owns len>=4 complete slots, len is a power of two, root*len=period,
    /// weight<root, 0<=n<=z<=len, and 1<=n+extra<=len.
    /// All complete rows 0..n1 are already inverted and hold width-scaled time coordinates.
    /// The partial row retains its n2 frequencies and established scaled tail;
    /// omitted coordinates are implicit zero. Readable guards are at most one.
    /// Disjoint scratch contains at least one coefficient.
    #[expect(
        clippy::too_many_arguments,
        reason = "Column and partial row inverse binds geometric dimensions, roots, and execution scratch"
    )]
    pub unsafe fn matrix_inverse_columns_and_partial<E: ParallelExecutor>(
        &self,
        view: &mut CoefficientView<'_>,
        root: usize,
        weight: usize,
        z: usize,
        n: usize,
        extra: bool,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let len = view.len();
        debug_assert!(
            len >= 4,
            "a column decomposition requires at least four slots"
        );
        let row_log = len.trailing_zeros() >> 1;
        // SAFETY: row_log<=log2(len)<usize::BITS and len>=4. The child
        // powers are representable, multiply to len, and width is positive.
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
        let partial = n2 != 0 || extra;
        // SAFETY: rows,width divide len, root*len=period, weight<root. Thus
        // child roots and weight products are bounded by the ring period.
        // n1,z1<=rows<=len/2 also bound both incremented supports.
        let (column_root, row_root, row_weight, z_next, n_next) = unsafe {
            (
                root.unchecked_mul(width.get()),
                root.unchecked_mul(rows),
                weight.unchecked_mul(rows),
                z1.unchecked_add(1),
                n1.unchecked_add(1),
            )
        };
        let right_split = n2.max(z2);
        for (range, inputs) in [(n2..right_split, z_next), (right_split..support, z1)] {
            // SAFETY: both ranges lie in width columns; each has n1 known
            // frequencies and the indicated scaled time support. The optional
            // extra output establishes the partial row. shift<column_root.
            unsafe {
                view.axis(width, true, range, executor, scratch, &|column, u, work| {
                    let shift = weight.unchecked_add(root.unchecked_mul(u));
                    self.matrix_inverse(
                        column,
                        column_root,
                        shift,
                        inputs,
                        n1,
                        partial,
                        executor,
                        work,
                    );
                });
            }
        }
        if partial {
            // SAFETY: partial implies n1<rows. Joined right columns establish
            // the time tail after n2 frequencies. Inverting this row completes
            // the left columns' last frequency and optionally frequency n.
            unsafe {
                view.axis(
                    width,
                    false,
                    n1..n_next,
                    executor,
                    scratch,
                    &|row, _, work| {
                        self.matrix_inverse(
                            row, row_root, row_weight, support, n2, extra, executor, work,
                        );
                    },
                );
            }
        }
        let left_split = n2.min(z2);
        for (range, inputs) in [(0..left_split, z_next), (left_split..n2, z1)] {
            // SAFETY: the ranges lie in n2<width columns. The partial row
            // establishes n1+1 frequencies per column; its inverse restores
            // len-scaled time since rows already contributed width scaling.
            unsafe {
                view.axis(width, true, range, executor, scratch, &|column, u, work| {
                    let shift = weight.unchecked_add(root.unchecked_mul(u));
                    self.matrix_inverse(
                        column,
                        column_root,
                        shift,
                        inputs,
                        n_next,
                        false,
                        executor,
                        work,
                    );
                });
            }
        }
    }
}
