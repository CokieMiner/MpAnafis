//! Exclusive strided coefficient views and disjoint row/column scheduling.
//!
//! A view owns only its coefficient slots, never the gaps between them. Batches
//! partition logical coordinates before forming references, so simultaneous
//! columns may have overlapping address envelopes but never alias a limb.

#![expect(
    unsafe_code,
    reason = "Exclusive view lifetimes and logical batch partitions prove strided slot initialization and disjoint ownership"
)]

use core::{
    marker::PhantomData,
    num::NonZeroUsize,
    ops::{Div, Range},
    slice::from_raw_parts_mut,
};

use crate::parallel::ParallelExecutor;

use super::{Limb, SsaTransform};

/// Exclusive ownership of `len` initialized, equally spaced coefficient slots.
pub struct CoefficientView<'matrix> {
    pointer: *mut Limb,
    len: usize,
    stride: usize,
    cl: NonZeroUsize,
    lifetime: PhantomData<&'matrix mut [Limb]>,
}

/// A partition of independent transforms within one rectangular view.
struct TransformBatch<'matrix> {
    pointer: *mut Limb,
    range: Range<usize>,
    spacing: usize,
    len: usize,
    stride: usize,
    cl: NonZeroUsize,
    lifetime: PhantomData<&'matrix mut [Limb]>,
}

// SAFETY: the lifetime owns every slot exclusively; moving the view transfers
// ownership. Limb is Send, and no reference to the gaps is ever constructed.
unsafe impl Send for CoefficientView<'_> {}

// SAFETY: batches own disjoint logical slots for their complete lifetime. A
// split transfers each slot to exactly one child, including strided columns.
unsafe impl Send for TransformBatch<'_> {}

impl<'matrix> CoefficientView<'matrix> {
    /// Borrows a complete contiguous coefficient matrix.
    ///
    /// # Safety
    /// len is positive, and matrix contains exactly len complete cl-limb slots.
    /// The validated transform plan supplies these dimensions before entry.
    pub unsafe fn new(matrix: &'matrix mut [Limb], len: usize, cl: NonZeroUsize) -> Self {
        debug_assert!(len > 0, "matrix must contain at least one coefficient");
        debug_assert_eq!(
            len.checked_mul(cl.get()),
            Some(matrix.len()),
            "matrix must contain exactly the declared coefficient slots"
        );
        Self {
            pointer: matrix.as_mut_ptr(),
            len,
            stride: cl.get(),
            cl,
            lifetime: PhantomData,
        }
    }

    /// Returns the number of owned coefficient slots.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Executes a row or column range, joining all work before releasing self.
    /// `width` is the number of columns in the row-major logical matrix.
    ///
    /// # Safety
    /// width is positive and divides len; range is ordered and lies within
    /// width columns or len/width rows, according to the selected axis.
    pub unsafe fn axis<E, F>(
        &mut self,
        width: NonZeroUsize,
        columns: bool,
        range: Range<usize>,
        executor: &E,
        scratch: &mut [Limb],
        kernel: &F,
    ) where
        E: ParallelExecutor,
        F: Fn(CoefficientView<'_>, usize, &mut [Limb]) + Sync,
    {
        debug_assert!(
            self.len.is_multiple_of(width.get()),
            "row width must divide the positive view length"
        );
        let rows = self.len.div(width);
        let count = if columns { width.get() } else { rows };
        debug_assert!(
            range.start <= range.end && range.end <= count,
            "axis range must lie inside the matrix rectangle"
        );
        if range.is_empty() {
            return;
        }
        // SAFETY: width <= len and stride*len is representable: every view is
        // a row/column subdivision of the original representable slice span.
        let row_stride = unsafe { self.stride.unchecked_mul(width.get()) };
        let batch = TransformBatch {
            pointer: self.pointer,
            range,
            spacing: if columns { self.stride } else { row_stride },
            len: if columns { rows } else { width.get() },
            stride: if columns { row_stride } else { self.stride },
            cl: self.cl,
            lifetime: PhantomData,
        };
        batch.run(executor, scratch, kernel);
    }

    /// Returns a slice only when every limb in its span belongs to this view.
    pub const fn contiguous(&mut self) -> Option<&mut [Limb]> {
        if self.stride != self.cl.get() {
            return None;
        }
        // SAFETY: len complete adjacent initialized slots belong exclusively
        // to this view. Their span fits the original allocation and is aligned;
        // the returned reference cannot outlive the exclusive borrow of self.
        unsafe {
            let span = self.len.unchecked_mul(self.cl.get());
            Some(from_raw_parts_mut(self.pointer, span))
        }
    }

    /// Borrows both slots of a two-point leaf without borrowing its gaps.
    ///
    /// # Safety
    /// The view contains exactly two slots.
    pub unsafe fn pair(&mut self) -> (&mut [Limb], &mut [Limb]) {
        debug_assert_eq!(
            self.len, 2,
            "a butterfly view must contain exactly two slots"
        );
        // SAFETY: stride >= cl, so these two aligned initialized cl-limb slots
        // are disjoint and lie in the original allocation. The references are
        // bounded by this exclusive borrow, which prevents a concurrent batch.
        unsafe {
            (
                from_raw_parts_mut(self.pointer, self.cl.get()),
                from_raw_parts_mut(self.pointer.add(self.stride), self.cl.get()),
            )
        }
    }
}

impl TransformBatch<'_> {
    /// Recursively partitions independent transforms and the staging arena.
    fn run<E, F>(self, executor: &E, scratch: &mut [Limb], kernel: &F)
    where
        E: ParallelExecutor,
        F: Fn(CoefficientView<'_>, usize, &mut [Limb]) + Sync,
    {
        // SAFETY: the range is ordered, cl is positive, and len complete slots
        // form a subset of the original slice. All dimensions are representable.
        let (count, work, scratch_slots) = unsafe {
            (
                self.range.end.unchecked_sub(self.range.start),
                self.len.unchecked_mul(self.cl.get()),
                scratch.len().div(self.cl),
            )
        };
        if SsaTransform::has_parallel_work(count, work, executor.parallelism().get())
            && scratch_slots >= 2
        {
            // SAFETY: the midpoint lies in the nonempty representable range.
            let middle = unsafe { self.range.start.unchecked_add(count >> 1) };
            let left = Self {
                range: self.range.start..middle,
                ..self
            };
            let right = Self {
                range: middle..self.range.end,
                ..self
            };
            let (first, second) = scratch.split_at_mut(scratch.len() >> 1);
            executor.join(
                || left.run(executor, first, kernel),
                || right.run(executor, second, kernel),
            );
            return;
        }
        for index in self.range {
            // SAFETY: row/column indexing selects an initialized slot in the
            // parent rectangle. Distinct batch indices select disjoint sets;
            // this child is consumed synchronously before the next is formed.
            let pointer = unsafe { self.pointer.add(index.unchecked_mul(self.spacing)) };
            kernel(
                CoefficientView {
                    pointer,
                    len: self.len,
                    stride: self.stride,
                    cl: self.cl,
                    lifetime: PhantomData,
                },
                index,
                scratch,
            );
        }
    }
}
