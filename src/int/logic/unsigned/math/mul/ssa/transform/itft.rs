//! Mixed time/frequency inverse truncated Fourier transform.
//!
//! `ITFT(L,z,n,f)` consumes frequencies below n, `L*a_i` in n..z, and implicit
//! zeros in z..L. It returns `L*a_i` below n and optionally frequency n. Cross
//! butterflies establish the missing time-domain row before its inverse;
//! omitted frequencies are never materialized.
//!
//! Resident nodes use the binary decomposition of Harvey's Algorithm 2;
//! larger nodes use its balanced matrix decomposition. Complete resident
//! inverse children delegate to radix-four kernels:
//! <https://arxiv.org/abs/0810.3203>.

#![expect(
    unsafe_code,
    reason = "Constructed ring geometry bounds recursive partitions; mixed-coordinate supports determine every readable slot"
)]

use core::ptr::{copy_nonoverlapping, from_mut};

use crate::parallel::ParallelExecutor;

use super::{CoefficientView, Limb, SsaRing, SsaTransform, TransformContext, TruncatedTransform};

impl TruncatedTransform {
    /// Inverts complete rows, prepares the partial row through right columns,
    /// inverts that row, then completes the left columns.
    ///
    /// # Safety
    /// The matrix has len complete slots and one private scratch coefficient.
    /// len is a positive power of two and root*len equals the ring shift period.
    /// 0 <= n <= z <= len, 1 <= n+f <= len; n..z holds len-scaled time values.
    /// z..len is implicit zero; its physical slots may contain arbitrary data.
    /// Ring dimensions, architecture kernel, and primitive root satisfy the
    /// same constructed-plan invariants as forward. At the product boundary,
    /// z=n and f=false give the ordinary zero-tail ITFT with full len scaling.
    #[expect(
        clippy::too_many_arguments,
        reason = "ITFT decomposition requires matrix slice, support bounds, root, flags, executor, and scratch"
    )]
    pub unsafe fn inverse<E: ParallelExecutor>(
        &self,
        matrix: &mut [Limb],
        len: usize,
        root: usize,
        z: usize,
        n: usize,
        extra: bool,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if len == 1 {
            return;
        }
        if len >= 4 && len > self.max_resident {
            // SAFETY: the complete matrix owns len slots with the validated
            // mixed-coordinate support, unweighted root, and private scratch.
            unsafe {
                self.matrix_inverse(
                    CoefficientView::new(matrix, len, self.cl),
                    root,
                    0,
                    z,
                    n,
                    extra,
                    executor,
                    scratch,
                );
            }
            return;
        }
        if n == len {
            let ctx = TransformContext {
                mod_bits: self.bits,
                cl: self.cl,
                period: self.period,
                kernel: self.kernel,
                executor,
            };
            // SAFETY: all frequencies are present and the complete radix-four
            // inverse returns exactly the len-scaled time coordinates.
            unsafe {
                SsaTransform::fft_recursive_dit_with_executor(
                    matrix, len, root, scratch, len, &ctx,
                );
            }
            return;
        }
        let half = len >> 1;
        // SAFETY: the complete power-of-two matrix divides into equal children.
        let (low, high) = unsafe { matrix.split_at_mut_unchecked(matrix.len() >> 1) };
        let full_row = n >= half;
        // SAFETY: len is a power of two and the len==1 case returned above,
        // so half is a positive power of two. Its mask computes n modulo half.
        let remainder = n & unsafe { half.unchecked_sub(1) };
        let row_support = z.min(half);
        let high_support = z.saturating_sub(half);
        // This bounded ring index is <= self.period because len >= 2.
        // SAFETY: root*len=period and len>=2, so 2*root<=period fits usize.
        let child_root = unsafe { root.unchecked_mul(2) };
        if full_row {
            // SAFETY: the first row has all half frequencies and disjoint scratch.
            unsafe {
                self.inverse(low, half, child_root, half, half, false, executor, scratch);
            }
        }
        let partial_row = remainder != 0 || extra;
        // SAFETY: columns contain a frequency when full_row, otherwise a scaled
        // time value. Only high_support high coordinates are read. Each private
        // column establishes its part of the time tail before the joined pass
        // releases the partial row. Every running shift stays below bits.
        unsafe {
            self.uniform_columns(
                [low, high],
                remainder,
                row_support,
                high_support,
                root,
                usize::from(full_row),
                partial_row,
                executor,
                scratch,
            );
        }
        if partial_row {
            let row = if full_row { &mut *high } else { &mut *low };
            // SAFETY: right columns establish half-scaled time values after
            // the remainder frequencies; positions above row_support are zero.
            unsafe {
                self.inverse(
                    row,
                    half,
                    child_root,
                    row_support,
                    remainder,
                    extra,
                    executor,
                    scratch,
                );
            }
        }
        // SAFETY: the row inverse establishes the additional frequency; each
        // independent left column now has one or two frequencies and a known
        // tail. The scheduler provides disjoint slots and private scratch, and
        // every running shift stays below bits.
        unsafe {
            self.uniform_columns(
                [low, high],
                0,
                remainder,
                high_support,
                root,
                if full_row { 2 } else { 1 },
                false,
                executor,
                scratch,
            );
        }
    }

    /// Runs one ITFT column range as uniform passes dispatched once per pass,
    /// so no kernel branches on the mixed-coordinate shape of its columns.
    ///
    /// # Safety
    /// The matrix pair, twiddle root, and scratch satisfy the column scheduler
    /// contract; `end <= half` keeps every running shift below the bit width.
    /// `frequencies + usize::from(extra)` lies in `1..=2` on every nonempty
    /// pass and readable inputs carry guards of at most one.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Dispatches the mixed-coordinate shape and peeled identity once per uniform pass, preserving dependent range order without forwarding helpers"
    )]
    pub unsafe fn uniform_columns<E: ParallelExecutor>(
        &self,
        matrices: [&mut [Limb]; 2],
        start: usize,
        end: usize,
        high_support: usize,
        root: usize,
        frequencies: usize,
        extra: bool,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if start >= end {
            return;
        }
        let [low, high] = matrices;
        if frequencies == 2 {
            // Both frequencies are present, so the high-coordinate support is
            // irrelevant. Only index zero has an identity untwiddle; peeling
            // it makes every inverse exponent in the uniform pass positive.
            if start == 0 {
                // SAFETY: start<end establishes both complete first frequency
                // slots. They are disjoint; the high input becomes its difference.
                unsafe {
                    let left = low.get_unchecked_mut(..self.cl.get());
                    let right = high.get_unchecked_mut(..self.cl.get());
                    let difference = from_mut::<[Limb]>(right);
                    SsaRing::add_sub(
                        left,
                        difference,
                        difference.cast::<Limb>(),
                        self.bits,
                        self.kernel,
                    );
                }
            }
            // SAFETY: the nonempty part starts at a positive index below the
            // half length, so 0<shift<bits. Thus period-shift is positive and
            // reduced; all slots and private staging are complete and disjoint.
            unsafe {
                self.columns(
                    [low, high],
                    start.max(1)..end,
                    root,
                    executor,
                    scratch,
                    &|left, right, shift, work| {
                        SsaRing::shift_in_place(
                            right,
                            self.period.get().unchecked_sub(shift),
                            self.bits,
                            work,
                        );
                        let difference = from_mut::<[Limb]>(right);
                        SsaRing::add_sub(
                            left,
                            difference,
                            difference.cast::<Limb>(),
                            self.bits,
                            self.kernel,
                        );
                    },
                );
            }
            return;
        }
        // SAFETY: split partitions start..end into uniform high-present and
        // high-absent subranges; both passes use reduced running shifts and
        // read only their established coordinates.
        let split = high_support.clamp(start, end);
        if frequencies == 0 {
            // SAFETY: both subranges partition the validated range with reduced
            // running shifts; the zero-frequency kernels read only established tails.
            unsafe {
                self.columns(
                    [low, high],
                    start..split,
                    root,
                    executor,
                    scratch,
                    &|left, right, shift, work| {
                        self.column::<0, false, true>(left, right, shift, work);
                    },
                );
                self.columns(
                    [low, high],
                    split..end,
                    root,
                    executor,
                    scratch,
                    &|left, right, shift, work| {
                        self.column::<0, false, false>(left, right, shift, work);
                    },
                );
            }
            return;
        }
        if extra {
            // SAFETY: both subranges partition the validated range with reduced
            // running shifts; each cross kernel reads its uniform coordinates.
            unsafe {
                self.columns(
                    [low, high],
                    start..split,
                    root,
                    executor,
                    scratch,
                    &|left, right, shift, work| {
                        self.column::<1, true, true>(left, right, shift, work);
                    },
                );
                self.columns(
                    [low, high],
                    split..end,
                    root,
                    executor,
                    scratch,
                    &|left, right, shift, work| {
                        self.column::<1, true, false>(left, right, shift, work);
                    },
                );
            }
            return;
        }
        // SAFETY: both subranges partition the validated range with reduced
        // running shifts; each single-frequency kernel reads its established tail.
        unsafe {
            self.columns(
                [low, high],
                start..split,
                root,
                executor,
                scratch,
                &|left, right, shift, work| {
                    self.column::<1, false, true>(left, right, shift, work);
                },
            );
            self.columns(
                [low, high],
                split..end,
                root,
                executor,
                scratch,
                &|left, right, shift, work| {
                    self.column::<1, false, false>(left, right, shift, work);
                },
            );
        }
    }

    /// For a frequency s and known tail `t=2*a_1` the cross butterfly returns
    /// `(2*s-t, zeta*(s-t))`. Zero frequencies return `(2*a_0+2*a_1)/2`.
    ///
    /// The mixed-coordinate shape rides on const parameters, so dispatch
    /// happens once per column pass and no column branches on it.
    ///
    /// # Safety
    /// left, right, and scratch are complete disjoint slots. right is readable
    /// exactly when `HIGH`. shift is reduced; `FREQUENCIES` is zero or one,
    /// and `EXTRA` selects a second output for the one-frequency case.
    /// All readable inputs have guard <= 1.
    pub unsafe fn column<const FREQUENCIES: usize, const EXTRA: bool, const HIGH: bool>(
        &self,
        left: &mut [Limb],
        right: &mut [Limb],
        shift: usize,
        scratch: &mut [Limb],
    ) {
        // SAFETY: all leaf operations use complete disjoint slots; right is
        // read only under its support condition. Exponents are reduced and
        // ring operations preserve semi-normalized guards on this target.
        unsafe {
            if FREQUENCIES == 0 {
                if HIGH {
                    SsaRing::add_in_place(left, right, self.bits);
                }
                SsaRing::halve_in_place(left, self.bits);
            } else if EXTRA {
                if !HIGH {
                    SsaRing::shift_from(right, left, shift, self.bits);
                    SsaRing::double_in_place(left, self.bits);
                    return;
                }
                // d=s-t; the low result is s+d=2s-t. The high slot receives
                // zeta*d without overlapping the disjoint source d.
                let work = scratch.get_unchecked_mut(..self.cl.get());
                // The column contract supplies cl initialized left limbs
                // disjoint from the private cl-limb scratch coefficient.
                copy_nonoverlapping(left.as_ptr(), work.as_mut_ptr(), self.cl.get());
                SsaRing::sub_in_place(work, right, self.bits);
                SsaRing::add_in_place(left, work, self.bits);
                SsaRing::shift_from(right, work, shift, self.bits);
            } else {
                SsaRing::double_in_place(left, self.bits);
                if HIGH {
                    SsaRing::sub_in_place(left, right, self.bits);
                }
            }
        }
    }
}
