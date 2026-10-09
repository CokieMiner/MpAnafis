//! Reusable scratch ownership for the recursive half-GCD reduction.
//!
//! Every recursion level retains one [`HgcdFrame`] with preallocated operands,
//! products and matrix entries. Capacities persist across subsequent reductions.

#![expect(
    unsafe_code,
    reason = "Normalized quotient limbs and six-bit Jacobi states bound reads; mutually exclusive TLS paths consume the operation exactly once"
)]

#[cfg(feature = "std")]
use core::cell::{Cell, RefCell};
use core::{cmp::Ordering, mem::swap};
#[cfg(feature = "std")]
use std::thread_local;

use alloc::vec::Vec;

use super::{
    DivScratch, Division, Gcd, HGCD_BLOCK_THRESHOLD, HgcdMatrix, InternalMpUint,
    LEHMER_BRANCHLESS_THRESHOLD,
};

#[derive(Debug)]
pub struct HgcdFrame {
    /// Narrow simulation policy inherited from the initial reduction width.
    pub branchless: bool,
    pub matrix: HgcdMatrix,
    pub next_matrix: HgcdMatrix,
    pub u: InternalMpUint,
    pub v: InternalMpUint,
    pub next_u: InternalMpUint,
    pub next_v: InternalMpUint,
    pub quotient: InternalMpUint,
    pub remainder: InternalMpUint,
    pub product_a: InternalMpUint,
    pub product_b: InternalMpUint,
    pub sum_a: InternalMpUint,
    pub sum_b: InternalMpUint,
}

impl Default for HgcdFrame {
    fn default() -> Self {
        Self {
            branchless: false,
            matrix: HgcdMatrix::default(),
            next_matrix: HgcdMatrix::default(),
            u: InternalMpUint::zero(),
            v: InternalMpUint::zero(),
            next_u: InternalMpUint::zero(),
            next_v: InternalMpUint::zero(),
            quotient: InternalMpUint::zero(),
            remainder: InternalMpUint::zero(),
            product_a: InternalMpUint::zero(),
            product_b: InternalMpUint::zero(),
            sum_a: InternalMpUint::zero(),
            sum_b: InternalMpUint::zero(),
        }
    }
}

impl HgcdFrame {
    /// Applies one simulated batch or an exact subdivision to this frame.
    /// A root without a matrix consumer omits matrix bookkeeping. Jacobi
    /// state follows the full operands' residues, even in a truncated window.
    pub fn reduce_step<const JACOBI: bool>(
        &mut self,
        target_len: usize,
        scratch: &mut DivScratch,
        track_matrix: bool,
        state: &mut usize,
    ) {
        if self.u.cmp(&self.v) == Ordering::Less {
            swap(&mut self.u, &mut self.v);
            if track_matrix {
                self.matrix.swap_columns();
            }
            if JACOBI {
                *state = ((*state & 3) << 2) | ((*state >> 2) & 3) | ((*state ^ 32) & 0x30);
            }
        }
        let mut candidate = *state;
        let (u0, v0, u1, v1, even) = Gcd::simulate_step::<true>(
            self.u.limbs(),
            self.v.limbs(),
            None,
            self.branchless,
            |q| {
                if JACOBI {
                    let index = (candidate << 2) | (q & 3);
                    // SAFETY: construction, relabeling and table outputs
                    // preserve six state bits; the quotient supplies two more.
                    candidate = usize::from(unsafe { *Gcd::JACOBI_QUOTIENT.get_unchecked(index) });
                }
            },
        );
        let identity = u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1;
        if !identity
            && Gcd::lehmer_update_dispatched(
                &mut self.u,
                &mut self.v,
                &mut self.next_u,
                &mut self.next_v,
                u0,
                v0,
                u1,
                v1,
                even,
                None,
            )
        {
            if track_matrix {
                self.matrix
                    .update_small(&mut self.next_matrix, u0, v0, u1, v1, even);
            }
            if JACOBI {
                *state = candidate;
            }
            return;
        }
        self.reduce_subdiv_step::<JACOBI>(target_len, scratch, track_matrix, state);
    }

    /// Commits a subtraction, then divides while retaining the HGCD boundary.
    /// For a quotient crossing that boundary, q-1 and r+v preserve the exact
    /// transition. Matrix and Jacobi updates consume the corrected quotient.
    fn reduce_subdiv_step<const JACOBI: bool>(
        &mut self,
        target_len: usize,
        scratch: &mut DivScratch,
        track_matrix: bool,
        state: &mut usize,
    ) {
        debug_assert!(
            self.u.cmp(&self.v) != Ordering::Less && !self.v.is_zero(),
            "subdivision requires ordered nonzero operands"
        );
        self.u.sub_assign(&self.v);
        let subtraction_reaches_target = self.u.limbs().len() <= target_len;
        // (u, v) = [[1, 1], [0, 1]] * (u - v, v).
        if track_matrix {
            self.matrix.m01.add_assign(&self.matrix.m00);
            self.matrix.m11.add_assign(&self.matrix.m10);
        }
        if JACOBI {
            let index = (*state << 2) | 1;
            // SAFETY: six state bits and quotient one bound index below 256.
            let exchanged = usize::from(unsafe { *Gcd::JACOBI_QUOTIENT.get_unchecked(index) });
            // Undo the table's exchange: subtraction retains the operand slots.
            *state = ((exchanged & 3) << 2) | ((exchanged >> 2) & 3) | ((exchanged ^ 32) & 0x30);
        }
        match self.u.cmp(&self.v) {
            Ordering::Equal => return,
            Ordering::Less => {
                swap(&mut self.u, &mut self.v);
                if track_matrix {
                    self.matrix.swap_columns();
                }
                if JACOBI {
                    *state = ((*state & 3) << 2) | ((*state >> 2) & 3) | ((*state ^ 32) & 0x30);
                }
            }
            Ordering::Greater => {}
        }
        if subtraction_reaches_target {
            return;
        }

        if let Some(mut q) = Gcd::fast_small_div_step(&mut self.u, &self.v) {
            if self.u.limbs().len() <= target_len {
                self.u.add_assign(&self.v);
                // SAFETY: fast_small_div_step receives u >= v > 0 here, so
                // its exact quotient is at least one before boundary correction.
                q = unsafe { q.unchecked_sub(1) };
            }
            if q == 0 {
                if track_matrix {
                    self.matrix.swap_columns();
                }
                if JACOBI {
                    *state = ((*state & 3) << 2) | ((*state >> 2) & 3) | ((*state ^ 32) & 0x30);
                }
            } else {
                if track_matrix {
                    self.matrix.update_quotient_scalar(&mut self.next_matrix, q);
                }
                if JACOBI {
                    let index = (*state << 2) | (q & 3);
                    // SAFETY: six state bits and two quotient bits give index < 256.
                    *state = usize::from(unsafe { *Gcd::JACOBI_QUOTIENT.get_unchecked(index) });
                }
            }
            swap(&mut self.u, &mut self.v);
            return;
        }

        // A plain GCD root consumes only the corrected remainder. Jacobi
        // additionally needs the quotient's low two bits for its sign.
        if !track_matrix && !JACOBI {
            Division::rem_into(&self.u, &self.v, &mut self.remainder, scratch);
            if self.remainder.limbs().len() <= target_len {
                self.remainder.add_assign(&self.v);
            }
            swap(&mut self.u, &mut self.v);
            swap(&mut self.v, &mut self.remainder);
            return;
        }
        Division::div_rem_into(
            &self.u,
            &self.v,
            &mut self.quotient,
            &mut self.remainder,
            scratch,
        );
        if self.remainder.limbs().len() <= target_len {
            self.remainder.add_assign(&self.v);
            self.quotient.decrement();
        }
        if self.quotient.is_zero() {
            if track_matrix {
                self.matrix.swap_columns();
            }
            if JACOBI {
                *state = ((*state & 3) << 2) | ((*state >> 2) & 3) | ((*state ^ 32) & 0x30);
            }
            swap(&mut self.u, &mut self.v);
            return;
        }
        if track_matrix {
            self.matrix
                .update_quotient(&mut self.next_matrix, &self.quotient, scratch);
        }
        if JACOBI {
            // SAFETY: zero quotients returned above; normalization proves
            // an initialized low limb and masking bounds it below four.
            let low = unsafe { *self.quotient.limbs().get_unchecked(0) } & 3;
            let index = (*state << 2) | low;
            // SAFETY: six state bits and two quotient bits give index < 256.
            *state = usize::from(unsafe { *Gcd::JACOBI_QUOTIENT.get_unchecked(index) });
        }
        swap(&mut self.u, &mut self.v);
        swap(&mut self.v, &mut self.remainder);
    }

    pub fn ensure_capacity(&mut self, cap: usize) {
        self.matrix.ensure_capacity(cap);
        self.next_matrix.ensure_capacity(cap);
        if self.u.capacity() < cap {
            self.u.reserve(cap.saturating_sub(self.u.limbs().len()));
        }
        if self.v.capacity() < cap {
            self.v.reserve(cap.saturating_sub(self.v.limbs().len()));
        }
        if self.next_u.capacity() < cap {
            self.next_u
                .reserve(cap.saturating_sub(self.next_u.limbs().len()));
        }
        if self.next_v.capacity() < cap {
            self.next_v
                .reserve(cap.saturating_sub(self.next_v.limbs().len()));
        }
        if self.product_a.capacity() < cap {
            self.product_a
                .reserve(cap.saturating_sub(self.product_a.limbs().len()));
        }
        if self.product_b.capacity() < cap {
            self.product_b
                .reserve(cap.saturating_sub(self.product_b.limbs().len()));
        }
        if self.sum_a.capacity() < cap {
            self.sum_a
                .reserve(cap.saturating_sub(self.sum_a.limbs().len()));
        }
        if self.sum_b.capacity() < cap {
            self.sum_b
                .reserve(cap.saturating_sub(self.sum_b.limbs().len()));
        }
        if self.quotient.capacity() < cap {
            self.quotient
                .reserve(cap.saturating_sub(self.quotient.limbs().len()));
        }
        if self.remainder.capacity() < cap {
            self.remainder
                .reserve(cap.saturating_sub(self.remainder.limbs().len()));
        }
    }
}

#[derive(Debug)]
pub struct HgcdLocals {
    pub u: InternalMpUint,
    pub v: InternalMpUint,
    pub next_u: InternalMpUint,
    pub next_v: InternalMpUint,
    pub rem: InternalMpUint,
    pub scratch: DivScratch,
}

impl Default for HgcdLocals {
    fn default() -> Self {
        Self {
            u: InternalMpUint::zero(),
            v: InternalMpUint::zero(),
            next_u: InternalMpUint::zero(),
            next_v: InternalMpUint::zero(),
            rem: InternalMpUint::zero(),
            scratch: DivScratch::default(),
        }
    }
}

/// Reusable ownership for every HGCD recursion level and the outer GCD loop.
#[derive(Debug, Default)]
pub struct HgcdWorkspace {
    pub frames: Vec<HgcdFrame>,
    pub locals: HgcdLocals,
    pub max_prepared_len: usize,
}

impl HgcdWorkspace {
    /// Executes a closure with a reusable thread-local workspace if available,
    /// falling back to a locally allocated workspace for oversized inputs, when
    /// re-entered, when `std` is disabled, or when the slot is already destroyed.
    ///
    /// `operand_len` bounds the original operand widths before any reduction.
    /// An asymmetric quotient step can allocate large locals and division scratch
    /// without preparing recursive frames, so admission uses this input bound as
    /// well as the prepared-frame bound checked before returning to the pool.
    ///
    /// The pooled workspace is taken out of its slot for the duration of the
    /// closure, so re-entering this function observes an empty slot and a live
    /// `RefCell` borrow. The borrow check therefore rejects the inner call
    /// before it can alias the outer workspace's frame buffers.
    ///
    /// During thread teardown, `try_with` falls back to detached storage when
    /// the workspace slot has already been destroyed.
    pub fn with_thread_local<R>(operand_len: usize, f: impl FnOnce(&mut Self) -> R) -> R {
        #[cfg(feature = "std")]
        {
            if operand_len > MAX_POOLED_HGCD_LIMBS {
                return f(&mut Self::default());
            }
            // `try_with` invokes its closure at most once, so the operation is
            // still owned here exactly when an `AccessError` prevented it from
            // running. `Cell` supplies the shared interior mutability that lets
            // both the slot body and the detached fallback consume it.
            let run = Cell::new(Some(f));
            let take_run = || {
                // SAFETY: try_with runs its body once or returns AccessError
                // without running it. The body's borrow-result branches are
                // exclusive; unwinding does not enter the AccessError fallback.
                unsafe { run.take().unwrap_unchecked() }
            };
            THREAD_HGCD_WORKSPACE
                .try_with(|slot| {
                    slot.try_borrow_mut().map_or_else(
                        |_| take_run()(&mut Self::default()),
                        |mut pooled| {
                            let mut workspace = pooled.take().unwrap_or_default();
                            let result = take_run()(&mut workspace);
                            workspace.reset_locals();
                            if workspace.max_prepared_len <= MAX_POOLED_HGCD_LIMBS {
                                *pooled = Some(workspace);
                            }
                            result
                        },
                    )
                })
                .unwrap_or_else(|_| take_run()(&mut Self::default()))
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = operand_len;
            f(&mut Self::default())
        }
    }

    /// Resets the locals without discarding their allocated capacities.
    #[cfg(feature = "std")]
    pub fn reset_locals(&mut self) {
        self.locals.u.clear();
        self.locals.v.clear();
        self.locals.next_u.clear();
        self.locals.next_v.clear();
        self.locals.rem.clear();
    }

    /// Reserves recursion frames for a materialized operand's limb count.
    /// `reduction_len` is bounded by a valid `Limb` slice length.
    pub fn prepare(&mut self, reduction_len: usize) {
        let branchless = reduction_len >= LEHMER_BRANCHLESS_THRESHOLD;
        let mut required = 1_usize;
        let mut recursive_len = reduction_len;
        while recursive_len >= HGCD_BLOCK_THRESHOLD {
            // SAFETY: each iteration halves a positive usize; at most
            // usize::BITS iterations give required <= usize::BITS + 1 <= 65.
            required = unsafe { required.unchecked_add(1) };
            recursive_len >>= 1;
        }
        while self.frames.len() < required {
            self.frames.push(HgcdFrame::default());
        }
        if reduction_len <= self.max_prepared_len {
            for frame in self.frames.iter_mut().take(required) {
                frame.branchless = branchless;
            }
            return;
        }
        self.max_prepared_len = reduction_len;
        for (i, frame) in self.frames.iter_mut().take(required).enumerate() {
            frame.branchless = branchless;
            // reduction_len is a materialized limb count. A valid Limb slice
            // uses at most isize::MAX bytes and limbs have at least two bytes.
            // SAFETY: every shifted count is at most reduction_len; adding
            // four guard limbs fits usize on 16-, 32-, and 64-bit targets.
            let frame_cap = unsafe { (reduction_len >> i).unchecked_add(4) };
            frame.ensure_capacity(frame_cap);
        }
    }
}

/// Maximum original operand or prepared reduction length admitted to the pool.
///
/// The limit is `2^18` limbs where representable. Bounding the original input
/// also bounds the locals and division scratch of asymmetric reductions that
/// finish without preparing HGCD frames. Larger operations use detached storage
/// and leave an existing pooled workspace available for subsequent smaller calls.
#[cfg(feature = "std")]
pub const MAX_POOLED_HGCD_LIMBS: usize = match 1_usize.checked_shl(18) {
    Some(limit) => limit,
    None => usize::MAX,
};

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static THREAD_HGCD_WORKSPACE: RefCell<Option<HgcdWorkspace>> = const { RefCell::new(None) };
}

#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static THREAD_HGCD_WORKSPACE: RefCell<Option<HgcdWorkspace>> = RefCell::from(None);
}
