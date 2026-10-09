//! Recursive high-half reduction for the production half-GCD tier.
//!
//! Reference: N. Möller, "On Schönhage's algorithm and subquadratic integer
//! GCD computation", Mathematics of Computation 77(261), 589-607, 2008,
//! Sections 5-6. DOI: 10.1090/S0025-5718-07-02017-0.

#![expect(
    unsafe_code,
    reason = "prepared recursion frames and validated high-window partitions establish initialized frame and operand bounds"
)]

use core::{cmp::Ordering, mem::swap};

use super::{
    DivScratch, Gcd, HGCD_BLOCK_THRESHOLD, HgcdFrame, HgcdMatrix, HgcdWorkspace, InternalMpUint,
    Limb,
};

impl Gcd {
    /// Reduces a balanced pair while retaining the exact accepted transition.
    ///
    /// Unlike [`hgcd_block`], the reduced pair is never sorted without
    /// reflection: an ordering exchange swaps the transition columns instead,
    /// so the returned matrix and pair always describe the same committed
    /// transformation. The reduction runs on tracked frames with full
    /// quotient retention, never the root remainder-only shortcut. The
    /// coefficient-tracking caller (extended GCD) folds the transition into
    /// its cofactors once per block instead of per Lehmer batch.
    pub fn hgcd_block_matrix(
        u: &mut InternalMpUint,
        v: &mut InternalMpUint,
        matrix: &mut HgcdMatrix,
        scratch: &mut DivScratch,
        workspace: &mut HgcdWorkspace,
    ) -> bool {
        let n = u.limbs().len();
        let m = v.limbs().len();
        if n < HGCD_BLOCK_THRESHOLD
            || m < HGCD_BLOCK_THRESHOLD
            || n.abs_diff(m) > 1
            || (*u).cmp(v) != Ordering::Greater
        {
            return false;
        }
        workspace.prepare(n);
        // SAFETY: halving a usize leaves room for one on every target.
        let target_len = unsafe { (n >> 1).unchecked_add(1) };
        if !reduce_recursive::<false>(
            u.limbs(),
            v.limbs(),
            target_len,
            scratch,
            &mut workspace.frames,
            false,
            &mut 0,
        ) {
            return false;
        }
        // SAFETY: prepare(n) ensured a nonempty frame slice and successful
        // recursion wrote its reduced pair into frames[0].
        let reduced = unsafe { workspace.frames.first_mut().unwrap_unchecked() };
        if reduced.u.cmp(&reduced.v) == Ordering::Less {
            swap(&mut reduced.u, &mut reduced.v);
            reduced.matrix.swap_columns();
        }
        // Same progress gate as hgcd_block: the smaller operand must shrink,
        // otherwise the caller would loop on an unchanged pair.
        if reduced.v.cmp(v) != Ordering::Less {
            return false;
        }
        swap(u, &mut reduced.u);
        swap(v, &mut reduced.v);
        swap(matrix, &mut reduced.matrix);
        true
    }

    /// Performs one subquadratic HGCD reduction step on `(u, v)` using Möller's
    /// `p = 2m/3` partitioning (`≈2n/3` since the `n−m ≤ 1` guard holds).
    ///
    /// Runs recursive HGCD on the upper `n-p` limbs, approximately `n/3`, constructing the
    /// transition matrix `M` and reduced high parts, then adjusts the full operands
    /// in place.
    #[expect(
        clippy::many_single_char_names,
        reason = "u, v denote the operand pair, n, m their limb lengths, and p Moller's partition point"
    )]
    pub fn hgcd_step<const JACOBI: bool>(
        u: &mut InternalMpUint,
        v: &mut InternalMpUint,
        scratch: &mut DivScratch,
        workspace: &mut HgcdWorkspace,
        state: &mut usize,
    ) -> bool {
        let n = u.limbs().len();
        let m = v.limbs().len();
        if n < HGCD_BLOCK_THRESHOLD
            || m < HGCD_BLOCK_THRESHOLD
            || n.abs_diff(m) > 1
            || (*u).cmp(v) != Ordering::Greater
        {
            return false;
        }

        // SAFETY: an allocated Limb slice occupies at most isize::MAX
        // bytes, with at least two bytes per limb; 2*m fits usize.
        let p = unsafe { m.unchecked_mul(2) }.div_euclid(3);
        // SAFETY: p=floor(2*m/3) <= m.
        let high_len = unsafe { m.unchecked_sub(p) };
        if high_len < HGCD_BLOCK_THRESHOLD {
            return hgcd_block::<JACOBI>(u, v, scratch, workspace, state);
        }

        // SAFETY: p = 2m/3 < m <= n, so p is strictly within the u slice.
        let u_high = unsafe { u.limbs().get_unchecked(p..) };
        // SAFETY: p = 2m/3 < m <= n, so p is strictly within the v slice.
        let v_high = unsafe { v.limbs().get_unchecked(p..) };
        if u_high == v_high {
            // Equal high windows cannot certify a positive remainder: their
            // first subtraction is zero. Let the caller subtract the full
            // operands before preparing any recursive frame storage.
            return false;
        }
        workspace.prepare(n);
        // SAFETY: halving a slice length leaves room for one.
        let target_len = unsafe { (u_high.len() >> 1).unchecked_add(1) };

        let mut candidate = *state;
        let generated = reduce_recursive::<JACOBI>(
            u_high,
            v_high,
            target_len,
            scratch,
            &mut workspace.frames,
            false,
            &mut candidate,
        );
        if !generated {
            return false;
        }

        // SAFETY: prepare guaranteed frames is non-empty.
        let (frame0, _) = unsafe { workspace.frames.split_first_mut().unwrap_unchecked() };
        if frame0.matrix.is_identity() {
            return false;
        }

        let applied = frame0.matrix.adjust_vector_into(
            u.limbs(),
            v.limbs(),
            p,
            frame0.u.limbs(),
            frame0.v.limbs(),
            &mut frame0.next_u,
            &mut frame0.next_v,
            &mut frame0.product_a,
            &mut frame0.product_b,
            &mut frame0.sum_a,
            &mut frame0.sum_b,
            scratch,
        );
        if applied {
            swap(u, &mut frame0.next_u);
            swap(v, &mut frame0.next_v);
            if JACOBI {
                *state = candidate;
            }
        }
        applied
    }
}

/// Reduces a balanced pair through remainder-only root frames.
pub fn hgcd_block<const JACOBI: bool>(
    u: &mut InternalMpUint,
    v: &mut InternalMpUint,
    scratch: &mut DivScratch,
    workspace: &mut HgcdWorkspace,
    state: &mut usize,
) -> bool {
    let n = u.limbs().len();
    let m = v.limbs().len();
    if n < HGCD_BLOCK_THRESHOLD
        || m < HGCD_BLOCK_THRESHOLD
        || n.abs_diff(m) > 1
        || (*u).cmp(v) != Ordering::Greater
    {
        return false;
    }
    workspace.prepare(n);
    // SAFETY: halving a usize leaves room for one on every target.
    let target_len = unsafe { (n >> 1).unchecked_add(1) };
    let mut candidate = *state;
    let generated = reduce_recursive::<JACOBI>(
        u.limbs(),
        v.limbs(),
        target_len,
        scratch,
        &mut workspace.frames,
        true,
        &mut candidate,
    );
    if !generated {
        return false;
    }

    // SAFETY: `prepare(n)` ensures `frames` has at least one frame,
    // and successful recursion wrote its reduced pair into `frames[0]`.
    let reduced = unsafe { workspace.frames.first_mut().unwrap_unchecked() };
    if reduced.u.cmp(&reduced.v) == Ordering::Less {
        swap(&mut reduced.u, &mut reduced.v);
        if JACOBI {
            candidate = ((candidate & 3) << 2) | ((candidate >> 2) & 3) | ((candidate ^ 32) & 0x30);
        }
    }
    if reduced.v.cmp(v) != Ordering::Less {
        return false;
    }
    swap(u, &mut reduced.u);
    swap(v, &mut reduced.v);
    if JACOBI {
        *state = candidate;
    }
    true
}

fn reduce_recursive<const JACOBI: bool>(
    u_limbs: &[Limb],
    v_limbs: &[Limb],
    target_len: usize,
    scratch: &mut DivScratch,
    frames: &mut [HgcdFrame],
    is_root: bool,
    state: &mut usize,
) -> bool {
    // SAFETY: the top-level workspace preparation and recursive `can_recurse`
    // guard prove every call receives at least one frame.
    let (frame, deeper) = unsafe { frames.split_first_mut().unwrap_unchecked() };

    if v_limbs.len() <= target_len
        || InternalMpUint::cmp_limbs(u_limbs, v_limbs) != Ordering::Greater
    {
        return false;
    }

    frame.matrix.reset();
    let mut progressed = false;

    let input_len = u_limbs.len();
    let can_recurse = input_len >= HGCD_BLOCK_THRESHOLD && !deeper.is_empty();

    if can_recurse {
        let low_len = input_len >> 1;
        // SAFETY: each call sets target_len=floor(input_len/2)+1, and the
        // early return leaves v_limbs.len()>target_len. Both suffixes exist.
        let (u_high, v_high) = unsafe {
            (
                u_limbs.get_unchecked(low_len..),
                v_limbs.get_unchecked(low_len..),
            )
        };
        // SAFETY: halving a slice length leaves room for one.
        let child_target = unsafe { (u_high.len() >> 1).unchecked_add(1) };
        let mut candidate = *state;
        if reduce_recursive::<JACOBI>(
            u_high,
            v_high,
            child_target,
            scratch,
            deeper,
            false,
            &mut candidate,
        ) {
            // SAFETY: successful recursion wrote its result into `deeper[0]`.
            let child = unsafe { deeper.first_mut().unwrap_unchecked() };
            // Reconstruct straight from the borrowed inputs into the owned
            // destinations. The entry copies are deferred: they are only
            // needed if this application is rejected below.
            let applied = child.matrix.adjust_vector_into(
                u_limbs,
                v_limbs,
                low_len,
                child.u.limbs(),
                child.v.limbs(),
                &mut frame.next_u,
                &mut frame.next_v,
                &mut frame.product_a,
                &mut frame.product_b,
                &mut frame.sum_a,
                &mut frame.sum_b,
                scratch,
            );
            if applied {
                // The frame matrix is still the identity, so ownership can be
                // transferred without multiplying by identity entries. At the
                // root the matrix is never consumed, so skip the transfer and
                // keep the child's buffers with their useful capacities.
                if !is_root {
                    swap(&mut frame.matrix, &mut child.matrix);
                }
                swap(&mut frame.u, &mut frame.next_u);
                swap(&mut frame.v, &mut frame.next_v);
                progressed = true;
                if JACOBI {
                    *state = candidate;
                }
            } else {
                // A rejected application falls back to owned copies; the
                // narrowing loop below makes progress with ordinary steps.
                frame.u.clone_from_slice(u_limbs);
                frame.v.clone_from_slice(v_limbs);
            }
        } else {
            frame.u.clone_from_slice(u_limbs);
            frame.v.clone_from_slice(v_limbs);
        }

        // The second recursive window follows reduction to floor(3*n/4)+1.
        // SAFETY: input_len >= HGCD_BLOCK_THRESHOLD > 0; ceil(n/4) is
        // in [1,n], so n-ceil(n/4)+1 is in [1,n].
        let three_quarter = unsafe {
            input_len
                .unchecked_sub(input_len.div_ceil(4))
                .unchecked_add(1)
        };
        while frame.u.limbs().len() > three_quarter && frame.v.limbs().len() > target_len {
            frame.reduce_step::<JACOBI>(target_len, scratch, !is_root, state);
            progressed = true;
        }

        if reduce_second_window::<JACOBI>(frame, deeper, target_len, scratch, is_root, state) {
            progressed = true;
        }
    } else {
        frame.u.clone_from_slice(u_limbs);
        frame.v.clone_from_slice(v_limbs);
        while !frame.v.is_zero() && frame.v.limbs().len() > target_len {
            frame.reduce_step::<JACOBI>(target_len, scratch, !is_root, state);
            progressed = true;
        }
    }

    progressed && (is_root || !frame.matrix.is_identity())
}

fn reduce_second_window<const JACOBI: bool>(
    frame: &mut HgcdFrame,
    deeper: &mut [HgcdFrame],
    target_len: usize,
    scratch: &mut DivScratch,
    is_root: bool,
    state: &mut usize,
) -> bool {
    let current_len = frame.u.limbs().len();
    // SAFETY: target_len is a halved materialized limb count plus one;
    // the slice byte bound leaves room for two further guard limbs.
    if current_len <= unsafe { target_len.unchecked_add(2) } {
        return false;
    }
    let Some(second_low) = target_len
        .checked_mul(2)
        .and_then(|value| value.checked_sub(current_len))
        .and_then(|value| value.checked_add(1))
    else {
        return false;
    };
    if second_low >= frame.v.limbs().len() {
        return false;
    }

    // SAFETY: current_len>target_len+2 and second_low=2*target_len-current_len+1
    // give second_low<=target_len-2<current_len. The guard also bounds it by v.
    let (u_second, v_second) = unsafe {
        (
            frame.u.limbs().get_unchecked(second_low..),
            frame.v.limbs().get_unchecked(second_low..),
        )
    };
    // SAFETY: halving a slice length leaves room for one.
    let second_target = unsafe { (u_second.len() >> 1).unchecked_add(1) };
    let mut candidate = *state;
    let child_ready = reduce_recursive::<JACOBI>(
        u_second,
        v_second,
        second_target,
        scratch,
        deeper,
        false,
        &mut candidate,
    );
    if !child_ready {
        return false;
    }

    // SAFETY: the successful child recursion wrote its result into `deeper[0]`.
    let child = unsafe { deeper.first_mut().unwrap_unchecked() };
    let applied = child.matrix.adjust_vector_into(
        frame.u.limbs(),
        frame.v.limbs(),
        second_low,
        child.u.limbs(),
        child.v.limbs(),
        &mut frame.next_u,
        &mut frame.next_v,
        &mut frame.product_a,
        &mut frame.product_b,
        &mut frame.sum_a,
        &mut frame.sum_b,
        scratch,
    );
    // A negative reconstruction means the second window's quotient sequence is
    // not valid for the full pair; the first window's reduction stands and the
    // caller continues from there.
    if !applied {
        return false;
    }
    swap(&mut frame.u, &mut frame.next_u);
    swap(&mut frame.v, &mut frame.next_v);
    if JACOBI {
        *state = candidate;
    }

    // The root matrix is never consumed (`hgcd_block` keeps only the reduced
    // pair), so composing it would spend a full matrix product on dead state.
    if !is_root {
        frame.matrix.multiply_right(
            &child.matrix,
            &mut frame.next_matrix,
            &mut frame.product_a,
            &mut frame.product_b,
            scratch,
        );
    }
    true
}
