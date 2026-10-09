//! GCD algorithm namespace and cross-tier dispatch.
//!
//! `Gcd` is the zero-sized namespace every tier implements against. Its inherent
//! methods are spread across the sibling modules, grouped by algorithm:
//!
//! - `dispatch.rs` — Lehmer and recursive half-GCD drivers and forced-tier entries
//! - `binary.rs` — in-register kernels for widths up to two limbs
//! - `lehmer_simulation.rs` — leading-limb extraction and quotient simulation
//! - `lehmer.rs` — full-operand matrix application
//! - `hgcd.rs` — recursive high-half reduction
//! - `reduction.rs` — asymmetric-width application and the quotient-step fallback
//! - `matrix.rs` — transition-matrix state and its exact reduction application
//! - `matrix_update.rs` — sparse transitions and scalar matrix composition
//! - `workspace.rs` — reusable scratch ownership for the recursive reduction
//! - `operations.rs` — the `InternalMpUint` public surface (`gcd`, `lcm`, `is_coprime`, ...)

#![expect(
    unsafe_code,
    reason = "GCD dispatch validates scalar widths and initializes every shifted output limb before committing its write guard"
)]

use core::{
    cmp::{Ordering, min},
    mem::{swap, take},
    num::NonZeroUsize,
};

use super::{
    DivScratch, Division, HGCD_BLOCK_THRESHOLD, HgcdLocals, HgcdWorkspace, InternalMpUint,
    LEHMER_BRANCHLESS_THRESHOLD, LIMB_BITS,
};

/// Namespace for the cross-file GCD algorithm surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gcd;

impl Gcd {
    /// Narrowest divisor for which a Lehmer transition can be simulated.
    ///
    /// Top-limb extraction reads two divisor limbs. This representation
    /// bound is fixed and excluded from the tuning profile.
    pub const LEHMER_MIN_LIMBS: usize = 2;

    /// Computes a GCD through pure Lehmer reduction.
    #[must_use]
    pub fn compute_lehmer(left: &InternalMpUint, right: &InternalMpUint) -> InternalMpUint {
        if let Some(result) = Self::small_gcd(left, right) {
            return result;
        }

        // Strip each operand's own power of two; only their minimum is
        // restored at the end. This drops whole zero limbs that a common
        // shift would copy and avoids a huge first remainder for highly
        // shifted inputs. For ordinary inputs the copy spans are identical.
        let left_shift = left.trailing_zeros();
        let right_shift = right.trailing_zeros();
        let shift = min(left_shift, right_shift);
        let mut locals = HgcdLocals::default();
        clone_shifted(&mut locals.u, left, left_shift);
        clone_shifted(&mut locals.v, right, right_shift);

        finish_lehmer(
            &mut locals.u,
            &mut locals.v,
            &mut locals.next_u,
            &mut locals.next_v,
            &mut locals.rem,
            shift,
            None,
            None,
            left.limbs().len().max(right.limbs().len()) >= LEHMER_BRANCHLESS_THRESHOLD,
            &mut locals.scratch,
        );
        // `locals` is dropped on return; transfer the result owner instead of
        // copying its limbs into a fresh allocation.
        locals.u
    }

    /// Computes a GCD through pure Lehmer reduction with explicit fused-update and wide-simulation policies.
    #[cfg(feature = "_internal-tune")]
    #[must_use]
    pub fn compute_lehmer_configured<const FUSED: bool>(
        left: &InternalMpUint,
        right: &InternalMpUint,
        force_wide: Option<bool>,
    ) -> InternalMpUint {
        if let Some(result) = Self::small_gcd(left, right) {
            return result;
        }

        // Strip each operand's own power of two; only their minimum is
        // restored at the end. This drops whole zero limbs that a common
        // shift would copy and avoids a huge first remainder for highly
        // shifted inputs. For ordinary inputs the copy spans are identical.
        let left_shift = left.trailing_zeros();
        let right_shift = right.trailing_zeros();
        let shift = min(left_shift, right_shift);
        let mut locals = HgcdLocals::default();
        clone_shifted(&mut locals.u, left, left_shift);
        clone_shifted(&mut locals.v, right, right_shift);

        finish_lehmer(
            &mut locals.u,
            &mut locals.v,
            &mut locals.next_u,
            &mut locals.next_v,
            &mut locals.rem,
            shift,
            force_wide,
            Some(FUSED),
            left.limbs().len().max(right.limbs().len()) >= LEHMER_BRANCHLESS_THRESHOLD,
            &mut locals.scratch,
        );
        // `locals` is dropped on return; transfer the result owner instead of
        // copying its limbs into a fresh allocation.
        locals.u
    }

    /// Computes half-GCD with reusable workspace.
    #[must_use]
    pub fn compute_half_gcd(
        left: &InternalMpUint,
        right: &InternalMpUint,
        workspace: &mut HgcdWorkspace,
    ) -> InternalMpUint {
        if let Some(result) = Self::small_gcd(left, right) {
            return result;
        }

        // Strip each operand's own power of two; only their minimum is
        // restored at the end. This drops whole zero limbs that a common
        // shift would copy and avoids a huge first remainder for highly
        // shifted inputs. For ordinary inputs the copy spans are identical.
        let left_shift = left.trailing_zeros();
        let right_shift = right.trailing_zeros();
        let shift = min(left_shift, right_shift);
        let mut locals = take(&mut workspace.locals);
        clone_shifted(&mut locals.u, left, left_shift);
        clone_shifted(&mut locals.v, right, right_shift);

        loop {
            match locals.u.cmp(&locals.v) {
                Ordering::Less => swap(&mut locals.u, &mut locals.v),
                Ordering::Equal => break,
                Ordering::Greater => {}
            }
            if locals.v.is_zero() {
                break;
            }
            if locals.v.limbs().len() < HGCD_BLOCK_THRESHOLD {
                break;
            }

            if locals.u.limbs().len() > locals.v.limbs().len() {
                // A quotient step restores equal widths before recursive
                // high-half work. In particular, an exact multiple requires
                // no transition matrix or HGCD frame allocation.
                if Self::fast_small_div_step(&mut locals.u, &locals.v).is_some() {
                    swap(&mut locals.u, &mut locals.v);
                } else {
                    Division::rem_into(&locals.u, &locals.v, &mut locals.rem, &mut locals.scratch);
                    swap(&mut locals.u, &mut locals.v);
                    swap(&mut locals.v, &mut locals.rem);
                }
                continue;
            }

            let progressed = Self::hgcd_step::<false>(
                &mut locals.u,
                &mut locals.v,
                &mut locals.scratch,
                workspace,
                &mut 0,
            );
            if progressed {
                continue;
            }

            // If HGCD construction makes no progress (often because the quotient
            // is too large for the simulated top limbs), perform a fast linear
            // vector-scalar reduction if q fits in a single limb, or fall back to
            // full multiprecision division if q is multi-limb.
            if Self::fast_small_div_step(&mut locals.u, &locals.v).is_some() {
                swap(&mut locals.u, &mut locals.v);
            } else {
                Division::rem_into(&locals.u, &locals.v, &mut locals.rem, &mut locals.scratch);
                swap(&mut locals.u, &mut locals.v);
                swap(&mut locals.v, &mut locals.rem);
            }
        }

        finish_lehmer(
            &mut locals.u,
            &mut locals.v,
            &mut locals.next_u,
            &mut locals.next_v,
            &mut locals.rem,
            shift,
            None,
            None,
            left.limbs().len().max(right.limbs().len()) >= LEHMER_BRANCHLESS_THRESHOLD,
            &mut locals.scratch,
        );
        let result = InternalMpUint::from_limbs_slice(locals.u.limbs());
        locals.u.clear();
        locals.v.clear();
        locals.next_u.clear();
        locals.next_v.clear();
        locals.rem.clear();
        workspace.locals = locals;
        result
    }
}

/// Copies only the significant suffix before removing the residual bit shift.
/// The caller supplies a factor of two of the nonzero operand, so the
/// discarded whole limbs are zero and at least one source limb remains.
///
/// The word skip fuses into the copy and the residual bit shift fuses into the
/// same pass, so each operand pays one read/write pass instead of copy plus shift.
fn clone_shifted(out: &mut InternalMpUint, value: &InternalMpUint, shift: usize) {
    let whole_limbs = shift >> LIMB_BITS.trailing_zeros();
    // SAFETY: callers pass trailing_zeros() of a nonzero normalized operand.
    // The first nonzero limb remains after whole_limbs zero limbs are skipped.
    let significant = unsafe { value.limbs().get_unchecked(whole_limbs..) };
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "bit remainder is below LIMB_BITS <= 64 and always fits in u32"
    )]
    let bit_shift = (shift & LIMB_BITS.wrapping_sub(1)) as u32;
    if bit_shift == 0 {
        out.clone_from_slice(significant);
        return;
    }
    let src_len = significant.len();
    let mut write = out.prepare_limb_write(src_len);
    let dst_ptr = write.as_mut_ptr();
    // SAFETY: the retained first nonzero limb proves src_len>0. The masked
    // bit_shift is in 1..Limb::BITS after the zero-shift return. The write guard
    // reserves src_len limbs disjoint from the initialized source; the loop
    // and last-digit write initialize that complete span before commit.
    unsafe {
        let src_ptr = significant.as_ptr();
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is 16, 32, or 64 and always fits in u32"
        )]
        let r_shift = (LIMB_BITS as u32).unchecked_sub(bit_shift);
        let last = src_len.unchecked_sub(1);
        for i in 0..last {
            let lo = *src_ptr.add(i);
            let hi = *src_ptr.add(i.unchecked_add(1));
            *dst_ptr.add(i) = lo.unchecked_shr(bit_shift) | hi.unchecked_shl(r_shift);
        }
        *dst_ptr.add(last) = (*src_ptr.add(last)).unchecked_shr(bit_shift);
        let _ = write.commit();
    }
    out.normalize();
}

#[expect(
    clippy::too_many_arguments,
    reason = "The reusable scratch buffers coordinate vector updates, quotient calculations, and remainder persistence across iterations without heap reallocation"
)]
fn finish_lehmer(
    u: &mut InternalMpUint,
    v: &mut InternalMpUint,
    next_u: &mut InternalMpUint,
    next_v: &mut InternalMpUint,
    rem: &mut InternalMpUint,
    shift: usize,
    force_wide: Option<bool>,
    force_fused: Option<bool>,
    branchless: bool,
    scratch: &mut DivScratch,
) {
    loop {
        match (*u).cmp(v) {
            Ordering::Less => swap(u, v),
            Ordering::Equal => break,
            Ordering::Greater => {}
        }
        if v.is_zero() {
            break;
        }
        if v.limbs().len() <= 2 {
            // Writes the leaf directly into the reused `u` owner and shifts in
            // place, so the final rescaling needs neither a result temporary nor
            // a copy back into `u`.
            if v.limbs().len() == 1 && u.limbs().len() > 1 {
                // SAFETY: v.limbs().len() == 1 proves index 0 exists.
                let mut v0 = unsafe { *v.limbs().get_unchecked(0) };
                // Entry removes each operand's power of two. Unimodular
                // transitions and remainder steps preserve their odd GCD,
                // so any power of two in v cannot also divide u.
                v0 >>= v0.trailing_zeros();
                // SAFETY: the zero case returned above. Removing trailing
                // zero bits from the nonzero scalar leaves a positive divisor.
                let divisor = unsafe { NonZeroUsize::new_unchecked(v0) };
                let g = Gcd::gcd_odd_limb(u.limbs(), divisor);
                u.set_limb(g);
            } else if u.limbs().len() <= 2 {
                let leaf = Gcd::gcd_leaf_pair(u, v);
                u.clone_from(&leaf);
            } else {
                Division::rem_into(u, v, rem, scratch);
                let leaf = Gcd::gcd_leaf_pair(v, rem);
                u.clone_from(&leaf);
            }
            if shift > 0 {
                u.shl_assign(shift);
            }
            return;
        }

        if force_wide.is_none()
            && u.limbs().len().wrapping_sub(v.limbs().len()) <= 1
            && let Some((u0, v0, u1, v1)) = Gcd::hgcd2(u.limbs(), v.limbs())
            && Gcd::lehmer_update_dispatched(
                u,
                v,
                next_u,
                next_v,
                u0,
                v0,
                u1,
                v1,
                true,
                force_fused,
            )
        {
            continue;
        }

        let (u0, v0, u1, v1, even) =
            Gcd::simulate_step::<false>(u.limbs(), v.limbs(), force_wide, branchless, |_| {});

        let is_identity = u0 == 1 && v0 == 0 && u1 == 0 && v1 == 1;
        if is_identity
            || !Gcd::lehmer_update_dispatched(
                u,
                v,
                next_u,
                next_v,
                u0,
                v0,
                u1,
                v1,
                even,
                force_fused,
            )
        {
            // As in the half-GCD driver above: failed simulation means the
            // quotient exceeds what the top limbs determine, so take a fast
            // single-limb step when q fits one limb, else a full remainder.
            if Gcd::fast_small_div_step(u, v).is_some() {
                swap(u, v);
            } else {
                Division::rem_into(u, v, rem, scratch);
                swap(u, v);
                swap(v, rem);
            }
        }
    }

    if shift > 0 {
        u.shl_assign(shift);
    }
}
