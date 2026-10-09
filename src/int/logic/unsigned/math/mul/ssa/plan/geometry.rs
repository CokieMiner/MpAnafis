//! Cost-selected Fermat transform dimensions and nested-ring alignment.
//!
//! Candidates satisfy the centered coefficient bound and root-of-unity period.
//! Each operation prices its own forward, pointwise, and inverse work, including
//! recursively planned inner rings.

#![expect(
    unsafe_code,
    reason = "Positive aligned ring widths and admitted exponents prove exact chunk shifts and nonzero fallback geometry"
)]

use core::num::NonZeroUsize;

use super::{CostMemo, LIMB_BITS, SSA_BASE_MODULUS_BITS, SsaOperation, SsaPlan};

/// Candidate offsets from the estimated nested transform exponent.
const NESTED_CENTRE_BUMPS: [u32; 3] = [1, 2, 3];

/// Dimensions derived from the transform exponent and ring width, without
/// scratch accounting.
#[derive(Clone, Copy)]
pub struct Geometry {
    pub transform_len: usize,
    pub transform_log: usize,
    pub chunk_bits: NonZeroUsize,
    pub inner_bits: usize,
    pub twist_step_half: usize,
}

impl Geometry {
    /// Selects the minimum-cost geometry for this ring and operation.
    ///
    /// A ring with insufficient trailing zeros to admit a shrinking inner ring
    /// uses the two-point transform. Every positive limb-aligned ring admits
    /// exponent 1: it divides the width and leaves a positive chunk width and
    /// an even inner ring containing a primitive square root of unity.
    pub fn best_for_operation(modulus_bits: usize, operation: SsaOperation) -> Self {
        debug_assert!(
            modulus_bits >= LIMB_BITS && modulus_bits.is_multiple_of(LIMB_BITS),
            "ring widths are always whole, non-empty limb counts"
        );
        if let Some(geometry) = Self::cached(modulus_bits, operation) {
            return geometry;
        }

        let geometry = SsaPlan::best_exponent(modulus_bits, 0, operation, &mut CostMemo::new())
            .map_or_else(|| Self::two_point(modulus_bits), |(_, geometry)| geometry);
        #[cfg(feature = "std")]
        geometry.cache(modulus_bits, operation);
        geometry
    }

    /// Constructs the two-point fallback directly from limb-aligned dimensions.
    ///
    /// Exponent 1 satisfies the conditions of [`Self::for_exponent_candidates`]
    /// for every positive limb-aligned ring, so construction is infallible.
    fn two_point(modulus_bits: usize) -> Self {
        // SAFETY: the enclosing ring boundary admits modulus_bits>=LIMB_BITS
        // on every target, so its two-point chunk has at least LIMB_BITS/2 bits.
        let chunk_bits = unsafe { NonZeroUsize::new_unchecked(modulus_bits >> 1) };
        // Centered reconstruction needs inner_bits >= 2*chunk_bits + log K + 1.
        // For K = 2 this is modulus_bits + 2; one guard limb covers the bound.
        let inner_bound = modulus_bits.saturating_add(2);
        let aligned = modulus_bits.saturating_add(LIMB_BITS);
        let inner_bits = if aligned <= SSA_BASE_MODULUS_BITS {
            aligned
        } else {
            inner_bound.checked_next_power_of_two().unwrap_or(aligned)
        };
        Self {
            transform_len: 2,
            transform_log: 1,
            chunk_bits,
            inner_bits,
            // 2*inner_bits/K at K = 2.
            twist_step_half: inner_bits,
        }
    }

    /// Derives the geometries for one exponent, or vacant slots when the
    /// exponent cannot produce a usable Fermat transform for this ring.
    ///
    /// `whole_step` aligns the inner ring to a whole transform length instead
    /// of half its length. Both admit valid twists; the cost model balances
    /// the whole-step ring's extra width against half-step sqrt(2) factors.
    ///
    /// Above the basecase width, each nested-alignment bump proposes a fixed
    /// point. Duplicate or unrepresentable widths leave vacant slots.
    pub fn for_exponent_candidates(
        exponent: u32,
        modulus_bits: usize,
        whole_step: bool,
    ) -> [Option<Self>; NESTED_CENTRE_BUMPS.len()] {
        let Some(transform_len) = 1_usize.checked_shl(exponent) else {
            return [None, None, None];
        };
        if transform_len < 2 || !modulus_bits.is_multiple_of(transform_len) {
            return [None, None, None];
        }
        // The admitted exponent is below usize::BITS, so the trailing-zero
        // count is exactly exponent and fits usize on every pointer width.
        #[expect(
            clippy::as_conversions,
            reason = "a usize trailing-zero count is at most usize::BITS and always fits"
        )]
        let transform_log = transform_len.trailing_zeros() as usize;
        // SAFETY: checked_shl above admitted exponent<usize::BITS. The exact
        // quotient needs no modular shift-count mask.
        let Some(chunk_bits) = NonZeroUsize::new(unsafe { modulus_bits.unchecked_shr(exponent) })
        else {
            return [None, None, None];
        };

        // |c_j| < K*2^(2M); one extra bit separates the centered signs.
        let Some(inner_bound) = chunk_bits
            .get()
            .checked_mul(2)
            .and_then(|bound| bound.checked_add(transform_log))
            .and_then(|bound| bound.checked_add(1))
        else {
            return [None, None, None];
        };
        // Half-step alignment admits an odd half-bit twist through sqrt(2).
        // All coefficient storage remains limb-aligned.
        let step_alignment = if whole_step {
            transform_len
        } else {
            transform_len >> 1
        };
        let alignment = step_alignment.max(LIMB_BITS);
        // SAFETY: alignment>=LIMB_BITS>=16 is positive on every target.
        let mask = unsafe { alignment.unchecked_sub(1) };
        let Some(aligned) = inner_bound.checked_add(mask).map(|bound| bound & !mask) else {
            return [None, None, None];
        };

        if aligned <= SSA_BASE_MODULUS_BITS {
            return [
                Self::assemble(transform_len, transform_log, chunk_bits, aligned),
                None,
                None,
            ];
        }
        let mut slots: [Option<Self>; NESTED_CENTRE_BUMPS.len()] = [None, None, None];
        // Fixed points are nondecreasing in bump order, so equal widths are
        // adjacent and can be skipped without dropping a distinct candidate.
        let mut previous = None;
        for (slot, bump) in slots.iter_mut().zip(NESTED_CENTRE_BUMPS) {
            let Some(inner_bits) = nested_ring_bits(aligned, alignment, bump) else {
                continue;
            };
            if previous == Some(inner_bits) {
                continue;
            }
            previous = Some(inner_bits);
            *slot = Self::assemble(transform_len, transform_log, chunk_bits, inner_bits);
        }
        slots
    }

    /// Validates the centered magnitude bound and root-of-unity requirement.
    ///
    /// Two has order `2 * inner_bits`, and `K` divides that order. Half-bit
    /// twist exponents have the representable period `4 * inner_bits`.
    fn assemble(
        transform_len: usize,
        transform_log: usize,
        chunk_bits: NonZeroUsize,
        inner_bits: usize,
    ) -> Option<Self> {
        let magnitude_bits = chunk_bits
            .get()
            .checked_mul(2)?
            .checked_add(transform_log)?;
        if magnitude_bits >= inner_bits {
            return None;
        }
        // Both whole- and half-bit exponents use a representable period.
        let doubled = inner_bits.checked_mul(4)? >> 1;
        if inner_bits < transform_len >> 1 || !doubled.is_multiple_of(transform_len) {
            return None;
        }
        // transform_len >= 2, and the quotient is at least one.
        let twist_step_half = doubled.div_euclid(transform_len);
        Some(Self {
            transform_len,
            transform_log,
            chunk_bits,
            inner_bits,
            twist_step_half,
        })
    }
}

/// Rounds an inner ring to a multiple of its estimated nested transform length.
///
/// The estimate uses [`SsaPlan::search_centre`] plus `centre_bump` to avoid
/// recursively invoking the cost search while generating its candidates. Each
/// adjustment strictly increases the width until a fixed point or overflow.
fn nested_ring_bits(width: usize, alignment: usize, centre_bump: u32) -> Option<usize> {
    let mut width = width;
    loop {
        let nested_log = SsaPlan::search_centre(width).saturating_add(centre_bump);
        // An unrepresentable nested length cannot impose an executable layout.
        let nested_len = 1_usize.checked_shl(nested_log).unwrap_or(1);
        let unit = alignment.max(nested_len).max(LIMB_BITS);
        let mask = unit.checked_sub(1)?;
        let rounded = width.checked_add(mask)? & !mask;
        if rounded == width {
            return Some(width);
        }
        width = rounded;
    }
}
