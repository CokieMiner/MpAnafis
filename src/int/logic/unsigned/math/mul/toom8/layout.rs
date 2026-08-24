//! Shape selection and scratch layout for Toom-8 and Toom-8.5.

use core::cmp::{max, min};
#[cfg(feature = "std")]
use std::sync::RwLock;

#[cfg(feature = "std")]
use alloc::vec::Vec;

use super::{
    BALANCED_PARTS, EVALUATION_GUARD_BITS, HALF_LARGE_PARTS, HALF_SMALL_PARTS,
    INTERPOLATION_GUARD_BITS, KARATSUBA_THRESHOLD, LIMB_BITS, Limb, MulShape, Multiplication,
    Recursive, SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_4_THRESHOLD, SQR_TOOM_COOK_6_THRESHOLD,
    SQR_TOOM_COOK_85_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, TOOM_COOK_4_THRESHOLD,
    TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD, TierCeiling, Toom8,
};

/// Longest child length whose scratch demand is scanned exhaustively.
///
/// A child product's scratch demand rises with its operand length inside a
/// single tier and only steps where a length crosses a dispatch threshold or a
/// shape predicate into a tier with a different layout. Those steps are
/// confined to short lengths: past this bound the demand is monotone, so the
/// top of a range is its maximum and the range below it needs no scan at all.
///
/// The floor is measured — the last observed non-monotone length is just under
/// 1200 — and the multiple of the tuned thresholds keeps the bound valid if
/// they are retuned upwards. `max_balanced_mul_scratch_below` debug-asserts the
/// resulting bound against a full scan.
const EXHAUSTIVE_CHILD_SCAN: usize = const_max(2048, 4 * max_tuned_threshold());

/// Stable const equivalent of [`max`].
const fn const_max(left: usize, right: usize) -> usize {
    if left > right { left } else { right }
}

const fn max_tuned_threshold() -> usize {
    let thresholds = [
        KARATSUBA_THRESHOLD,
        TOOM_COOK_THRESHOLD,
        TOOM_COOK_4_THRESHOLD,
        TOOM_COOK_6_THRESHOLD,
        TOOM_COOK_85_THRESHOLD,
        SQR_KARATSUBA_THRESHOLD,
        SQR_TOOM_COOK_THRESHOLD,
        SQR_TOOM_COOK_4_THRESHOLD,
        SQR_TOOM_COOK_6_THRESHOLD,
        SQR_TOOM_COOK_85_THRESHOLD,
    ];
    let mut largest = 0;
    let mut remaining: &[usize] = &thresholds;
    while let [first, rest @ ..] = remaining {
        if *first != usize::MAX && *first != usize::MAX - 1 {
            largest = const_max(largest, *first);
        }
        remaining = rest;
    }
    largest
}

pub struct MulScratchLayout<'buffer> {
    pub one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub four: &'buffer mut [Limb],
    pub eight: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub quarter: &'buffer mut [Limb],
    pub eighth: &'buffer mut [Limb],
    pub temporary: &'buffer mut [Limb],
    pub eval_a: &'buffer mut [Limb],
    pub eval_b: &'buffer mut [Limb],
    pub odd_a: &'buffer mut [Limb],
    pub odd_b: &'buffer mut [Limb],
    pub inner: &'buffer mut [Limb],
}

pub struct SqrScratchLayout<'buffer> {
    pub one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub four: &'buffer mut [Limb],
    pub eight: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub quarter: &'buffer mut [Limb],
    pub eighth: &'buffer mut [Limb],
    pub temporary: &'buffer mut [Limb],
    pub eval: &'buffer mut [Limb],
    pub odd: &'buffer mut [Limb],
    pub inner: &'buffer mut [Limb],
}

pub struct DestinationPoints<'buffer> {
    pub zero: &'buffer [Limb],
    pub half: &'buffer mut [Limb],
    pub one: &'buffer mut [Limb],
    pub four: &'buffer mut [Limb],
    pub infinity: &'buffer [Limb],
}

/// Snapshot table of one child-demand family's prefix maxima.
///
/// The probes behind these tables are pure functions of the generated
/// thresholds, so a computed maximum never changes. The table grows in fixed
/// chunks: a product at a new width pays only for the prefix it needs, and every
/// later product at any smaller width is one read-locked indexed lookup. One
/// lock owns both the values and their published length, so concurrent growth
/// cannot expose a mismatched pointer/length pair.
#[cfg(feature = "std")]
struct PrefixTable {
    values: RwLock<Vec<usize>>,
}

#[cfg(feature = "std")]
static MUL_PREFIX: PrefixTable = PrefixTable::empty();
#[cfg(feature = "std")]
static SQR_PREFIX: PrefixTable = PrefixTable::empty();

/// Discriminant for the two demand families without `std`, where
/// [`child_scratch_prefix`] recomputes the scan instead.
#[cfg(not(feature = "std"))]
struct PrefixFamily;

#[cfg(not(feature = "std"))]
const MUL_PREFIX: PrefixFamily = PrefixFamily;
#[cfg(not(feature = "std"))]
const SQR_PREFIX: PrefixFamily = PrefixFamily;

/// Table-or-family selector resolved by target capability.
#[cfg(feature = "std")]
type PrefixTableOrFamily = PrefixTable;
#[cfg(not(feature = "std"))]
type PrefixTableOrFamily = PrefixFamily;

impl Toom8 {
    pub const fn split_mul_scratch(
        scratch: &mut [Limb],
        split_len: usize,
        value_len: usize,
        eval_len: usize,
        points_are_placed: bool,
    ) -> MulScratchLayout<'_> {
        let packed_len = value_len.wrapping_add(split_len);
        let scratch_point_len = if points_are_placed { 0 } else { packed_len };
        let (one, after_one) = scratch.split_at_mut(scratch_point_len);
        let (two, after_two) = after_one.split_at_mut(packed_len);
        let (four, after_four) = after_two.split_at_mut(scratch_point_len);
        let (eight, after_eight) = after_four.split_at_mut(packed_len);
        let (half, after_half) = after_eight.split_at_mut(scratch_point_len);
        let (quarter, after_quarter) = after_half.split_at_mut(packed_len);
        let (eighth, after_eighth) = after_quarter.split_at_mut(packed_len);
        let (temporary, after_temporary) = after_eighth.split_at_mut(packed_len);
        let (eval_a, after_eval_a) = after_temporary.split_at_mut(eval_len);
        let (eval_b, after_eval_b) = after_eval_a.split_at_mut(eval_len);
        let (odd_a, after_odd_a) = after_eval_b.split_at_mut(eval_len);
        let (odd_b, inner) = after_odd_a.split_at_mut(eval_len);
        MulScratchLayout {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
            temporary,
            eval_a,
            eval_b,
            odd_a,
            odd_b,
            inner,
        }
    }

    pub const fn split_sqr_scratch(
        scratch: &mut [Limb],
        split_len: usize,
        value_len: usize,
        eval_len: usize,
        points_are_placed: bool,
    ) -> SqrScratchLayout<'_> {
        let packed_len = value_len.wrapping_add(split_len);
        let scratch_point_len = if points_are_placed { 0 } else { packed_len };
        let (one, after_one) = scratch.split_at_mut(scratch_point_len);
        let (two, after_two) = after_one.split_at_mut(packed_len);
        let (four, after_four) = after_two.split_at_mut(scratch_point_len);
        let (eight, after_eight) = after_four.split_at_mut(packed_len);
        let (half, after_half) = after_eight.split_at_mut(scratch_point_len);
        let (quarter, after_quarter) = after_half.split_at_mut(packed_len);
        let (eighth, after_eighth) = after_quarter.split_at_mut(packed_len);
        let (temporary, after_temporary) = after_eighth.split_at_mut(packed_len);
        let (eval, after_eval) = after_temporary.split_at_mut(eval_len);
        let (odd, inner) = after_eval.split_at_mut(eval_len);
        SqrScratchLayout {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
            temporary,
            eval,
            odd,
            inner,
        }
    }

    pub fn multiplication_split_len(shape: MulShape, len_a: usize, len_b: usize) -> usize {
        let smaller = min(len_a, len_b);
        let larger = max(len_a, len_b);
        match shape {
            MulShape::Balanced => larger.div_ceil(BALANCED_PARTS),
            MulShape::Half => max(
                larger.div_ceil(HALF_LARGE_PARTS),
                smaller.div_ceil(HALF_SMALL_PARTS),
            ),
        }
    }

    pub const fn multiplication_degree(shape: MulShape) -> usize {
        match shape {
            MulShape::Balanced => 14,
            MulShape::Half => 15,
        }
    }

    pub fn local_mul_scratch_len(
        shape: MulShape,
        split_len: usize,
        len_a: usize,
        len_b: usize,
    ) -> usize {
        let eval_len = Self::evaluation_len(split_len);
        let value_len = Self::interpolation_value_len(split_len);
        let mut inner = max_recursive_mul_scratch(split_len, eval_len);
        if matches!(shape, MulShape::Half) {
            let larger = max(len_a, len_b);
            let smaller = min(len_a, len_b);
            let high_large = larger.saturating_sub(split_len.wrapping_mul(8));
            let high_small = smaller.saturating_sub(split_len.wrapping_mul(7));
            let plan = Multiplication::select_plan(high_large, high_small, TierCeiling::Toom6);
            inner = max(
                inner,
                Multiplication::scratch_len(plan, high_large, high_small),
            );
        }
        let packed_len = value_len.wrapping_add(split_len);
        let infinity_len = Self::multiplication_infinity_len(shape, split_len, len_a, len_b);
        let points_are_placed = Self::destination_points_fit(
            len_a.wrapping_add(len_b),
            split_len,
            packed_len,
            infinity_len,
        );
        let point_buffers = if points_are_placed { 5 } else { 8 };
        packed_len
            .wrapping_mul(point_buffers)
            .wrapping_add(eval_len.wrapping_mul(4))
            .wrapping_add(inner)
    }

    pub fn local_sqr_scratch_len(len: usize) -> usize {
        let split_len = len.div_ceil(BALANCED_PARTS);
        let eval_len = Self::evaluation_len(split_len);
        let value_len = Self::interpolation_value_len(split_len);
        let inner = max_sqr_scratch_up_to(eval_len);
        let packed_len = value_len.wrapping_add(split_len);
        let points_are_placed =
            Self::destination_points_fit(len.wrapping_mul(2), split_len, packed_len, 0);
        let point_buffers = if points_are_placed { 5 } else { 8 };
        packed_len
            .wrapping_mul(point_buffers)
            .wrapping_add(eval_len.wrapping_mul(2))
            .wrapping_add(inner)
    }

    pub fn multiplication_infinity_len(
        shape: MulShape,
        split_len: usize,
        len_a: usize,
        len_b: usize,
    ) -> usize {
        if matches!(shape, MulShape::Balanced) {
            return 0;
        }
        let larger = max(len_a, len_b);
        let smaller = min(len_a, len_b);
        larger
            .saturating_sub(split_len.wrapping_mul(HALF_LARGE_PARTS - 1))
            .wrapping_add(smaller.saturating_sub(split_len.wrapping_mul(HALF_SMALL_PARTS - 1)))
    }

    pub const fn evaluation_len(split_len: usize) -> usize {
        split_len.wrapping_add(EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS))
    }

    pub const fn interpolation_value_len(split_len: usize) -> usize {
        // Direct and reciprocal point products are below 2^50*B^(2m). The exact
        // interpolation schedule's longest undivided elimination chain stays below
        // 2^80*B^(2m): the symmetric x=8 row is the limiting case. Thus 96 guard
        // bits leave sixteen proof-margin bits while also providing at least the
        // two extra limbs required by a full recursive product on 64-bit targets.
        split_len
            .wrapping_mul(2)
            .wrapping_add(INTERPOLATION_GUARD_BITS.div_ceil(LIMB_BITS))
    }

    pub const fn split_destination_points(
        dst: &mut [Limb],
        split_len: usize,
        packed_len: usize,
        infinity_len: usize,
    ) -> Option<DestinationPoints<'_>> {
        let zero_len = split_len.wrapping_mul(2);
        let half_offset = split_len.wrapping_mul(3);
        let one_offset = split_len.wrapping_mul(7);
        let four_offset = split_len.wrapping_mul(11);
        let infinity_offset = split_len.wrapping_mul(15);
        let after_half_offset = half_offset.wrapping_add(packed_len);
        let after_one_offset = one_offset.wrapping_add(packed_len);
        let placed_end = four_offset.wrapping_add(packed_len);
        if !Self::destination_points_fit(dst.len(), split_len, packed_len, infinity_len) {
            return None;
        }

        let (before_half, half_and_after) = dst.split_at_mut(half_offset);
        let (half, after_half) = half_and_after.split_at_mut(packed_len);
        let gap_before_one = one_offset.wrapping_sub(after_half_offset);
        let (_, one_and_after) = after_half.split_at_mut(gap_before_one);
        let (one, after_one) = one_and_after.split_at_mut(packed_len);
        let gap_before_four = four_offset.wrapping_sub(after_one_offset);
        let (_, four_and_after) = after_one.split_at_mut(gap_before_four);
        let (four, after_four) = four_and_after.split_at_mut(packed_len);
        let (zero, _) = before_half.split_at(zero_len);
        let infinity = if infinity_len == 0 {
            &[]
        } else {
            let gap_before_infinity = infinity_offset.wrapping_sub(placed_end);
            let (_, infinity_and_after) = after_four.split_at(gap_before_infinity);
            let (infinity, _) = infinity_and_after.split_at(infinity_len);
            infinity
        };
        Some(DestinationPoints {
            zero,
            half,
            one,
            four,
            infinity,
        })
    }

    pub const fn destination_points_fit(
        product_len: usize,
        split_len: usize,
        packed_len: usize,
        infinity_len: usize,
    ) -> bool {
        let zero_len = split_len.wrapping_mul(2);
        let half_offset = split_len.wrapping_mul(3);
        let one_offset = split_len.wrapping_mul(7);
        let four_offset = split_len.wrapping_mul(11);
        let infinity_offset = split_len.wrapping_mul(15);
        let after_half_offset = half_offset.wrapping_add(packed_len);
        let after_one_offset = one_offset.wrapping_add(packed_len);
        let placed_end = four_offset.wrapping_add(packed_len);
        zero_len <= half_offset
            && after_half_offset <= one_offset
            && after_one_offset <= four_offset
            && placed_end <= product_len
            && (infinity_len == 0
                || (placed_end <= infinity_offset
                    && infinity_offset.wrapping_add(infinity_len) <= product_len))
    }

    pub fn clear_destination_gaps(
        dst: &mut [Limb],
        split_len: usize,
        packed_len: usize,
        zero_product_len: usize,
        infinity_len: usize,
        points_are_placed: bool,
    ) {
        let infinity_offset = split_len.wrapping_mul(15);
        let after_infinity = infinity_offset.wrapping_add(infinity_len);
        if !points_are_placed {
            let middle_end = if infinity_len == 0 {
                dst.len()
            } else {
                infinity_offset
            };
            clear_range(dst, zero_product_len, middle_end);
            if infinity_len != 0 {
                clear_range(dst, after_infinity, dst.len());
            }
            return;
        }

        // The point buffers at 3m, 7m, and 11m overwrite their complete packed
        // ranges. Initialize only the disjoint gaps that coefficient additions can
        // observe, preserving the endpoint products at shifts zero and fifteen.
        let half_offset = split_len.wrapping_mul(3);
        let one_offset = split_len.wrapping_mul(7);
        let four_offset = split_len.wrapping_mul(11);
        let after_half = half_offset.wrapping_add(packed_len);
        let after_one = one_offset.wrapping_add(packed_len);
        let after_four = four_offset.wrapping_add(packed_len);
        clear_range(dst, zero_product_len, half_offset);
        clear_range(dst, after_half, one_offset);
        clear_range(dst, after_one, four_offset);
        let upper_gap_end = if infinity_len == 0 {
            dst.len()
        } else {
            infinity_offset
        };
        clear_range(dst, after_four, upper_gap_end);
        if infinity_len != 0 {
            clear_range(dst, after_infinity, dst.len());
        }
    }

    pub const fn endpoint_slices(
        dst: &[Limb],
        split_len: usize,
        infinity_len: usize,
    ) -> (&[Limb], &[Limb]) {
        let zero_len = split_len.wrapping_mul(2);
        let (zero, _) = dst.split_at(zero_len);
        if infinity_len == 0 {
            return (zero, &[]);
        }
        let infinity_offset = split_len.wrapping_mul(15);
        let (_, infinity_and_after) = dst.split_at(infinity_offset);
        let (infinity, _) = infinity_and_after.split_at(infinity_len);
        (zero, infinity)
    }

    pub fn multiply_endpoints(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        shape: MulShape,
        split_len: usize,
    ) -> (usize, usize) {
        let low_len_a = min(a.len(), split_len);
        let low_len_b = min(b.len(), split_len);
        let (low_a, _) = a.split_at(low_len_a);
        let (low_b, _) = b.split_at(low_len_b);
        let zero_product_len = low_a.len().wrapping_add(low_b.len());
        let (zero_product, _) = dst.split_at_mut(zero_product_len);
        Recursive::recursive_mul(zero_product, low_a, low_b, scratch, TierCeiling::Toom6);

        if matches!(shape, MulShape::Balanced) {
            return (zero_product_len, 0);
        }

        let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let large_offset = split_len.wrapping_mul(8);
        let small_offset = split_len.wrapping_mul(7);
        let (_, high_large) = larger.split_at(large_offset);
        let (_, high_small) = smaller.split_at(small_offset);
        let product_len = high_large.len().wrapping_add(high_small.len());
        let product_offset = split_len.wrapping_mul(15);
        let (_, product_and_tail) = dst.split_at_mut(product_offset);
        let (product, _) = product_and_tail.split_at_mut(product_len);
        Recursive::recursive_mul(product, high_large, high_small, scratch, TierCeiling::Toom6);
        (zero_product_len, product_len)
    }
}

/// Snapshot-table access for one child-demand family.
#[cfg(feature = "std")]
impl PrefixTable {
    const fn empty() -> Self {
        Self {
            values: RwLock::new(Vec::new()),
        }
    }

    /// Returns the prefix maximum over lengths `1..=limit`.
    fn max_through(&self, limit: usize, probe: &dyn Fn(usize) -> usize) -> usize {
        /// Growth granularity balancing first-product latency against rebuilds.
        const CHUNK: usize = 512;

        {
            let values = match self.values.read() {
                Ok(values) => values,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Some(value) = values.get(limit.wrapping_sub(1)) {
                return *value;
            }
        }

        let mut values = match self.values.write() {
            Ok(values) => values,
            Err(poisoned) => poisoned.into_inner(),
        };
        if values.len() < limit {
            let target = limit.next_multiple_of(CHUNK);
            let additional = target.wrapping_sub(values.len());
            values.reserve(additional);
            let mut running = values.last().copied().unwrap_or(0);
            for len in values.len().wrapping_add(1)..=target {
                running = max(running, probe(len));
                values.push(running);
            }
        }
        values.get(limit.wrapping_sub(1)).copied().unwrap_or(0)
    }
}

fn clear_range(dst: &mut [Limb], start: usize, end: usize) {
    debug_assert!(
        start <= end && end <= dst.len(),
        "clear range exceeds product"
    );
    let (_, range_and_after) = dst.split_at_mut(start);
    let (range, _) = range_and_after.split_at_mut(end.wrapping_sub(start));
    range.fill(0);
}

fn max_recursive_mul_scratch(split_len: usize, eval_len: usize) -> usize {
    let mut inner = max_balanced_mul_scratch_below(split_len);
    for len_a in split_len..=eval_len {
        for len_b in split_len..=eval_len {
            let plan = Multiplication::select_plan(len_a, len_b, TierCeiling::Toom6);
            inner = max(inner, Multiplication::scratch_len(plan, len_a, len_b));
        }
    }
    inner
}

/// Largest scratch demand of any child square no longer than `limit`.
///
/// Bounded the same way as [`max_balanced_mul_scratch_below`], and for the same
/// reason: the exhaustive scan is quadratic in the operand length.
fn max_sqr_scratch_up_to(limit: usize) -> usize {
    let probe = |len: usize| {
        let plan = Multiplication::select_square_plan(len, TierCeiling::Toom6);
        Multiplication::square_scratch_len(plan, len)
    };
    let scanned = min(limit, EXHAUSTIVE_CHILD_SCAN);
    let mut inner = child_scratch_prefix(&SQR_PREFIX, scanned, &probe);
    if limit > scanned {
        inner = max(inner, probe(limit));
    }
    debug_assert!(
        limit > EXHAUSTIVE_CHILD_SCAN.wrapping_mul(4)
            || inner == (1..=limit).map(probe).max().unwrap_or(0),
        "prefix-plus-top bound missed a child scratch maximum up to {limit}"
    );
    inner
}

/// Largest scratch demand of any balanced child product shorter than `limit`.
///
/// Scanning the whole range is quadratic in the operand length — each length
/// costs a full recursive plan walk — which dominates the multiplication it is
/// sizing at large widths. See [`EXHAUSTIVE_CHILD_SCAN`] for why the prefix and
/// the range top together are exhaustive.
fn max_balanced_mul_scratch_below(limit: usize) -> usize {
    let probe = |len: usize| {
        let plan = Multiplication::select_plan(len, len, TierCeiling::Toom6);
        Multiplication::scratch_len(plan, len, len)
    };
    let scanned = min(limit, EXHAUSTIVE_CHILD_SCAN);
    let mut inner = child_scratch_prefix(&MUL_PREFIX, scanned.wrapping_sub(1), &probe);
    if limit > scanned {
        inner = max(inner, probe(limit.wrapping_sub(1)));
    }
    // The reference scan is the quadratic form this replaces, so it is only
    // affordable near the crossover where the prefix stops covering the range.
    debug_assert!(
        limit > EXHAUSTIVE_CHILD_SCAN.wrapping_mul(4)
            || inner == (1..limit).map(probe).max().unwrap_or(0),
        "prefix-plus-top bound missed a child scratch maximum below {limit}"
    );
    inner
}

/// Prefix maximum of one child-demand family over lengths `1..=limit`.
///
/// Both families share the shape: `[max over 1..=limit]`. With pointer-width
/// `std` that value is one synchronized indexed read after the first product at
/// a new width; without it the scan is recomputed, where the cost is immaterial.
fn child_scratch_prefix(
    family: &PrefixTableOrFamily,
    limit_inclusive: usize,
    probe: &dyn Fn(usize) -> usize,
) -> usize {
    if limit_inclusive == 0 {
        return 0;
    }
    #[cfg(feature = "std")]
    {
        family.max_through(limit_inclusive, probe)
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = family;
        (1..=limit_inclusive).map(probe).max().unwrap_or(0)
    }
}
