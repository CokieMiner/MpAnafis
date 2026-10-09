//! Integer powers, exponent bounds, and sliding-window planning.
//!
//! References:
//! - D. E. Knuth, "The Art of Computer Programming", Volume 2,
//!   3rd edition, Addison-Wesley, 1997, Section 4.6.3 (evaluation of powers).
//! - R. P. Brent and P. Zimmermann, "Modern Computer Arithmetic",
//!   Cambridge University Press, 2011, Section 2.6.3 (sliding windows).

#![expect(
    unsafe_code,
    reason = "Positive exponent widths and set leading bits prove exact decrements and initialized sliding-window accesses."
)]

use core::mem::swap;

use super::{InternalMpUint, LIMB_BITS, Limb, MulScratch};

/// Namespace for exponent bounds and sliding-window planning.
#[derive(Clone, Copy, Debug)]
pub struct Exponentiation;

/// A positive exponent's sliding window and required odd-power prefix.
#[derive(Clone, Copy, Debug)]
pub struct SlidingWindowPlan {
    /// Maximum number of exponent bits in one odd window.
    pub width: usize,
    /// Initialized prefix of `a, a^3, a^5, ...`, including `a`.
    pub powers: usize,
    /// Leading odd-power index and number of exponent bits consumed.
    pub initial: (usize, usize),
}

/// Bounds odd digits and admits a width when its product estimate wins.
#[derive(Clone, Copy, Debug)]
struct WindowBound {
    powers: usize,
    population_threshold: usize,
}

impl InternalMpUint {
    /// Computes `self^exp`.
    #[must_use]
    pub fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::one();
        }
        if exp == 1 || self.is_zero() || self.is_one() {
            return self.clone();
        }
        self.pow_nontrivial(exp)
    }

    /// Computes `self^exp` after the caller establishes `self > 1` and `exp > 1`.
    ///
    /// Public precision policies handle zero, unit bases, and degree zero or one
    /// before entering this shared arithmetic driver.
    #[must_use]
    pub fn pow_nontrivial(&self, exp: u32) -> Self {
        debug_assert!(exp > 1, "nontrivial exponent exceeds one");
        debug_assert!(
            !self.is_zero() && !self.is_one(),
            "nontrivial base exceeds one"
        );
        let base_val = self;
        if exp == 2 {
            return base_val.square();
        }
        if base_val.is_power_of_two()
            && let Some(shift) = usize::try_from(exp)
                .ok()
                .and_then(|degree| base_val.trailing_zeros().checked_mul(degree))
        {
            // (2^k)^exp = 2^(k*exp). A representable bit offset constructs the
            // exact result with one allocation and no multiplication scratch.
            return Self::power_of_two(shift);
        }

        // SAFETY: exp > 1 gives leading_zeros <= 30 and a width in 2..=32.
        let exp_bits = unsafe { 32_u32.unchecked_sub(exp.leading_zeros()) };
        // base < 2^k implies base^exp < 2^(k*exp). An unrepresentable limb
        // bound leaves allocation to incremental growth.
        let expected_limbs = limb_bound(base_val.significant_bits(), exp).unwrap_or_default();
        let mut result = Self::with_capacity(expected_limbs);
        result.clone_from(base_val);

        let mut temp = Self::with_capacity(expected_limbs);
        let mut scratch = MulScratch::default();

        // SAFETY: exp > 1 gives exp_bits >= 2; the leading bit is consumed above.
        let remaining = unsafe { exp_bits.unchecked_sub(1) };
        for i in (0..remaining).rev() {
            temp.assign_square_with_scratch(&result, &mut scratch);
            swap(&mut result, &mut temp);

            if (exp >> i) & 1 == 1 {
                temp.assign_product_with_scratch(&result, base_val, &mut scratch);
                swap(&mut result, &mut temp);
            }
        }
        result
    }
}

impl Exponentiation {
    /// Maximum supported sliding window; the smallest limb has more bits.
    pub const MAX_WINDOW: usize = 6;
    /// Odd digits below `2^MAX_WINDOW` index this many cached powers.
    pub const ODD_POWER_CAPACITY: usize = 1 << (Self::MAX_WINDOW - 1);

    // Rows are minimum set-bit distances, columns are supported window widths.
    // Both dimensions and every odd-digit bound derive from MAX_WINDOW.
    const WINDOW_BOUNDS: [[WindowBound; Self::MAX_WINDOW]; Self::MAX_WINDOW] = window_bounds();
}

/// Returns an upper bound on the limb width of `base^exp`.
///
/// Quotient/remainder decomposition avoids forming the possibly overflowing
/// bit count. An unrepresentable bound leaves allocation to incremental growth.
pub fn limb_bound(significant_bits: usize, exp: u32) -> Option<usize> {
    if significant_bits == 0 || exp == 0 {
        return Some(0);
    }
    let exp_usize = usize::try_from(exp).ok()?;
    let full_limbs = significant_bits.div_euclid(LIMB_BITS);
    let partial_bits = significant_bits.rem_euclid(LIMB_BITS);
    let exponent_limbs = exp_usize.div_euclid(LIMB_BITS);
    let exponent_bits = exp_usize.rem_euclid(LIMB_BITS);
    let whole = full_limbs.checked_mul(exp_usize)?;
    let partial_whole = partial_bits.checked_mul(exponent_limbs)?;
    let partial_product = partial_bits.checked_mul(exponent_bits)?;
    let partial = partial_whole.checked_add(partial_product.div_ceil(LIMB_BITS))?;
    whole.checked_add(partial)
}

impl Exponentiation {
    /// Plans a window and a sufficient odd-power prefix for a positive exponent.
    ///
    /// `bits` is the positive significant width of `exp`. A width needs only
    /// the odd-power prefix through its largest referenced digit. With `k`
    /// remaining windows, largest index `j`, and leading length `l`, execution
    /// performs `k+j` products and `bits-l+usize::from(j!=0)` squares.
    /// General residue engines minimize their sum over every supported width.
    ///
    /// The model counts a reduced square and product as one operation each;
    /// it does not claim an exact hardware timing. Common domain preparation
    /// and encoding are independent of the window. Equal work retains the
    /// narrower window, including binary when a table saves no arithmetic.
    /// `FIXED_WIDTH` selects the cheaper population/spacing model for scalar
    /// and inline residue engines; it bounds digits without scanning each width.
    #[expect(
        clippy::inline_always,
        reason = "Inlining population admission and const-table choices removes the observed out-of-line 32-byte plan return in fixed-width powers; general candidate scans remain in a separate kernel"
    )]
    #[inline(always)]
    pub fn window_plan<const FIXED_WIDTH: bool>(
        exp: &InternalMpUint,
        bits: usize,
    ) -> SlidingWindowPlan {
        debug_assert!(bits != 0, "window planning requires a positive exponent");
        debug_assert_eq!(bits, exp.significant_bits(), "exact exponent width");
        let limbs = exp.limbs();
        let mut ones = 0_usize;
        for &limb in limbs {
            // SAFETY: native bit counts fit usize on all supported targets;
            // the total population is at most bits<=usize::MAX.
            ones = unsafe {
                ones.unchecked_add(usize::try_from(limb.count_ones()).unwrap_unchecked())
            };
        }
        let binary = SlidingWindowPlan {
            width: 1,
            powers: 1,
            initial: (0, 1),
        };
        // At most three set bits cannot reduce the counted square/product sum.
        // For a leading odd digit of length l, index j and k remaining windows,
        // j+k>=l when there are three set bits and j>0; table preparation then
        // replaces at least every skipped square. Two set bits also cannot win.
        if ones <= 3 {
            return binary;
        }
        let gap = minimum_spacing(limbs);
        if gap == Self::MAX_WINDOW {
            return binary;
        }
        if FIXED_WIDTH {
            let mut best = binary;
            // SAFETY: minimum_spacing returns 1..=MAX_WINDOW, and the
            // preceding branch excluded MAX_WINDOW.
            let bounds = unsafe { Self::WINDOW_BOUNDS.get_unchecked(gap.unchecked_sub(1)) };
            for (column, bound) in bounds.iter().enumerate() {
                if ones > bound.population_threshold {
                    // Admission beats binary and implies ones>g, because
                    // powers>=g makes that intersection at least g. With
                    // set-bit spacing d, bits>=1+(ones-1)*d>width, so no
                    // runtime width limit is required for accepted choices.
                    // SAFETY: column<MAX_WINDOW bounds the positive width.
                    best.width = unsafe { column.unchecked_add(1) };
                    best.powers = bound.powers;
                }
            }
            if best.width != 1 {
                best.initial = Self::window(limbs, bits, best.width);
            }
            return best;
        }
        general_window(limbs, bits, ones, gap)
    }
}

/// Counts the exact square/product sum and largest referenced odd digit.
/// The separate kernel keeps candidate state outside inline admission.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Only widened work scores are added: at most two native bit counts and 32 table entries fit u128 on every supported pointer width"
)]
#[inline(never)]
fn general_window(limbs: &[Limb], bits: usize, ones: usize, gap: usize) -> SlidingWindowPlan {
    let mut best = SlidingWindowPlan {
        width: 1,
        powers: 1,
        initial: (0, 1),
    };
    // SAFETY: ones>=4 and bits>=ones prove both positive decrements.
    // Widening usize to u128 is exact on every supported pointer width.
    // Scores are bounded by 2*usize::MAX+ODD_POWER_CAPACITY, below 2^66.
    let mut best_work = unsafe {
        u128::try_from(bits.unchecked_sub(1)).unwrap_unchecked()
            + u128::try_from(ones.unchecked_sub(1)).unwrap_unchecked()
    };
    for width in 2..=Exponentiation::MAX_WINDOW.min(bits) {
        if width <= gap {
            continue;
        }
        let initial = Exponentiation::window(limbs, bits, width);
        // A window contains at most width set bits. This optimistic bound
        // includes the known leading power and discards impossible winners
        // before scanning the remaining exponent.
        // SAFETY: initial.0<2^(width-1), width<=6<LIMB_BITS bounds this
        // odd digit. Its population fits usize and is at most ones.
        let leading_ones =
            unsafe { usize::try_from(((initial.0 << 1) | 1).count_ones()).unwrap_unchecked() };
        // SAFETY: leading_ones<=ones and initial.1<=bits prove decrements;
        // index<=31, bits<=usize::MAX and the widened sum is below 2^66.
        let lower_work = unsafe {
            u128::try_from(ones.unchecked_sub(leading_ones).div_ceil(width)).unwrap_unchecked()
                + u128::try_from(initial.0).unwrap_unchecked()
                + u128::try_from(bits.unchecked_sub(initial.1)).unwrap_unchecked()
                + u128::from(initial.0 != 0)
        };
        if lower_work >= best_work {
            continue;
        }
        let mut largest = initial.0;
        let mut products = 0_usize;
        // SAFETY: the leading window consumes 1..=bits bits.
        let mut remaining = unsafe { bits.unchecked_sub(initial.1) };
        loop {
            remaining = next_set_bit(limbs, remaining);
            if remaining == 0 {
                break;
            }
            let (index, consumed) = Exponentiation::window(limbs, remaining, width);
            largest = largest.max(index);
            // SAFETY: each iteration consumes at least one distinct set
            // bit, so products+1<=ones-1<=bits-1. consumed<=remaining.
            unsafe {
                products = products.unchecked_add(1);
                remaining = remaining.unchecked_sub(consumed);
            }
        }
        // SAFETY: all values widen exactly. The index is at most 31,
        // products<=bits-1 and the widened total remains below 2^66.
        let work = unsafe {
            u128::try_from(products).unwrap_unchecked()
                + u128::try_from(largest).unwrap_unchecked()
                + u128::try_from(bits.unchecked_sub(initial.1)).unwrap_unchecked()
                + u128::from(largest != 0)
        };
        if work < best_work {
            best_work = work;
            best = SlidingWindowPlan {
                width,
                // SAFETY: largest<ODD_POWER_CAPACITY<=32.
                powers: unsafe { largest.unchecked_add(1) },
                initial,
            };
        }
    }
    best
}

/// Finds the minimum set-bit distance, capped at `MAX_WINDOW`.
/// A nonzero limb pair bridges the low word's highest and high word's
/// lowest set bit. An intervening zero limb exceeds every supported window.
fn minimum_spacing(limbs: &[Limb]) -> usize {
    let mut gap = Exponentiation::MAX_WINDOW;
    let mut previous_zeros = LIMB_BITS;
    for &limb in limbs {
        if limb == 0 {
            previous_zeros = LIMB_BITS;
            continue;
        }
        for distance in 1..gap {
            if limb & (limb >> distance) != 0 {
                gap = distance;
                break;
            }
        }
        if gap == 1 {
            return 1;
        }
        // SAFETY: bit counts are at most 64; the bridge is at most 129,
        // representable even on 16-bit targets. Every conversion is exact.
        let bridge = unsafe {
            previous_zeros
                .unchecked_add(1)
                .unchecked_add(usize::try_from(limb.trailing_zeros()).unwrap_unchecked())
        };
        gap = gap.min(bridge);
        if gap == 1 {
            return 1;
        }
        // SAFETY: this native bit count is at most 64 and fits usize.
        previous_zeros = unsafe { usize::try_from(limb.leading_zeros()).unwrap_unchecked() };
    }
    gap
}

/// Generates digit bounds and exact integer admission for each rational score.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Const generation visits distances and widths 1..=6; digits fit six bits, guarded decrements are exact, and intersection numerators are at most 31*6*5=930"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "Both const-generation indices are guarded by the corresponding array dimension; safe indexing keeps the table initializer compatible with stable Rust"
)]
const fn window_bounds() -> [[WindowBound; Exponentiation::MAX_WINDOW]; Exponentiation::MAX_WINDOW]
{
    let mut bounds = [[WindowBound {
        powers: 1,
        population_threshold: usize::MAX,
    }; Exponentiation::MAX_WINDOW]; Exponentiation::MAX_WINDOW];
    let mut populations = [[1_usize; Exponentiation::MAX_WINDOW]; Exponentiation::MAX_WINDOW];
    let mut row = 0;
    while row < Exponentiation::MAX_WINDOW {
        let distance = row + 1;
        let mut column = 0;
        while column < Exponentiation::MAX_WINDOW {
            // An odd digit contains bit zero. Greedily choose its largest
            // remaining bit, leaving distance bits above each lower one.
            let mut digit = 1_usize;
            let mut position = column;
            while position >= distance {
                digit |= 1 << position;
                position -= distance;
            }
            bounds[row][column].powers = (digit >> 1) + 1;
            populations[row][column] = 1 + column.div_euclid(distance);
            column += 1;
        }
        column = 1;
        while column < Exponentiation::MAX_WINDOW {
            let population = populations[row][column];
            if population > 1 {
                let mut threshold = 0;
                let mut previous = 0;
                while previous < column {
                    let earlier_population = populations[row][previous];
                    if population == earlier_population {
                        // An earlier width with the same capacity has a
                        // smaller power prefix, so this width never wins.
                        threshold = usize::MAX;
                        break;
                    }
                    // For prefixes p,q and capacities g,h with g>h,
                    // p-2+ones/g < q-2+ones/h iff
                    // ones > (p-q)*g*h/(g-h). Taking the floor gives exact
                    // strict admission for integer populations. Prefixes
                    // and capacities increase with width; denominators
                    // are positive and the numerator is at most 930.
                    let numerator = (bounds[row][column].powers - bounds[row][previous].powers)
                        * population
                        * earlier_population;
                    let intersection = numerator.div_euclid(population - earlier_population);
                    if intersection > threshold {
                        threshold = intersection;
                    }
                    previous += 1;
                }
                bounds[row][column].population_threshold = threshold;
            }
            column += 1;
        }
        row += 1;
    }
    bounds
}

/// Returns the significant width of the next set bit below `bits`.
/// Whole zero limbs are skipped without examining their individual bits.
fn next_set_bit(limbs: &[Limb], mut bits: usize) -> usize {
    while bits != 0 {
        // SAFETY: positive bits<=the exponent's significant width bounds the
        // initialized limb containing bits-1 and the native bit count.
        let (index, available) = unsafe {
            let position = bits.unchecked_sub(1);
            (
                position >> LIMB_BITS.trailing_zeros(),
                (position & (LIMB_BITS - 1)).unchecked_add(1),
            )
        };
        // SAFETY: 1<=available<=LIMB_BITS bounds the shift below LIMB_BITS;
        // index addresses the initialized exponent limb proved above.
        let digit = unsafe {
            *limbs.get_unchecked(index) & (usize::MAX >> LIMB_BITS.unchecked_sub(available))
        };
        // SAFETY: available<=bits removes only the current limb prefix.
        let preceding = unsafe { bits.unchecked_sub(available) };
        if digit != 0 {
            // SAFETY: a nonzero digit has leading_zeros<LIMB_BITS, which
            // fits usize; its significant width is at most available.
            return unsafe {
                preceding.unchecked_add(
                    LIMB_BITS
                        .unchecked_sub(usize::try_from(digit.leading_zeros()).unwrap_unchecked()),
                )
            };
        }
        bits = preceding;
    }
    0
}

impl Exponentiation {
    /// Extracts the longest odd prefix of a window starting with a set bit.
    /// Returns its odd-power table index and the number of consumed bits.
    /// `limbs` is the canonical exponent; `bits` is positive, at most its
    /// significant width, and ends at a set bit;
    /// `maximum` is in `1..=6`.
    pub fn window(limbs: &[Limb], bits: usize, maximum: usize) -> (usize, usize) {
        let width = maximum.min(bits);
        // SAFETY: callers supply bits>0 and 1<=maximum<=6. Thus 1<=width<=bits
        // and width<LIMB_BITS; start is a representable initialized exponent bit.
        let start = unsafe { bits.unchecked_sub(width) };
        let word = start >> LIMB_BITS.trailing_zeros();
        let offset = start & (LIMB_BITS - 1);
        // SAFETY: start<bits<=the exponent's significant width bounds this read.
        let mut value = unsafe { *limbs.get_unchecked(word) } >> offset;
        // SAFETY: offset<LIMB_BITS proves a positive, nonoverflowing complement.
        let available = unsafe { LIMB_BITS.unchecked_sub(offset) };
        if width > available {
            // SAFETY: start+width=bits<=the significant width and width>available
            // prove that word+1 is initialized. available<width<=6 bounds the shift.
            value |= unsafe { *limbs.get_unchecked(word.unchecked_add(1)) } << available;
        }
        // SAFETY: 1<=width<=6<LIMB_BITS makes the shift and subtraction exact.
        value &= unsafe { (1_usize << width).unchecked_sub(1) };
        // SAFETY: the set leading bit bounds this count below width<=6,
        // which fits usize on all supported pointer widths.
        let trailing = unsafe { usize::try_from(value.trailing_zeros()).unwrap_unchecked() };
        // SAFETY: the leading bit is set, so trailing<width and trailing+1<=6.
        // Removing zero suffix bits yields an odd digit; division by two indexes
        // the corresponding table entry for that digit's power.
        unsafe {
            (
                value >> trailing.unchecked_add(1),
                width.unchecked_sub(trailing),
            )
        }
    }
}
