//! Exact operand-ratio preferences and nonempty polynomial-part constraints.

#![expect(
    unsafe_code,
    reason = "Ratio identities subtract bounded integer quotients; ceiling chunk offsets fit every supported usize width"
)]

use super::{MulShape, Multiplication, Toom8, Widths};
#[cfg(not(target_pointer_width = "16"))]
use super::{TRANSFORM_MAX_OPERAND_RATIO, TRANSFORM_MIN_SMALLER_LIMBS};

// Performance preference for the balanced evaluator: larger/smaller <= 18/17.
const TOOM6_BALANCED_MAX_LARGER_UNITS: usize = 18;
// Performance preference for the balanced evaluator: larger/smaller <= 21/20.
// The separate nonempty-part checks below establish mathematical validity.
const TOOM8_BALANCED_MAX_LARGER_UNITS: usize = 21;

impl Widths {
    /// Whether the nonempty operand ratio is at least four to three.
    #[inline]
    pub const fn prefers_blocked_product(self) -> bool {
        // 3*l >= 4*s iff s <= floor(3*l/4) = l-ceil(l/4).
        // SAFETY: ceil(l/4) <= l for every usize value, including zero.
        let maximum_smaller = unsafe { self.larger.unchecked_sub(self.larger.div_ceil(4)) };
        self.smaller != 0 && self.smaller <= maximum_smaller
    }

    /// Whether the operands split three ways against two.
    ///
    /// With `m=ceil(larger/3)`, both high parts must be nonempty and the
    /// shorter operand must occupy at most two parts.
    #[inline]
    pub const fn toom32_suitable(self) -> bool {
        if self.smaller < 2 {
            return false;
        }
        let split_len = self.larger.div_ceil(3);
        // SAFETY: usize::MAX is divisible by three on 16/32/64-bit targets.
        // Thus ceil(larger/3) <= usize::MAX/3 and twice the split fits.
        let two_parts = unsafe { split_len.unchecked_mul(2) };
        self.larger > two_parts && self.smaller > split_len && self.smaller <= two_parts
    }

    /// Whether blocking would leave an unbalanced residue block.
    ///
    /// The block residue must lie strictly between the retained quarter bounds.
    #[inline]
    pub const fn prefers_fractional_split(self) -> bool {
        // Equal widths have zero block residue, including every balanced
        // recursive evaluation. Establish that before the variable remainder.
        if self.smaller == 0 || self.larger == self.smaller {
            return false;
        }
        let residue = self.larger.rem_euclid(self.smaller);
        let quarter_block = self.smaller.div_euclid(4);
        // SAFETY: quarter_block = smaller/4 <= smaller for every usize value.
        let upper_residue = unsafe { self.smaller.unchecked_sub(quarter_block) };
        residue > quarter_block && residue < upper_residue
    }

    /// Whether the operands split four ways against three.
    ///
    /// With `m=ceil(larger/4)`, the longer operand has four nonempty parts
    /// and the shorter operand has three.
    #[inline]
    pub const fn toom43_suitable(self) -> bool {
        if self.smaller < 3 {
            return false;
        }
        let split_len = self.larger.div_ceil(4);
        // SAFETY: ceil(larger/4) <= 2^(w-2), so twice and three times
        // this split are below 2^w on every supported pointer width.
        let (two_parts, three_parts) =
            unsafe { (split_len.unchecked_mul(2), split_len.unchecked_mul(3)) };
        self.larger > three_parts && self.smaller > two_parts && self.smaller <= three_parts
    }

    /// Whether single-transform evaluation avoids excessive zero-padding overhead.
    ///
    /// The shorter operand clears its minimum width and the tuned aspect ratio.
    #[cfg(not(target_pointer_width = "16"))]
    #[inline]
    pub const fn transform_padding_is_affordable(self) -> bool {
        self.smaller >= TRANSFORM_MIN_SMALLER_LIMBS
            && self.smaller >= self.larger.div_ceil(TRANSFORM_MAX_OPERAND_RATIO)
    }

    /// Whether a recursive Toom child is too lopsided for its own split.
    ///
    /// Decides whether an evaluated child subproblem collapses directly to basecase
    /// multiplication rather than continuing Toom polynomial evaluation.
    #[inline]
    pub const fn degenerate_child_split(self) -> bool {
        self.smaller != 0 && self.smaller <= self.larger.div_euclid(8)
    }

    /// Whether similarly sized operands are suitable for the balanced 4-way split.
    #[inline]
    pub const fn toom4_balanced(self) -> bool {
        // 4*s >= 3*l iff s >= ceil(3*l/4) = l-floor(l/4).
        // SAFETY: floor(l/4) <= l for every usize value.
        self.smaller >= unsafe { self.larger.unchecked_sub(self.larger.div_euclid(4)) }
    }

    /// Whether similarly sized operands are suitable for the balanced six-way split.
    #[inline]
    pub const fn toom6_balanced(self) -> bool {
        // 18*s >= 17*l iff s >= l-floor(l/18).
        // SAFETY: floor(l/18) <= l for every usize value.
        self.smaller
            >= unsafe {
                self.larger
                    .unchecked_sub(self.larger.div_euclid(TOOM6_BALANCED_MAX_LARGER_UNITS))
            }
    }

    /// Whether operands fit the adjacent seven-by-six Toom-6.5 split.
    #[inline]
    pub const fn toom6_half_suitable(self) -> bool {
        if self.smaller < 6 || !self.toom4_balanced() {
            return false;
        }
        let split_len = Self::new(self.larger.div_ceil(7), self.smaller.div_ceil(6)).larger;
        self.larger > split_len.saturating_mul(6) && self.smaller > split_len.saturating_mul(5)
    }

    /// Whether similarly sized operands fit an eight-by-eight split.
    #[inline]
    pub const fn toom8_balanced(self) -> bool {
        // 21*s >= 20*l iff s >= l-floor(l/21).
        // SAFETY: floor(l/21) <= l for every usize value.
        let minimum_smaller = unsafe {
            self.larger
                .unchecked_sub(self.larger.div_euclid(TOOM8_BALANCED_MAX_LARGER_UNITS))
        };
        if self.smaller < Toom8::BALANCED_PARTS || self.smaller < minimum_smaller {
            return false;
        }
        let split_len = self.larger.div_ceil(Toom8::BALANCED_PARTS);
        // SAFETY: ceil(larger/8) <= 2^(w-3), so seven times the split fits.
        self.smaller > unsafe { split_len.unchecked_mul(Toom8::BALANCED_PARTS - 1) }
    }

    /// Whether operands fit the adjacent nine-by-eight Toom-8.5 split.
    #[inline]
    pub const fn toom8_half_suitable(self) -> bool {
        // 5*s >= 4*l iff s >= l-floor(l/5).
        // SAFETY: floor(l/5) <= l for every usize value.
        let minimum_smaller = unsafe { self.larger.unchecked_sub(self.larger.div_euclid(5)) };
        if self.smaller < Toom8::HALF_SMALL_PARTS || self.smaller < minimum_smaller {
            return false;
        }
        let split_len = Self::new(
            self.larger.div_ceil(Toom8::HALF_LARGE_PARTS),
            self.smaller.div_ceil(Toom8::HALF_SMALL_PARTS),
        )
        .larger;
        self.larger > split_len.saturating_mul(Toom8::HALF_LARGE_PARTS - 1)
            && self.smaller > split_len.saturating_mul(Toom8::HALF_SMALL_PARTS - 1)
    }

    /// Select the multiplication shape for six-way Toom-Cook operands.
    ///
    /// Execution and scratch sizing use the same shape decision.
    #[inline]
    pub const fn toom6_shape(self) -> Option<MulShape> {
        if self.toom6_balanced() {
            Some(MulShape::Balanced)
        } else if self.toom6_half_suitable() {
            Some(MulShape::Half)
        } else {
            None
        }
    }

    /// Select the multiplication shape for eight-way Toom-Cook operands.
    #[inline]
    pub const fn toom8_shape(self) -> Option<MulShape> {
        if self.toom8_balanced() {
            Some(MulShape::Balanced)
        } else if self.toom8_half_suitable() {
            Some(MulShape::Half)
        } else {
            None
        }
    }
}

impl Multiplication {
    /// Whether a single operand has four radix-B^m chunks according to `split_len`.
    pub const fn operand_has_four_parts(len: usize, split_len: usize) -> bool {
        len > split_len.saturating_mul(3)
    }

    /// Whether a single operand has eight radix-B^m chunks.
    pub const fn operand_has_eight_parts(len: usize) -> bool {
        if len < Toom8::BALANCED_PARTS {
            return false;
        }
        let split_len = len.div_ceil(Toom8::BALANCED_PARTS);
        // SAFETY: usize::MAX = 2^w-1 on 16/32/64-bit targets, so ceil(len/8)
        // <= 2^(w-3). Multiplication by seven is strictly below 2^w.
        len > unsafe { split_len.unchecked_mul(Toom8::BALANCED_PARTS - 1) }
    }

    /// Whether a width crossover admits this shape. Zero disables the tier.
    ///
    /// The larger operand determines the polynomial split width.
    #[inline]
    pub const fn crossover_admits(threshold: usize, widths: Widths) -> bool {
        match threshold {
            0 => false,
            enabled_threshold => widths.larger >= enabled_threshold,
        }
    }
}
