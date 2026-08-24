//! Operand-shape policy: which splits a pair of widths can and should use.
//!
//! Every predicate here is a method on [`Widths`], which orders the two operand
//! lengths exactly once. Selection consults up to five of them per call and runs
//! at every recursive level, so re-deriving `min`/`max` inside each one was pure
//! repeated work on the hot dispatch path.

use core::cmp::max;

use super::{
    MulShape, Multiplication, TRANSFORM_MAX_OPERAND_RATIO, TRANSFORM_MIN_SMALLER_LIMBS, Widths,
};

const TOOM8_BALANCED_PARTS: usize = 8;
const TOOM8_HALF_LARGE_PARTS: usize = 9;
const TOOM8_HALF_SMALL_PARTS: usize = 8;
const TOOM8_BALANCED_MIN_SMALLER_UNITS: usize = 20;
const TOOM8_BALANCED_MAX_LARGER_UNITS: usize = 21;

impl Widths {
    /// Whether the product is better computed as a row of blocked balanced
    /// products than as one unbalanced split.
    ///
    /// Every Toom tier splits both operands at one width, so an unbalanced pair
    /// wastes part of every evaluated point. Cutting the longer operand into
    /// blocks the width of the shorter one turns the product into a row of
    /// *balanced* products instead, which is where the tuned tower is strong.
    ///
    /// The crossover sits where the four-way split stops accepting the shape,
    /// because that is the first ratio at which no balanced Toom tier wants the
    /// operands. Moving it here from `8:1` took the two-to-one worst cell from
    /// 2.28x to 1.13x and the four-to-one worst cell from 2.23x to 1.11x.
    #[inline]
    pub const fn prefers_blocked_product(self) -> bool {
        self.smaller != 0 && self.larger.saturating_mul(3) >= self.smaller.saturating_mul(4)
    }

    /// Whether the operands split three ways against two.
    ///
    /// The first of the fractional-ratio shapes. Blocking is efficient only when
    /// `larger / smaller` is near an integer: at 1.5 it yields one full block and
    /// one half-width block, so a balanced product plus a two-to-one product, and
    /// the bad shape reappears inside the solution. A three-by-two split has no
    /// such residue — it is the exact shape.
    ///
    /// The admitted band is `[1.5, 3)`. Below it the longer operand does not
    /// reach three parts at this split; at or above it the shorter one falls to a
    /// single part and the product is not a Toom split at all.
    #[inline]
    pub const fn toom32_suitable(self) -> bool {
        if self.smaller < 2 {
            return false;
        }
        let split_len = self.larger.div_ceil(3);
        self.larger > split_len.saturating_mul(2)
            && self.smaller > split_len
            && self.smaller <= split_len.saturating_mul(2)
    }

    /// Whether blocking would leave a residue block as unbalanced as the shape
    /// it was called to repair.
    ///
    /// Blocking cuts the longer operand into blocks the width of the shorter
    /// one, so its cost is `larger / smaller` balanced products plus one product
    /// of width `larger % smaller` against the shorter operand. That residue is
    /// nearly free when it is close to zero or close to a full block, and worst
    /// when it is close to *half* a block — there the row ends in a two-to-one
    /// product, which is the very shape blocking exists to remove.
    ///
    /// Measured against the blocked path across widths from 96 to 4608 limbs,
    /// with the fractional tier forced on every shape it can split:
    ///
    /// | `larger / smaller` | residue | Toom-3-by-2 against blocked |
    /// |---|---|---|
    /// | 2.00 | 0 | 6% to 15% *slower* |
    /// | 1.81 | 0.81 blocks | 0% to 9% faster |
    /// | 1.68 | 0.68 blocks | 14% to 26% faster |
    /// | 1.50 | 0.50 blocks | 9% to 25% faster |
    ///
    /// The quarter-block margin is where that crossed over, and it is a
    /// statement about the residue rather than about any particular ratio, so it
    /// carries to the four-by-three and five-by-four splits unchanged.
    #[inline]
    pub const fn prefers_fractional_split(self) -> bool {
        if self.smaller == 0 {
            return false;
        }
        let residue = self.larger.rem_euclid(self.smaller);
        let quarter_block = self.smaller.div_euclid(4);
        residue > quarter_block && residue < self.smaller.wrapping_sub(quarter_block)
    }

    /// Whether the operands split four ways against three.
    ///
    /// The band below [`Self::toom32_suitable`]: four parts against three admits
    /// ratios in `[4/3, 2)`, where the three-way split cannot reach because its
    /// shorter operand would still hold three parts.
    #[inline]
    pub const fn toom43_suitable(self) -> bool {
        if self.smaller < 3 {
            return false;
        }
        let split_len = self.larger.div_ceil(4);
        self.larger > split_len.saturating_mul(3)
            && self.smaller > split_len.saturating_mul(2)
            && self.smaller <= split_len.saturating_mul(3)
    }

    /// Whether one transform over the whole product beats blocking it.
    ///
    /// Two conditions, because the transform loses to blocking for two unrelated
    /// reasons.
    ///
    /// A short operand loses on padding. The transform runs in a ring of the CRT
    /// half-width whatever the shorter operand holds, so below
    /// [`TRANSFORM_MIN_SMALLER_LIMBS`] the ring is mostly padding and blocking's
    /// row of small balanced products wins outright. Forced A/B over 58 shapes
    /// from 4091 to 65537 limbs put that boundary between 1024 and 1170 limbs,
    /// softly — the two paths stay within about 10% across the band.
    ///
    /// An extreme ratio loses for a different reason, and it is the ratio rather
    /// than either width. Blocking cuts the product into pieces that are
    /// *themselves* transforms — see `lopsided::transform_block_len` — so both
    /// sides scale the same way in the longer operand and what remains is how
    /// much of the single ring is padding. Measured against production blocking
    /// over 36 shapes from 16385 to 524289 limbs: the transform wins every ratio
    /// through 16 to 1 by 1.06x to 1.51x, the two are within 10% at 32 to 1, and
    /// blocking wins every ratio from 64 to 1 by 1.14x to 1.72x.
    ///
    /// The ratio bound only became visible once blocking stopped capping its
    /// blocks just above the shorter operand. While it did, blocking measured so
    /// badly at extreme ratios that the shorter operand alone appeared to
    /// separate the two, and this predicate tested only that.
    #[inline]
    pub const fn transform_padding_is_affordable(self) -> bool {
        self.smaller >= TRANSFORM_MIN_SMALLER_LIMBS
            && self.larger <= self.smaller.saturating_mul(TRANSFORM_MAX_OPERAND_RATIO)
    }

    /// Whether a recursive Toom child is too lopsided for its own split.
    ///
    /// Distinct from [`Self::prefers_blocked_product`] even though both measure
    /// the same ratio: this one decides that an evaluated child collapses to
    /// the basecase, and it governs products all the way down the recursion
    /// rather than the top-level shape. Sharing one constant between the two
    /// meant that retuning the blocked crossover silently retuned every Toom
    /// child's floor, which measured as a broad small regression across the
    /// shape matrix even where the blocked path itself had improved.
    #[inline]
    pub const fn degenerate_child_split(self) -> bool {
        self.smaller != 0 && self.larger >= self.smaller.saturating_mul(8)
    }

    /// Whether similarly sized operands are suitable for the balanced 4-way split.
    #[inline]
    pub const fn toom4_balanced(self) -> bool {
        self.smaller.saturating_mul(4) >= self.larger.saturating_mul(3)
    }

    /// Whether similarly sized operands are suitable for the balanced six-way split.
    #[inline]
    pub const fn toom6_balanced(self) -> bool {
        self.smaller.saturating_mul(18) >= self.larger.saturating_mul(17)
    }

    /// Whether operands fit the adjacent seven-by-six Toom-6.5 split.
    #[inline]
    pub fn toom6_half_suitable(self) -> bool {
        // Widen early-reject ratio to 4/3 to evaluate suitability more broadly.
        // The exact split requirements are verified below.
        if self.smaller < 6 || self.smaller.saturating_mul(4) < self.larger.saturating_mul(3) {
            return false;
        }
        let split_len = max(self.larger.div_ceil(7), self.smaller.div_ceil(6));
        self.larger > split_len.saturating_mul(6) && self.smaller > split_len.saturating_mul(5)
    }

    /// Whether similarly sized operands fit an eight-by-eight split.
    #[inline]
    pub const fn toom8_balanced(self) -> bool {
        let scaled_smaller = self.smaller.saturating_mul(TOOM8_BALANCED_MAX_LARGER_UNITS);
        let scaled_larger = self.larger.saturating_mul(TOOM8_BALANCED_MIN_SMALLER_UNITS);
        if self.smaller < TOOM8_BALANCED_PARTS || scaled_smaller < scaled_larger {
            return false;
        }
        let split_len = self.larger.div_ceil(TOOM8_BALANCED_PARTS);
        self.smaller > split_len.saturating_mul(TOOM8_BALANCED_PARTS.wrapping_sub(1))
    }

    /// Whether operands fit the adjacent nine-by-eight Toom-8.5 split.
    #[inline]
    pub fn toom8_half_suitable(self) -> bool {
        // Widen early-reject ratio to 5/4 to evaluate suitability more broadly.
        // The exact split requirements are verified below.
        if self.smaller < TOOM8_HALF_SMALL_PARTS
            || self.smaller.saturating_mul(5) < self.larger.saturating_mul(4)
        {
            return false;
        }
        let split_len = max(
            self.larger.div_ceil(TOOM8_HALF_LARGE_PARTS),
            self.smaller.div_ceil(TOOM8_HALF_SMALL_PARTS),
        );
        self.larger > split_len.saturating_mul(TOOM8_HALF_LARGE_PARTS.wrapping_sub(1))
            && self.smaller > split_len.saturating_mul(TOOM8_HALF_SMALL_PARTS.wrapping_sub(1))
    }

    /// Select the multiplication shape for six-way Toom-Cook operands.
    ///
    /// The single resolution of the balanced-or-half question for Toom-6, for
    /// the same reason [`Self::toom8_shape`] is one for Toom-8: `toom6::mul` and
    /// `dispatch::scratch::toom6_mul_scratch_len` must agree on it exactly, and
    /// while each derived it from the two predicates itself they were only ever
    /// agreeing by hand. They stopped when the blocked crossover moved — the
    /// algorithm recursed under a Toom-4 ceiling while the sizing had sized
    /// under the full one — and the Toom-6 property test caught the undersized
    /// buffer only because its random widths happened to land on a lopsided
    /// ratio.
    #[inline]
    pub fn toom6_shape(self) -> Option<MulShape> {
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
    pub fn toom8_shape(self) -> Option<MulShape> {
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
        if len < TOOM8_BALANCED_PARTS {
            return false;
        }
        let split_len = len.div_ceil(TOOM8_BALANCED_PARTS);
        len > split_len.wrapping_mul(TOOM8_BALANCED_PARTS.wrapping_sub(1))
    }

    /// Whether a width crossover admits this shape. Zero disables the tier.
    ///
    /// Keyed on the *longer* operand, for every tier. A Toom tier splits the longer
    /// operand into `k` parts and the shorter into `k` or `k-1` parts of the same
    /// width, so every recursive child is about `larger / k` limbs wide whatever the
    /// shorter operand is. A transform is keyed the same way for a different reason:
    /// its own cost is the CRT half-width regardless of the ratio while the
    /// conventional tower's is not, so the transform's advantage grows with
    /// imbalance and the width at which it wins must fall as the ratio worsens —
    /// which `larger >= threshold` is exactly.
    ///
    /// Both forms collapse to the balanced crossover at equal widths, so this
    /// generalises those thresholds rather than retuning them. Testing the shorter
    /// operand instead denied each tier to the unbalanced shapes it exists for: at
    /// 200 by 160 limbs `toom4_balanced` accepted the shape while the shorter
    /// operand sat below the four-way crossover, so the whole ladder was skipped and
    /// the product fell to Karatsuba at 1.47x behind the reference. For the
    /// transform the same defect left the band below the crossover on the tower, at
    /// 4091 limbs measuring 1.15x to 1.22x behind where the transform was ahead.
    #[inline]
    pub const fn crossover_admits(threshold: usize, widths: Widths) -> bool {
        match threshold {
            0 => false,
            enabled_threshold => widths.larger >= enabled_threshold,
        }
    }
}
