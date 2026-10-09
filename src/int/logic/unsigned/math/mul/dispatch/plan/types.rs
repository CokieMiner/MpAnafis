//! Multiplication and squaring plan value types.

/// Highest conventional tier available to a recursive child product.
///
/// Toom evaluators use a ceiling to guarantee that an invalid root geometry
/// falls to a strictly lower algorithm instead of redispatching to itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TierCeiling {
    Toom3,
    Toom4,
    Toom6,
    Full,
}

/// Namespace for multiplication-tower planning, dispatch, and execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Multiplication;

/// Exact large-product backend selected above the conventional Toom tower.
///
/// Unavailable backends are omitted at their target gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(not(target_pointer_width = "16"))]
pub enum LargePlan {
    /// Recursive Schonhage-Strassen over Fermat rings.
    Ssa,
}

/// Complete dispatch decision for one multiplication.
///
/// The selector establishes the chosen tier's shape and crossover constraints.
/// Execution and workspace sizing use that same decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MulPlan {
    Schoolbook,
    Lopsided,
    Karatsuba,
    Toom3,
    /// Three parts against two, for operand ratios in `[1.5, 3)`.
    Toom32,
    /// Four parts against three, for operand ratios in `[4/3, 2)`.
    Toom43,
    Toom4,
    Toom6,
    Toom8,
    #[cfg(not(target_pointer_width = "16"))]
    Large(LargePlan),
}

impl MulPlan {
    /// Whether this plan reaches the widest tier a block product can use.
    ///
    /// Lopsided block selection admits Toom-8/8.5 and transform plans here.
    #[inline]
    pub const fn reaches_widest_tier(self) -> bool {
        #[cfg(not(target_pointer_width = "16"))]
        {
            matches!(self, Self::Toom8 | Self::Large(_))
        }
        #[cfg(target_pointer_width = "16")]
        {
            matches!(self, Self::Toom8)
        }
    }

    /// Whether this plan is a transform rather than a conventional split.
    ///
    /// Wider transform blocks use a distinct policy from conventional splits.
    #[inline]
    pub const fn is_transform(self) -> bool {
        #[cfg(not(target_pointer_width = "16"))]
        {
            matches!(self, Self::Large(_))
        }
        #[cfg(target_pointer_width = "16")]
        {
            let _ = self;
            false
        }
    }
}

/// Complete dispatch decision for one square.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SquarePlan {
    Schoolbook,
    Karatsuba,
    Toom3,
    Toom4,
    Toom6,
    Toom8,
    #[cfg(not(target_pointer_width = "16"))]
    Large(LargePlan),
}

impl SquarePlan {
    /// Whether this plan uses the transform tier, when that tier exists.
    #[inline]
    pub const fn is_transform(self) -> bool {
        #[cfg(not(target_pointer_width = "16"))]
        {
            matches!(self, Self::Large(_))
        }
        #[cfg(target_pointer_width = "16")]
        {
            let _ = self;
            false
        }
    }
}

/// Shape selected for an eight-way Toom-Cook multiplication.
#[derive(Clone, Copy, Debug)]
pub enum MulShape {
    Balanced,
    Half,
}

/// Two operand widths in ascending order.
///
/// Selection and workspace sizing share this ordered pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Widths {
    pub smaller: usize,
    pub larger: usize,
}

impl Widths {
    /// Orders one operand pair.
    #[inline]
    pub const fn new(len_a: usize, len_b: usize) -> Self {
        if len_a <= len_b {
            Self {
                smaller: len_a,
                larger: len_b,
            }
        } else {
            Self {
                smaller: len_b,
                larger: len_a,
            }
        }
    }
}
