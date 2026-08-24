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
/// Keeping large backends behind one nested plan gives multiplication and
/// squaring one extensible dispatcher shape. A backend that is unavailable on
/// a target is omitted here, so selectors and exhaustive matches cannot name
/// an impossible tier and need no dead-code allowance or sentinel arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(not(target_pointer_width = "16"))]
pub enum LargePlan {
    /// Recursive Schönhage-Strassen over Fermat rings.
    Ssa,
}

/// Complete dispatch decision for one multiplication.
///
/// One variant names one algorithm. A plan is a decision, not a hint: the
/// selector only names a tier it has already established can compute the
/// product, so [`Multiplication::execute_plan`] runs exactly the named algorithm and
/// [`Multiplication::required_scratch`] sizes exactly its
/// workspace. This value is `Copy` and contains no allocation.
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
    /// `lopsided::lopsided_block_len` sizes its blocks so that each one lands
    /// on the widest available tier, and asks this to check a candidate width.
    /// Toom-8.5 is the widest conventional split, and the transform is wider
    /// still, so both answer yes: a block that reaches a transform is
    /// better sized than one that reaches Toom-8.5, not worse.
    ///
    /// Named rather than matched at each call site because it is a policy, and
    /// two copies of a policy is precisely how the transform tier came to be
    /// unreachable for unbalanced shapes.
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
    /// Deliberately narrower than [`Self::reaches_widest_tier`], which also
    /// admits Toom-8.5. `lopsided::transform_block_len` asks this one because
    /// its widened block is justified only by a transform's indifference to the
    /// operand ratio, which Toom-8.5 does not share.
    #[inline]
    pub const fn is_transform(self) -> bool {
        #[cfg(not(target_pointer_width = "16"))]
        {
            matches!(self, Self::Large(_))
        }
        #[cfg(target_pointer_width = "16")]
        {
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
/// Every ratio test in [`shape`](super::shape) is a comparison between the two, so the pair is
/// ordered once here and carried, rather than each predicate recomputing its own
/// `min`/`max`.
///
/// Declared here rather than beside its predicates so that the fields stay
/// visible to `dispatch::scratch`, which sizes workspaces from the same ordered
/// pair.
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
