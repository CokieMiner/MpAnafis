//! Recursive Schönhage-Strassen multiplication over Fermat rings.
//!
//! The tier works on flat, caller-owned `&mut [Limb]` coefficient matrices
//! rather than on a `Vec` of bignums, so a whole multiplication allocates once
//! at the entry point and never again. All Fermat ring arithmetic operates
//! directly on raw limb slices.
//!
//! # Layout
//!
//! - [`entry`]: the surface the rest of the crate uses — the tower's product
//!   and square, their capability predicates, and their scratch lengths.
//! - `TransformBench`: forced geometries and bare ring products, compiled only
//!   for the internal benchmark facade.
//! - [`plan`]: chooses the transform geometry and the CRT half-width.
//! - [`crt`]: the `B^n - 1` half of the top-level split.
//! - [`carry`]: carry and borrow propagation shared across the tier.
//! - [`negacyclic`]: odd-factor decomposition of the Fermat modulus.
//! - [`reconstruct`]: operand to coefficient matrix, and back.
//! - [`transform`]: the butterflies and the matrix addressing.
//! - [`product`]: the pointwise stage.
//! - [`ring`]: arithmetic in `Z/(2^n + 1)`.
//!
//! # Tuning constants
//!
//! Each constant is documented where it is defined; this is the index of which
//! ones are safe to leave alone on a new target and which are not.
//!
//! Portable — these follow from the algorithm's structure, so they change which
//! geometries are *reachable*, not which one is fastest:
//!
//! - `plan::TOP_LEVEL_SEARCH_RADIUS`, `plan::NESTED_SEARCH_RADIUS` — how far the
//!   exponent search looks either side of the analytic centre.
//! - `plan::MAX_COST_RECURSION_DEPTH` — a termination bound on the cost model.
//! - `SSA_BNM1_BASECASE_LIMBS` — where `mul_mod_bnm1` stops splitting. Also a
//!   *correctness* input: [`plan::crt_half_width`] only emits half-widths whose
//!   odd part fits inside it, which is what keeps the halving recursion off an
//!   odd width. Raising it is always safe; lowering it below the half-widths the
//!   planner emits is not.
//! - `SSA_NEGACYCLIC_FACTOR{3,5}_THRESHOLD` — where an odd factor of the
//!   Fermat modulus repays its extra folds and CRT merge.
//!
//! Hardware-tuned — these trade one machine-dependent cost against another and
//! were measured on x86-64. Other targets inherit the generic defaults from
//! `build.rs` and should be swept before those are trusted:
//!
//! - `SSA_BASE_MODULUS_BITS` — widest inner ring whose pointwise products go to
//!   the multiplication tower instead of a nested transform. The crossover
//!   depends on how fast the target's Toom tier is relative to its transform.
//! - `SSA_COEFFICIENT_VISIT_OVERHEAD` — cost of one coefficient visit relative
//!   to one limb of multiplication. That ratio is exactly what varies by target.
//! - `SSA_BASECASE_COST_WEIGHT_16THS` — fits the planner's lower-tower model
//!   between `n^1.5` and `n^1.75`; faster Toom kernels favor the lower end.
//! - `SSA_GEOMETRY_EXPONENTS` — exact measured exponents for power-of-two ring
//!   widths. Zero entries delegate to the portable recursive cost model.
//! - `SSA_SQRT2_TWIST_PASSES` — planner penalty for a half-step geometry.
//! - `SSA_FOUR_STEP_MIN_LOG` — where the cache-blocked four-step layout starts
//!   paying for its transposes. A cache-size question.
//! - `SSA_TRANSPOSE_TILE_LIMBS` — the transpose tile's cache working-set budget.
//! - `SSA_DIRECT_SHIFT_MAX_LIMBS` — coefficient width below which an
//!   in-place shift beats staging through scratch.
//! - `SSA_THRESHOLD` — where the tower switches to SSA at all.

#![allow(
    unsafe_code,
    reason = "SSA FFT orchestration uses unchecked limb access on validated buffer layouts"
)]

use super::{
    Addition, ArchKernels, LIMB_BITS, Limb, MulPlan, Multiplication, SSA_BASE_MODULUS_BITS,
    SSA_BASECASE_COST_WEIGHT_16THS, SSA_BNM1_BASECASE_LIMBS, SSA_COEFFICIENT_VISIT_OVERHEAD,
    SSA_DIRECT_SHIFT_MAX_LIMBS, SSA_FOUR_STEP_MIN_LOG, SSA_GEOMETRY_EXPONENTS,
    SSA_NEGACYCLIC_FACTOR3_THRESHOLD, SSA_NEGACYCLIC_FACTOR5_THRESHOLD,
    SSA_NESTED_COST_PENALTY_16THS, SSA_SQRT2_TWIST_PASSES, SSA_TRANSPOSE_TILE_LIMBS, SharedEval,
    TierCeiling,
};

// Every submodule reaches its siblings' namespaces through `super::`, so this is
// the one place each name is bound to the module that defines it. A file that
// imports `super::SsaRing` does not have to know which file declares it, and
// moving a declaration between files stays a one-line change here.
use carry::SsaCarry;
use negacyclic::NegacyclicPlan;
use product::SsaPointwise;
use reconstruct::{InverseTwist, SsaCoefficients};
use ring::Residue;

/// The CRT half-widths this tier computes cannot be represented on a 16-bit
/// `usize`, so the entry points are compiled out there and the tower keeps
/// using Toom-8.5 at every size.
mod carry;
mod crt;
#[cfg(not(target_pointer_width = "16"))]
mod entry;
mod negacyclic;
mod plan;
mod product;
mod reconstruct;
mod ring;
mod transform;

#[cfg(test)]
mod tests;

// Only the entry points and their tuner counterpart reach these, and both are
// compiled out on 16-bit targets.
#[cfg(not(target_pointer_width = "16"))]
pub use crt::SsaCrt;
// The multiplication tower's surface: one namespace carrying every entry point,
// plus the geometry choice they take. The forced and pinned-exponent variants are
// methods on the same namespace, gated inside `entry` rather than named here.
#[cfg(not(target_pointer_width = "16"))]
pub use entry::{Ssa, TransformChoice};
pub use plan::FftPlan;
#[cfg(not(target_pointer_width = "16"))]
pub use plan::SsaPlan;
pub use ring::SsaRing;
pub use transform::SsaTransform;
