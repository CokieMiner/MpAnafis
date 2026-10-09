//! Full and truncated Fermat-ring transforms and product execution.
//!
//! Complete transforms use matched DIF/DIT radix-four subtrees. A polynomial
//! with a proven zero tail uses frequency-prefix TFTs and a mixed-coordinate
//! ITFT; reconstruction consumes only the established coefficient prefix.

use super::{
    AddSubFromKernel, ArchKernels, CACHE_BLOCK_BYTES, FftPlan, LIMB_BITS, Limb, LimbOutput,
    MulTransformInput, MulTransformPlan, PointwiseMulPlan, PointwiseSquarePlan, Residue,
    SSA_BASE_MODULUS_BITS, SSA_PARALLEL_MIN_LIMB_WORK, SquareTransformInput, SquareTransformPlan,
    SsaCoefficients, SsaPlan, SsaPointwise, SsaRing,
};

mod convolution;
mod convolution_entry;
mod dense;
mod drive;
mod entry;
mod forward;
mod forward_stage;
mod inverse;
mod inverse_stage;
mod itft;
mod matrix_forward;
mod matrix_fused;
mod matrix_inverse;
mod matrix_view;
mod square;
mod staging;
mod tft;
mod truncated_product;
mod two_by_one;

pub use convolution::Convolution;
pub use dense::DenseWorkspace;
pub use drive::SsaTransform;
pub use entry::TransformContext;
pub use matrix_view::CoefficientView;
pub use tft::TruncatedTransform;

#[cfg(test)]
mod tests;
