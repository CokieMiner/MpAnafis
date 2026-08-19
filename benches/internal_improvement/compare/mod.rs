//! Where Mp stands against the fastest available big-integer libraries.
//!
//! These benchmarks answer one question — are we ahead, and where are we not —
//! and unlike every other group here they tune nothing. Each library is called
//! at its raw limb-slice entry point so no allocator or wrapper sits inside the
//! timed region, and Mp runs through the full public dispatcher rather than
//! any forced tier, so what is compared is three complete towers.
//!
//! Two adversaries, for different reasons:
//!
//! - **GMP** is the reference below the transform crossover. Its Toom tier and
//!   assembly basecases are the bar for operands that fit in cache.
//! - **FLINT 3** is the reference above it. Its `fft_small` module is a
//!   small-prime FFT requiring AVX2 or NEON, and it is the direct competitor to
//!   our own `ntt` path. Comparing only against GMP at the top of the tower
//!   measures us against an algorithm that is no longer state of the art.
//!
//! GMP remains serial. FLINT has a one-worker production row and, when Mp's
//! feature-selected executor reports more than one worker, a production row at
//! that exact worker budget. Mp executor, geometry, and workspace comparisons
//! live in `transform_matrix`, with every dimension encoded in its case label.

pub mod addition;
pub mod balanced;
pub mod direct;
pub mod flint;
pub mod low_product;
pub mod transform_matrix;
pub mod unbalanced;
