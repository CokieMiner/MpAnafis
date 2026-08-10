//! Where Arbi stands against the fastest available big-integer libraries.
//!
//! These benchmarks answer one question — are we ahead, and where are we not —
//! and unlike every other group here they tune nothing. Each library is called
//! at its raw limb-slice entry point so no allocator or wrapper sits inside the
//! timed region, and Arbi runs through the full public dispatcher rather than
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
//! Both are pinned to a single thread so the comparison stays core-for-core.

pub mod addition;
pub mod balanced;
pub mod flint;
pub mod low_product;
pub mod unbalanced;
