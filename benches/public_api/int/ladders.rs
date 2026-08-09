//! The bit-width ladders every integer benchmark draws its arguments from.
//!
//! One definition per shape of cost curve, shared by the signed and unsigned
//! trees so a cell on one side is directly comparable with the same cell on the
//! other. A ladder is chosen by how the operation scales, not by how fast it is:
//! [`ADDITIVE`] runs to four megabits because linear operations stay cheap
//! there, while [`MULTIPLICATIVE`] stops at 262144 bits and steps in powers of
//! two so every multiplication threshold (Karatsuba, each Toom variant, the
//! transforms) gets at least one argument on each side of it.

/// Linear-cost operations: addition, subtraction, bitwise operators, shifts.
///
/// Reaches four megabits, where the operands no longer fit any cache level and
/// the loop is purely memory bound.
pub const ADDITIVE: [usize; 8] = [
    256, 1_024, 4_096, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216,
];

/// Multiplication and squaring, stepping by powers of two through every
/// sub-quadratic threshold.
pub const MULTIPLICATIVE: [usize; 15] = [
    256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288,
    1_048_576, 2_097_152, 4_194_304,
];

/// Division and other quadratic-to-quasilinear operations.
pub const DIVISION: [usize; 9] = [
    256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 2_097_152, 4_194_304,
];

/// Operations dominated by a fixed per-call cost, where the interesting region
/// is the inline-storage boundary at 256 bits rather than the asymptote.
pub const BALANCED: [usize; 7] = [128, 256, 1_024, 4_096, 65_536, 262_144, 1_048_576];

/// The default three-point ladder for cheap operations whose cost is linear and
/// uninteresting past a few thousand bits.
pub const NARROW: [usize; 7] = [256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

/// Number theory: GCD and friends, which are superlinear and allocate heavily.
pub const THEORY: [usize; 7] = [256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

/// Roots, which are cheap enough per bit to run the full division ladder.
pub const ROOTS: [usize; 7] = [256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

/// Modular arithmetic, at the widths real moduli take.
pub const MODULAR: [usize; 7] = [256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

/// Modular exponentiation, whose cost is cubic in the width: one modular
/// multiplication per exponent bit, and the exponent is as wide as the modulus.
pub const MODULAR_EXP: [usize; 7] = [256, 1_024, 2_048, 4_096, 16_384, 65_536, 262_144];

/// Extended GCD and modular inversion, quadratic without a half-GCD.
pub const EXTENDED_GCD: [usize; 7] = [256, 1_024, 2_048, 4_096, 16_384, 65_536, 262_144];

/// Primality, where a single call is already hundreds of modular
/// exponentiations.
pub const PRIMALITY: [usize; 7] = [256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];
