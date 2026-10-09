//! Registered bit-width ladders shared by signed and unsigned benchmark cases.
//! Each category selects the ladder appropriate to its operand construction.

/// Widths for addition, subtraction, logical operators, and shifts.
pub const ADDITIVE: [usize; 8] = [
    256, 1_024, 4_096, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216,
];

/// Power-of-two widths for multiplication and squaring.
pub const MULTIPLICATIVE: [usize; 17] = [
    256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288,
    1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216,
];

/// Division widths, including partial limbs, short-quotient transition
/// neighborhoods, and large recursive operands. Shaped division arguments
/// specify divisor bits; a balanced dividend has up to twice this width.
pub const DIVISION: [usize; 35] = [
    8, 16, 32, 60, 64, 68, 128, 132, 192, 256, 320, 512, 768, 1_024, 2_048, 4_032, 4_096, 4_160,
    6_080, 6_144, 6_208, 8_128, 8_192, 8_256, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288,
    1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216,
];

/// Widths for single-value arithmetic helpers.
pub const BALANCED: [usize; 7] = [128, 256, 1_024, 4_096, 65_536, 262_144, 1_048_576];

/// Seven widths for inspection, conversion, and arithmetic helpers.
pub const NARROW: [usize; 7] = [256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

/// GCD-family widths spanning short operands and recursive reduction.
pub const GCD: [usize; 23] = [
    64, 128, 192, 256, 320, 512, 768, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072,
    262_144, 524_288, 1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216, 33_554_432,
];

/// Integer-root widths through sixteen megabits for recursive scaling comparisons.
pub const ROOTS: [usize; 11] = [
    64, 128, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216,
];

/// Modulus widths for modular arithmetic.
pub const MODULAR: [usize; 9] = [
    64, 128, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576,
];

/// Modulus widths for modular exponentiation. Random exponent fixtures use the
/// same bit width; named scenarios specify their own exponent distributions.
pub const MODULAR_EXP: [usize; 9] = [64, 128, 256, 1_024, 2_048, 4_096, 16_384, 65_536, 262_144];

/// Extended GCD and modular inversion through recursive cofactor reduction.
pub const EXTENDED_GCD: [usize; 20] = [
    64, 128, 192, 256, 320, 512, 768, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072,
    262_144, 524_288, 1_048_576, 2_097_152, 4_194_304,
];

/// Operand widths for primality classifications and exclusive prime search.
pub const PRIMALITY: [usize; 9] = [
    64, 128, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576,
];
