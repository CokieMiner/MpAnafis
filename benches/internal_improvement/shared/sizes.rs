//! Crossover size sets and operand-shape ladders.

pub const SCHOOLBOOK_SIZES: [usize; 45] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 36, 40, 43, 44, 48, 52, 56, 63, 64, 80, 87, 96, 127, 128,
];
pub const KARATSUBA_SIZES: [usize; 61] = [
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 36, 40, 43, 44, 48, 52, 56, 63, 64, 65, 66, 75, 76, 80, 87, 96, 127, 128, 160, 192, 224,
    240, 248, 252, 255, 256, 257, 260, 264, 300, 320, 384, 512, 768, 1_024, 1_280,
];
pub const TOOM3_SIZES: [usize; 38] = [
    32, 36, 40, 43, 44, 48, 52, 56, 63, 64, 68, 80, 87, 96, 127, 128, 160, 192, 224, 240, 248, 252,
    255, 256, 257, 260, 264, 300, 320, 384, 512, 768, 1_024, 1_280, 1_536, 2_048, 3_072, 4_096,
];
pub const TOOM4_SIZES: [usize; 35] = [
    64, 68, 80, 87, 96, 127, 128, 160, 192, 224, 240, 248, 252, 255, 256, 257, 260, 264, 300, 320,
    384, 512, 513, 544, 768, 1_024, 1_280, 1_536, 2_048, 3_072, 4_096, 6_144, 8_192, 12_288,
    16_384,
];
pub const TOOM6_SIZES: [usize; 25] = [
    128, 160, 255, 256, 257, 260, 264, 300, 320, 384, 512, 544, 768, 1_024, 1_280, 1_536, 2_048,
    3_072, 4_096, 6_144, 8_192, 12_288, 16_384, 24_576, 32_768,
];
pub const TOOM8_SIZES: [usize; 24] = [
    256, 300, 320, 384, 400, 448, 480, 512, 544, 640, 768, 1_024, 1_280, 1_536, 2_048, 3_072,
    4_096, 6_144, 8_192, 12_288, 16_384, 24_576, 32_768, 49_152,
];
pub const TOWER_SIZES: [usize; 87] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 36, 40, 43, 44, 48, 52, 56, 60, 63, 64, 68, 80, 87, 96, 127, 128, 160, 192,
    224, 240, 248, 252, 255, 256, 257, 260, 264, 300, 320, 384, 512, 513, 768, 1_024, 1_280, 1_536,
    2_048, 3_072, 4_096, 6_144, 8_192, 12_288, 16_384, 24_576, 32_768, 49_152, 65_536, 98_304,
    131_072, 196_608, 262_144, 524_288, 1_048_576, 2_097_152, 4_194_304, 8_388_608,
];
pub const ADD_SUB_SIZES: [usize; 6] = [64, 96, 128, 192, 256, 384];
/// Every inner width the ADX basecase specializes, plus the two portable
/// fixed kernels below them and the generic widths just above.
pub const BASECASE_WIDTH_SIZES: [usize; 27] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 32, 33, 40, 48,
    64,
];

/// Row widths for the rectangular schoolbook shape. `karatsuba::mul_forced`
/// hands back any pair whose smaller half would be empty, so a narrow operand
/// against a long row is a production shape, and it is the one the fixed-width
/// table never covered.
pub const BASECASE_ROW_SIZES: [usize; 12] = [8, 12, 16, 17, 18, 20, 24, 32, 40, 48, 56, 64];
pub const BASECASE_CROSSOVER_SIZES: [usize; 41] = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
];
pub const LOWER_CHILD_SIZES: [usize; 27] = [
    56, 60, 61, 62, 63, 64, 65, 66, 68, 72, 76, 80, 81, 82, 83, 84, 85, 86, 87, 88, 92, 96, 120,
    124, 125, 126, 128,
];
pub const LOWER_TOWER_CROSSOVER_SIZES: [usize; 25] = [
    176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248,
    252, 256, 260, 264, 268, 272,
];

/// Debug builds use a 128 Kibit-4 Mibit smoke range on 64-bit targets.
#[cfg(debug_assertions)]
pub const TRANSFORM_SIZES: [usize; 4] = [2_048, 4_096, 16_384, 65_536];
/// Release builds cover the 128 Kibit-288 Mibit transform crossover range on 64-bit targets.
#[cfg(not(debug_assertions))]
pub const TRANSFORM_SIZES: [usize; 25] = [
    384, 512, 768, 1_024, 1_536, 2_048, 4_096, 8_192, 16_384, 24_576, 32_768, 49_152, 65_536,
    98_304, 131_072, 196_608, 262_144, 524_288, 1_048_576, 2_097_152, 3_145_728, 3_670_016,
    4_194_304, 4_718_592, 4_718_593,
];
/// Exact equal-width cells used for repeatable SSA/GMP crossover scorecards.
pub const SSA_SCORECARD_SIZES: [usize; 3] = [4_096, 8_192, 16_384];
/// Full-width Fermat products around the practical transform crossover.
pub const FERMAT_TRANSFORM_SIZES: [usize; 12] = [
    256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 524_288, 1_048_576, 2_097_152, 4_194_304,
    8_388_608,
];
pub const HALF_SIZES: [(usize, usize); 6] = [
    (140, 120),
    (280, 240),
    (560, 480),
    (1_120, 960),
    (2_240, 1_920),
    (4_480, 3_840),
];
pub const TOOM8_HALF_SIZES: [(usize, usize); 8] = [
    (144, 128),
    (288, 256),
    (576, 512),
    (1_152, 1_024),
    (2_304, 2_048),
    (4_608, 4_096),
    (9_216, 8_192),
    (18_432, 16_384),
];

// ============================================================================
// External-library comparison ladders (`compare`)
// ============================================================================

/// Powers of two across the tower's upper half, for log-log scaling fits.
///
/// A regression of `log(time)` on `log(n)` is only well conditioned when the
/// abscissae are evenly spaced in `log(n)`, which `TOWER_SIZES` is not — it
/// clusters points at every measured crossover. These are deliberately uniform
/// so the fitted exponent means something, and they start at 1024 because below
/// that the dispatcher is still in Toom territory and no asymptotic model for
/// the transform applies.
pub const SCALING_SIZES: [usize; 9] = [
    1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,
];

/// Cache-sized SSA planning probes, kept separate from production comparisons
/// because forced-SSA preparation is not meaningful at basecase widths.
pub const SSA_PLANNING_SIZES: [usize; 28] = [
    256, 384, 512, 640, 768, 896, 1_024, 1_280, 1_536, 1_792, 2_048, 2_560, 3_072, 3_584, 4_096,
    5_120, 6_144, 7_168, 8_192, 10_240, 12_288, 14_336, 16_384, 20_480, 24_576, 32_768, 49_152,
    65_536,
];

/// Full production-tower comparison ladder through the cache-sized range.
///
/// Exact neighbors surround the measured schoolbook/Karatsuba, direct
/// balanced-Toom-8, and SSA crossovers. Remaining points keep the ladder dense
/// enough to expose geometry cliffs while preserving powers of two for scaling
/// fits.
pub const PRODUCTION_COMPARE_SIZES: [usize; 63] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 19, 20, 21, 24, 32, 48, 64, 96, 128, 192, 256, 287, 288,
    289, 384, 512, 535, 536, 537, 640, 768, 799, 800, 801, 896, 1_024, 1_280, 1_536, 1_792, 2_048,
    2_560, 2_944, 2_960, 2_975, 2_976, 2_977, 3_072, 3_584, 4_096, 5_120, 6_144, 7_168, 8_192,
    10_240, 12_288, 14_336, 16_384, 20_480, 24_576, 32_768, 49_152, 65_536,
];

/// Continuation of the production comparison through 33,554,432 limbs (2 Gibit).
pub const PRODUCTION_COMPARE_HUGE_SIZES: [usize; 19] = [
    98_304, 131_072, 196_608, 262_144, 393_216, 524_288, 786_432, 1_048_576, 1_572_864, 2_097_152,
    3_145_728, 4_194_304, 6_291_456, 8_388_608, 16_777_216, 20_971_520, 25_165_824, 29_360_128,
    33_554_432,
];

/// Longer width crossed with the seven ratios the shape matrix tracks:
/// 1:1, 4:5, 3:4, 2:3, 1:2, 1:4 and 1:16.
///
/// The odd widths (4091, 16385, 32769) are deliberate: a width one limb off a
/// power of two catches padding and geometry-selection cliffs that exact powers
/// hide.
pub const SHAPES: [(usize, usize); 70] = [
    (200, 200),
    (200, 160),
    (200, 150),
    (200, 133),
    (200, 100),
    (200, 50),
    (200, 12),
    (400, 400),
    (400, 320),
    (400, 300),
    (400, 266),
    (400, 200),
    (400, 100),
    (400, 25),
    (1000, 1000),
    (1000, 800),
    (1000, 750),
    (1000, 666),
    (1000, 500),
    (1000, 250),
    (1000, 62),
    (2000, 2000),
    (2000, 1600),
    (2000, 1500),
    (2000, 1333),
    (2000, 1000),
    (2000, 500),
    (2000, 125),
    (3000, 3000),
    (3000, 2400),
    (3000, 2250),
    (3000, 2000),
    (3000, 1500),
    (3000, 750),
    (3000, 187),
    (4091, 4091),
    (4091, 3272),
    (4091, 3068),
    (4091, 2727),
    (4091, 2045),
    (4091, 1022),
    (4091, 255),
    (6000, 6000),
    (6000, 4800),
    (6000, 4500),
    (6000, 4000),
    (6000, 3000),
    (6000, 1500),
    (6000, 375),
    (10000, 10000),
    (10000, 8000),
    (10000, 7500),
    (10000, 6666),
    (10000, 5000),
    (10000, 2500),
    (10000, 625),
    (16385, 16385),
    (16385, 13108),
    (16385, 12288),
    (16385, 10923),
    (16385, 8192),
    (16385, 4096),
    (16385, 1024),
    (32769, 32769),
    (32769, 26215),
    (32769, 24576),
    (32769, 21846),
    (32769, 16384),
    (32769, 8192),
    (32769, 2048),
];

/// Unbalanced shapes above the transform crossover, where `SHAPES` stops.
///
/// `PLAN.md` reports the widest band as empty for every ratio but 1:16, and
/// that reading rests on 32769 limbs being the widest shape ever measured.
pub const HUGE_SHAPES: [(usize, usize); 12] = [
    (262_144, 262_144),
    (262_144, 196_608),
    (262_144, 131_072),
    (262_144, 16_384),
    (1_048_576, 1_048_576),
    (1_048_576, 786_432),
    (1_048_576, 524_288),
    (1_048_576, 65_536),
    (4_194_304, 4_194_304),
    (4_194_304, 3_145_728),
    (4_194_304, 2_097_152),
    (4_194_304, 262_144),
];
