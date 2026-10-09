//! Score-cell ladders, weights, and deterministic operand constants.

use mp_anafis::tune_api::Limb;

/// Multiplication cells for forced-tier scoring.
#[cfg(not(target_pointer_width = "16"))]
pub const MUL_SCORE_CELLS: [ScoreCell; 11] = [
    ScoreCell::new_balanced(4_096, 64, 15),
    ScoreCell::new_balanced(16_384, 16, 15),
    ScoreCell::new_balanced(65_536, 4, 11),
    ScoreCell::new_balanced(262_144, 2, 7),
    ScoreCell::new_balanced(1_048_576, 1, 5),
    ScoreCell::new_balanced(2_097_152, 1, 3),
    ScoreCell::new_balanced(4_194_304, 1, 3),
    ScoreCell::new_balanced(8_388_608, 1, 3),
    // Unbalanced shapes exercise transform shape policy and blocked fallback.
    ScoreCell::new(32_768, 16_384, 2, 5),
    ScoreCell::new(262_144, 16_384, 1, 5),
    ScoreCell::new(262_144, 8_192, 1, 5),
];

/// Squaring cells for forced-tier scoring.
#[cfg(not(target_pointer_width = "16"))]
pub const SQR_SCORE_CELLS: [ScoreCell; 5] = [
    ScoreCell::new_balanced(4_096, 64, 15),
    ScoreCell::new_balanced(65_536, 4, 11),
    ScoreCell::new_balanced(262_144, 2, 7),
    ScoreCell::new_balanced(1_048_576, 1, 5),
    ScoreCell::new_balanced(4_194_304, 1, 3),
];

/// Balanced cells that directly exercise Toom-8.5 reconstruction choices.
#[cfg(not(target_pointer_width = "16"))]
pub const TOOM85_MUL_SCORE_CELLS: [ScoreCell; 7] = [
    ScoreCell::new_balanced(512, 64, 15),
    ScoreCell::new_balanced(768, 48, 15),
    ScoreCell::new_balanced(1_024, 32, 15),
    ScoreCell::new_balanced(2_048, 16, 11),
    ScoreCell::new_balanced(4_096, 8, 11),
    ScoreCell::new_balanced(6_144, 4, 7),
    ScoreCell::new_balanced(8_192, 4, 7),
];

/// Squaring cells that directly exercise Toom-8.5 reconstruction choices.
#[cfg(not(target_pointer_width = "16"))]
pub const TOOM85_SQR_SCORE_CELLS: [ScoreCell; 5] = [
    ScoreCell::new_balanced(512, 64, 15),
    ScoreCell::new_balanced(1_024, 32, 15),
    ScoreCell::new_balanced(2_048, 16, 11),
    ScoreCell::new_balanced(4_096, 8, 11),
    ScoreCell::new_balanced(8_192, 4, 7),
];

/// Divisor-width ladder for compile-time division recursion constants.
///
/// Widths through 4096 limbs span the declared recursive basecase candidates.
#[cfg(not(target_pointer_width = "16"))]
pub const DIVISION_SCORE_CELLS: [ScoreCell; 7] = [
    ScoreCell::new(128, 64, 64, 15),
    ScoreCell::new(256, 128, 32, 15),
    ScoreCell::new(512, 256, 16, 11),
    ScoreCell::new(1_024, 512, 8, 11),
    ScoreCell::new(2_048, 1_024, 4, 7),
    ScoreCell::new(4_096, 2_048, 2, 7),
    ScoreCell::new(8_192, 4_096, 1, 5),
];

/// Multiplication cells for end-to-end production-dispatch validation.
///
/// Cells below and above transform crossovers exercise both the conventional
/// tower and transform tier, while boundary cells expose incorrect cutoffs.
#[cfg(not(target_pointer_width = "16"))]
pub const PRODUCTION_MUL_CELLS: [ScoreCell; 15] = [
    // Conventional tower neighborhoods (basecase, Karatsuba, Toom-3, Toom-4, Toom-6)
    ScoreCell::new_balanced(32, 256, 15),
    ScoreCell::new_balanced(64, 128, 15),
    ScoreCell::new_balanced(128, 96, 15),
    ScoreCell::new_balanced(256, 64, 15),
    ScoreCell::new_balanced(512, 64, 15),
    ScoreCell::new_balanced(2_048, 16, 15),
    ScoreCell::new_balanced(4_096, 8, 11),
    ScoreCell::new_balanced(16_384, 4, 11),
    ScoreCell::new_balanced(65_536, 2, 7),
    ScoreCell::new_balanced(262_144, 2, 7),
    ScoreCell::new_balanced(1_048_576, 1, 5),
    ScoreCell::new_balanced(4_194_304, 1, 3),
    ScoreCell::new_balanced(8_388_608, 1, 3),
    // Unbalanced shapes exercising transform admission and blocked fallback
    ScoreCell::new(4_096, 256, 16, 11),
    ScoreCell::new(32_768, 512, 4, 7),
];

/// Squaring cells for end-to-end production-dispatch validation.
#[cfg(not(target_pointer_width = "16"))]
pub const PRODUCTION_SQR_CELLS: [ScoreCell; 11] = [
    // Conventional tower neighborhoods (basecase, Karatsuba, Toom-3, Toom-4, Toom-6)
    ScoreCell::new_balanced(32, 256, 15),
    ScoreCell::new_balanced(64, 128, 15),
    ScoreCell::new_balanced(128, 96, 15),
    ScoreCell::new_balanced(256, 64, 15),
    ScoreCell::new_balanced(512, 64, 15),
    ScoreCell::new_balanced(2_048, 16, 15),
    ScoreCell::new_balanced(8_192, 8, 11),
    ScoreCell::new_balanced(65_536, 2, 7),
    ScoreCell::new_balanced(262_144, 2, 7),
    ScoreCell::new_balanced(1_048_576, 1, 5),
    ScoreCell::new_balanced(4_194_304, 1, 3),
];

/// Transform admission shapes plus balanced guards near the crossover.
///
/// Each unbalanced column holds one aspect ratio (8, 16, 32, 64, 128) while the
/// shorter operand sweeps 256 to 2,048 limbs, so the grid straddles the
/// candidate admission floor (including 256) and ratio cap (up to 128) from
/// both sides.
#[cfg(not(target_pointer_width = "16"))]
pub const TRANSFORM_SHAPE_CELLS: [ScoreCell; 39] = [
    // Balanced bypass policies are measured independently of the ascending tower.
    ScoreCell::new_balanced(64, 32, 5),
    ScoreCell::new_balanced(128, 32, 5),
    ScoreCell::new_balanced(191, 16, 5),
    ScoreCell::new_balanced(192, 16, 5),
    ScoreCell::new_balanced(255, 16, 5),
    ScoreCell::new_balanced(256, 16, 5),
    ScoreCell::new_balanced(313, 16, 5),
    ScoreCell::new_balanced(314, 16, 5),
    ScoreCell::new_balanced(315, 16, 5),
    ScoreCell::new_balanced(319, 16, 5),
    ScoreCell::new_balanced(320, 16, 5),
    ScoreCell::new_balanced(383, 16, 5),
    ScoreCell::new_balanced(384, 16, 5),
    ScoreCell::new_balanced(511, 8, 5),
    ScoreCell::new_balanced(512, 8, 5),
    ScoreCell::new_balanced(767, 8, 5),
    ScoreCell::new_balanced(768, 8, 5),
    ScoreCell::new_balanced(1_023, 8, 5),
    ScoreCell::new_balanced(1_024, 8, 5),
    // Shorter operand 256 (sweeping ratio 8, 16, 32, 64, 128)
    ScoreCell::new(2_048, 256, 8, 5),
    ScoreCell::new(4_096, 256, 4, 5),
    ScoreCell::new(8_192, 256, 4, 5),
    ScoreCell::new(16_384, 256, 2, 5),
    ScoreCell::new(32_768, 256, 2, 5),
    // Shorter operand 512 (sweeping ratio 8, 16, 32, 64, 128)
    ScoreCell::new(4_096, 512, 4, 5),
    ScoreCell::new(8_192, 512, 4, 5),
    ScoreCell::new(16_384, 512, 2, 5),
    ScoreCell::new(32_768, 512, 2, 5),
    ScoreCell::new(65_536, 512, 1, 5),
    // Shorter operand 1,100 (ratio 8, 16, 32)
    ScoreCell::new(8_800, 1_100, 4, 5),
    ScoreCell::new(17_600, 1_100, 2, 5),
    ScoreCell::new(35_200, 1_100, 2, 5),
    // Shorter operand 2,048 (ratio 8, 16, 32)
    ScoreCell::new(16_384, 2_048, 2, 5),
    ScoreCell::new(32_768, 2_048, 2, 5),
    ScoreCell::new(65_536, 2_048, 1, 5),
    // Shorter operand 1,024 (ratio 64, 128)
    ScoreCell::new(65_536, 1_024, 1, 5),
    ScoreCell::new(131_072, 1_024, 1, 3),
    // Balanced guards inside the crossover neighborhood
    ScoreCell::new_balanced(2_048, 8, 5),
    ScoreCell::new_balanced(4_096, 4, 5),
];

/// Operand shape, batch iterations, and sample count for one measurement cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScoreCell {
    /// Left or first operand length in limbs.
    pub len_a: usize,
    /// Right or second operand length in limbs.
    pub len_b: usize,
    /// Inner repetitions per timed batch.
    pub iterations: u32,
    /// Number of repeat batch samples taken.
    pub samples: usize,
}

impl ScoreCell {
    /// Construct a general unbalanced or balanced cell.
    #[must_use]
    pub const fn new(len_a: usize, len_b: usize, iterations: u32, samples: usize) -> Self {
        Self {
            len_a,
            len_b,
            iterations,
            samples,
        }
    }

    /// Construct a balanced square or multiplication cell (`len_a == len_b`).
    #[must_use]
    pub const fn new_balanced(len: usize, iterations: u32, samples: usize) -> Self {
        Self::new(len, len, iterations, samples)
    }

    /// Preserve every width and iteration count while reducing repeat samples.
    #[must_use]
    pub const fn coarse(self) -> Self {
        Self {
            samples: if self.samples > 3 { 3 } else { 1 },
            ..self
        }
    }

    /// The wider operand, which bounds the transform ring.
    #[must_use]
    pub const fn larger(self) -> usize {
        if self.len_a > self.len_b {
            self.len_a
        } else {
            self.len_b
        }
    }

    /// Per-cell logarithmic weights over concatenated multiplication and square cells.
    #[must_use]
    pub fn cell_weights(mul_cells: &[Self], sqr_cells: &[Self]) -> Vec<u32> {
        mul_cells
            .iter()
            .chain(sqr_cells)
            .map(|cell| cell.larger().ilog2().max(1))
            .collect()
    }

    /// Deterministic nonzero operand limbs shared by every worker domain.
    #[must_use]
    pub fn operand(len: usize, hash: Limb) -> Vec<Limb> {
        (0..len).map(|index| index.wrapping_mul(hash) | 1).collect()
    }
}

/// Deterministic hash seed A.
#[cfg(target_pointer_width = "64")]
pub const HASH_A: Limb = 0x9E37_79B9_7F4A_7C15;
/// Deterministic hash seed B.
#[cfg(target_pointer_width = "64")]
pub const HASH_B: Limb = 0xC2B2_AE3D_27D4_EB4F;
/// Deterministic hash seed A.
#[cfg(target_pointer_width = "32")]
pub const HASH_A: Limb = 0x9E37_79B9;
/// Deterministic hash seed B.
#[cfg(target_pointer_width = "32")]
pub const HASH_B: Limb = 0xC2B2_AE3D;
/// Deterministic hash seed A.
#[cfg(target_pointer_width = "16")]
pub const HASH_A: Limb = 0x9E37;
/// Deterministic hash seed B.
#[cfg(target_pointer_width = "16")]
pub const HASH_B: Limb = 0xC2B2;
