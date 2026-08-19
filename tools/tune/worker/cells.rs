//! Score-cell ladders, weights, and deterministic operand constants.

use mp_anafis::tune_api::Limb;

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
/// These cells stop at 4096 divisor limbs because division has no RAM-sized
/// geometry whose behavior appears only millions of limbs later. The ladder
/// crosses every candidate base block and reciprocal basecase several times.
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
pub const PRODUCTION_MUL_CELLS: [ScoreCell; 9] = [
    ScoreCell::new_balanced(512, 64, 15),
    ScoreCell::new_balanced(2_048, 16, 15),
    ScoreCell::new_balanced(4_096, 8, 11),
    ScoreCell::new_balanced(16_384, 4, 11),
    ScoreCell::new_balanced(65_536, 2, 7),
    ScoreCell::new_balanced(262_144, 2, 7),
    ScoreCell::new_balanced(1_048_576, 1, 5),
    ScoreCell::new_balanced(4_194_304, 1, 3),
    ScoreCell::new_balanced(8_388_608, 1, 3),
];

#[cfg(not(target_pointer_width = "16"))]
pub const PRODUCTION_SQR_CELLS: [ScoreCell; 7] = [
    ScoreCell::new_balanced(512, 64, 15),
    ScoreCell::new_balanced(2_048, 16, 15),
    ScoreCell::new_balanced(8_192, 8, 11),
    ScoreCell::new_balanced(65_536, 2, 7),
    ScoreCell::new_balanced(262_144, 2, 7),
    ScoreCell::new_balanced(1_048_576, 1, 5),
    ScoreCell::new_balanced(4_194_304, 1, 3),
];

/// Division cells for final production-dispatch validation.
#[cfg(not(target_pointer_width = "16"))]
pub const PRODUCTION_DIV_CELLS: [ScoreCell; 7] = DIVISION_SCORE_CELLS;

#[cfg(not(target_pointer_width = "16"))]
#[derive(Clone, Copy)]
pub struct ScoreCell {
    pub len_a: usize,
    pub len_b: usize,
    pub iterations: u32,
    pub samples: usize,
}

#[cfg(not(target_pointer_width = "16"))]
impl ScoreCell {
    const fn new(len_a: usize, len_b: usize, iterations: u32, samples: usize) -> Self {
        Self {
            len_a,
            len_b,
            iterations,
            samples,
        }
    }

    const fn new_balanced(len: usize, iterations: u32, samples: usize) -> Self {
        Self::new(len, len, iterations, samples)
    }

    /// Preserve every width and iteration count while reducing repeat samples.
    pub const fn coarse(self) -> Self {
        Self {
            samples: if self.samples > 3 { 3 } else { 1 },
            ..self
        }
    }

    /// The wider operand, which bounds the transform ring.
    pub const fn larger(self) -> usize {
        if self.len_a > self.len_b {
            self.len_a
        } else {
            self.len_b
        }
    }
}

/// Per-cell logarithmic weights over concatenated multiplication and square cells.
#[cfg(not(target_pointer_width = "16"))]
pub fn cell_weights(mul_cells: &[ScoreCell], sqr_cells: &[ScoreCell]) -> Vec<u32> {
    mul_cells
        .iter()
        .chain(sqr_cells)
        .map(|cell| cell.larger().ilog2())
        .collect()
}

/// Logarithmic weights for the division recursion ladder.
#[cfg(not(target_pointer_width = "16"))]
pub fn division_cell_weights() -> Vec<u32> {
    DIVISION_SCORE_CELLS
        .iter()
        .map(|cell| cell.len_b.ilog2())
        .collect()
}

/// Deterministic nonzero operand limbs shared by every worker domain.
pub fn operand(len: usize, hash: Limb) -> Vec<Limb> {
    (0..len).map(|index| index.wrapping_mul(hash) | 1).collect()
}

#[cfg(target_pointer_width = "64")]
pub const HASH_A: Limb = 0x9E37_79B9_7F4A_7C15;
#[cfg(target_pointer_width = "64")]
pub const HASH_B: Limb = 0xC2B2_AE3D_27D4_EB4F;
#[cfg(target_pointer_width = "32")]
pub const HASH_A: Limb = 0x9E37_79B9;
#[cfg(target_pointer_width = "32")]
pub const HASH_B: Limb = 0xC2B2_AE3D;
#[cfg(target_pointer_width = "16")]
pub const HASH_A: Limb = 0x9E37;
#[cfg(target_pointer_width = "16")]
pub const HASH_B: Limb = 0xC2B2;
