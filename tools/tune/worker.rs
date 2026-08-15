//! Hidden workers that score one profile in a subprocess.
//!
//! Worker modes time forced SSA, Toom-8.5, or division domains for compiled
//! constants. A separate production worker scores the configured dispatcher
//! over the final validation ladder.

use core::hint::black_box;

use mp_anafis::tune_api::{
    Limb, division, formatting, multiplication, squaring,
    tier::state::{MulBenchScratch, SquareBenchScratch},
};

use crate::measure::{ProbeQuality, balanced_batch, median_batch_samples, median_pair_batches};

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
    // Unbalanced shapes exercise the transform's shape policy and the blocked
    // fallback rather than a second balanced copy of the same work.
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
/// These cells deliberately stop at 4096 divisor limbs: unlike SSA, division
/// has no RAM-sized geometry whose behavior appears only millions of limbs
/// later. The ladder crosses every candidate base block and reciprocal
/// basecase several times while keeping one full score inexpensive.
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

/// Multiplication cells for the end-to-end production-dispatch validation.
///
/// The ladder deliberately crosses the transform crossovers: cells below the
/// tuned `SSA_THRESHOLD` exercise the conventional tower, cells above it
/// exercise the transform tier, and the boundary cells sit where a wrong
/// crossover loses the most.
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

/// Division cells for the final production-dispatch validation.
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

    /// Preserve every size and iteration count while reducing repeat samples.
    const fn coarse(self) -> Self {
        Self {
            samples: if self.samples > 3 { 3 } else { 1 },
            ..self
        }
    }

    /// The wider operand, which bounds the transform ring.
    #[must_use]
    pub const fn larger(&self) -> usize {
        if self.len_a > self.len_b {
            self.len_a
        } else {
            self.len_b
        }
    }
}

/// Per-cell weights over the concatenated score-cell lists.
///
/// A cell contributes to the weighted mean in proportion to the logarithm of
/// its wider operand, so a RAM-sized product outweighs a cache-sized one
/// without letting either dominate.
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

/// Weights for only the cells whose top-level ring exactly matches a pin.
#[cfg(not(target_pointer_width = "16"))]
pub fn ring_cell_weights(ring_bits: usize) -> Vec<u32> {
    MUL_SCORE_CELLS
        .iter()
        .filter(|cell| ring_affects_mul_cell(ring_bits, cell))
        .chain(
            SQR_SCORE_CELLS
                .iter()
                .filter(|cell| ring_affects_sqr_cell(ring_bits, cell)),
        )
        .map(|cell| cell.larger().ilog2())
        .collect()
}

/// Print forced-SSA timings for the compiled-parameter worker.
pub fn print_ssa_score() {
    print_ssa_score_with(false);
}

/// Print reduced-sample forced-SSA timings for candidate screening.
pub fn print_ssa_coarse_score() {
    print_ssa_score_with(true);
}

fn print_ssa_score_with(coarse: bool) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let mut values =
            Vec::with_capacity(MUL_SCORE_CELLS.len().wrapping_add(SQR_SCORE_CELLS.len()));
        for cell in MUL_SCORE_CELLS {
            values.push(score_mul_cell(if coarse { cell.coarse() } else { cell }));
        }
        for cell in SQR_SCORE_CELLS {
            values.push(score_sqr_cell(if coarse { cell.coarse() } else { cell }));
        }
        let prefix = if coarse {
            "MP_ANAFIS_SSA_COARSE_SCORE="
        } else {
            "MP_ANAFIS_SSA_SCORE="
        };
        print_encoded(prefix, &values);
    }
    #[cfg(target_pointer_width = "16")]
    if coarse {
        println!("MP_ANAFIS_SSA_COARSE_SCORE=");
    } else {
        println!("MP_ANAFIS_SSA_SCORE=");
    }
}

/// Print only the SSA score cells reachable from one geometry ring pin.
pub fn print_ssa_ring_score(ring_bits: usize, coarse: bool) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let mut values = Vec::new();
        for cell in MUL_SCORE_CELLS {
            if ring_affects_mul_cell(ring_bits, &cell) {
                values.push(score_mul_cell(if coarse { cell.coarse() } else { cell }));
            }
        }
        for cell in SQR_SCORE_CELLS {
            if ring_affects_sqr_cell(ring_bits, &cell) {
                values.push(score_sqr_cell(if coarse { cell.coarse() } else { cell }));
            }
        }
        print_encoded("MP_ANAFIS_SSA_RING_SCORE=", &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("MP_ANAFIS_SSA_RING_SCORE=");
}

#[cfg(not(target_pointer_width = "16"))]
const fn ring_affects_mul_cell(ring_bits: usize, cell: &ScoreCell) -> bool {
    let limb_bits = core::mem::size_of::<usize>().wrapping_mul(8);
    let half_width = cell.len_a.wrapping_add(cell.len_b).wrapping_div(2);
    let cell_ring = half_width.wrapping_mul(limb_bits);
    cell_ring == ring_bits
}

#[cfg(not(target_pointer_width = "16"))]
const fn ring_affects_sqr_cell(ring_bits: usize, cell: &ScoreCell) -> bool {
    let limb_bits = core::mem::size_of::<usize>().wrapping_mul(8);
    let cell_ring = cell.larger().wrapping_mul(limb_bits);
    cell_ring == ring_bits
}

/// Print direct forced-Toom-8.5 timings for its reconstruction knob.
pub fn print_toom85_score() {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let mut values = Vec::with_capacity(
            TOOM85_MUL_SCORE_CELLS
                .len()
                .wrapping_add(TOOM85_SQR_SCORE_CELLS.len()),
        );
        for cell in TOOM85_MUL_SCORE_CELLS {
            values.push(score_forced_mul_cell(
                cell,
                multiplication::Algorithm::ToomCook85,
            ));
        }
        for cell in TOOM85_SQR_SCORE_CELLS {
            values.push(score_forced_sqr_cell(cell, squaring::Algorithm::ToomCook85));
        }
        print_encoded("MP_ANAFIS_TOOM85_SCORE=", &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("MP_ANAFIS_TOOM85_SCORE=");
}

/// Print forced Toom-8.5 multiplication timings for multiplication-only knobs.
pub fn print_toom85_mul_score() {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let values = TOOM85_MUL_SCORE_CELLS
            .map(|cell| score_forced_mul_cell(cell, multiplication::Algorithm::ToomCook85));
        print_encoded("MP_ANAFIS_TOOM85_MUL_SCORE=", &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("MP_ANAFIS_TOOM85_MUL_SCORE=");
}

/// Print forced Burnikel-Ziegler timings for its compiled block-size tuner.
pub fn print_burnikel_score() {
    print_division_score(
        "MP_ANAFIS_BURNIKEL_SCORE=",
        division::Algorithm::BurnikelZiegler,
    );
}

/// Print forced Newton-Raphson timings for its compiled basecase tuner.
pub fn print_newton_score() {
    print_division_score(
        "MP_ANAFIS_NEWTON_SCORE=",
        division::Algorithm::NewtonRaphson,
    );
}

/// Print production-division timings for coupled threshold tuning.
pub fn print_production_division_score() {
    print_division_score(
        "MP_ANAFIS_PRODUCTION_DIVISION_SCORE=",
        division::Algorithm::Production,
    );
}

/// Print production-dispatch timings for the end-to-end validation worker.
pub fn print_production_score() {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let mut values = Vec::with_capacity(
            PRODUCTION_MUL_CELLS
                .len()
                .wrapping_add(PRODUCTION_SQR_CELLS.len())
                .wrapping_add(PRODUCTION_DIV_CELLS.len()),
        );
        for cell in PRODUCTION_MUL_CELLS {
            values.push(score_production_mul_cell(cell));
        }
        for cell in PRODUCTION_SQR_CELLS {
            values.push(score_production_sqr_cell(cell));
        }
        for cell in PRODUCTION_DIV_CELLS {
            values.push(score_division_cell(cell, division::Algorithm::Production));
        }
        print_encoded("MP_ANAFIS_PRODUCTION_SCORE=", &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("MP_ANAFIS_PRODUCTION_SCORE=");
}

/// Compare two forced multiplication roots while the candidate profile is
/// compiled into every recursively dispatched child.
pub fn print_mul_pair_score(specification: &str) -> Result<(), String> {
    let PairSpecification {
        baseline: baseline_name,
        candidate: candidate_name,
        len,
        quality,
        iterations,
    } = PairSpecification::parse(specification)?;
    let baseline_algorithm = multiplication_algorithm(baseline_name)?;
    let candidate_algorithm = multiplication_algorithm(candidate_name)?;
    let left: Vec<Limb> = (0..len)
        .map(|index| index.wrapping_mul(hash_a()) | 1)
        .collect();
    let right: Vec<Limb> = (0..len)
        .map(|index| index.wrapping_mul(hash_b()) | 1)
        .collect();
    let mut baseline_dst = vec![0; len.wrapping_mul(2)];
    let mut candidate_dst = vec![0; len.wrapping_mul(2)];
    let mut baseline_runner = multiplication::Tuner::new(baseline_algorithm, len, len);
    let mut candidate_runner = multiplication::Tuner::new(candidate_algorithm, len, len);
    baseline_runner.run(&mut baseline_dst, &left, &right);
    candidate_runner.run(&mut candidate_dst, &left, &right);
    if baseline_dst != candidate_dst {
        return Err(format!("multiplication candidates disagree at {len} limbs"));
    }
    let batch = quality.batch(balanced_batch(iterations, len));
    let (baseline_time, candidate_time) = median_pair_batches(
        || {
            baseline_runner.run(
                black_box(&mut baseline_dst),
                black_box(&left),
                black_box(&right),
            );
        },
        || {
            candidate_runner.run(
                black_box(&mut candidate_dst),
                black_box(&left),
                black_box(&right),
            );
        },
        batch,
        quality.samples(),
    );
    println!(
        "MP_ANAFIS_TIER_PAIR={},{}",
        baseline_time.as_nanos(),
        candidate_time.as_nanos()
    );
    Ok(())
}

/// Compare two forced square roots with recursively consistent thresholds.
pub fn print_sqr_pair_score(specification: &str) -> Result<(), String> {
    let PairSpecification {
        baseline: baseline_name,
        candidate: candidate_name,
        len,
        quality,
        iterations,
    } = PairSpecification::parse(specification)?;
    let baseline_algorithm = squaring_algorithm(baseline_name)?;
    let candidate_algorithm = squaring_algorithm(candidate_name)?;
    let value: Vec<Limb> = (0..len)
        .map(|index| index.wrapping_mul(hash_a()) | 1)
        .collect();
    let mut baseline_dst = vec![0; len.wrapping_mul(2)];
    let mut candidate_dst = vec![0; len.wrapping_mul(2)];
    let mut baseline_runner = squaring::Tuner::new(baseline_algorithm, len);
    let mut candidate_runner = squaring::Tuner::new(candidate_algorithm, len);
    baseline_runner.run(&mut baseline_dst, &value);
    candidate_runner.run(&mut candidate_dst, &value);
    if baseline_dst != candidate_dst {
        return Err(format!("squaring candidates disagree at {len} limbs"));
    }
    let batch = quality.batch(balanced_batch(iterations, len));
    let (baseline_time, candidate_time) = median_pair_batches(
        || {
            baseline_runner.run(black_box(&mut baseline_dst), black_box(&value));
        },
        || {
            candidate_runner.run(black_box(&mut candidate_dst), black_box(&value));
        },
        batch,
        quality.samples(),
    );
    println!(
        "MP_ANAFIS_TIER_PAIR={},{}",
        baseline_time.as_nanos(),
        candidate_time.as_nanos()
    );
    Ok(())
}

/// Compare forced schoolbook and recursive formatting tiers.
pub fn print_fmt_pair_score(specification: &str) -> Result<(), String> {
    let FormattingPairSpecification {
        baseline: baseline_name,
        candidate: candidate_name,
        radix,
        len,
        quality,
        iterations,
    } = FormattingPairSpecification::parse(specification)?;
    let baseline_algorithm = formatting_algorithm(baseline_name)?;
    let candidate_algorithm = formatting_algorithm(candidate_name)?;
    let mut baseline_runner = formatting::Tuner::new(baseline_algorithm, len, radix);
    let mut candidate_runner = formatting::Tuner::new(candidate_algorithm, len, radix);
    let baseline_output = baseline_runner.output();
    let candidate_output = candidate_runner.output();
    if baseline_output != candidate_output {
        return Err(format!(
            "formatting candidates disagree at {len} limbs: schoolbook={}, recursive={}",
            baseline_output.len(),
            candidate_output.len()
        ));
    }
    let batch = quality.batch(balanced_batch(iterations, len));
    let (baseline_time, candidate_time) = median_pair_batches(
        || baseline_runner.run(),
        || candidate_runner.run(),
        batch,
        quality.samples(),
    );
    println!(
        "MP_ANAFIS_FMT_PAIR={},{}",
        baseline_time.as_nanos(),
        candidate_time.as_nanos()
    );
    Ok(())
}

struct PairSpecification<'specification> {
    baseline: &'specification str,
    candidate: &'specification str,
    len: usize,
    quality: ProbeQuality,
    iterations: u32,
}

impl<'specification> PairSpecification<'specification> {
    fn parse(specification: &'specification str) -> Result<Self, String> {
        let mut fields = specification.split(',');
        let baseline = fields.next().ok_or("missing baseline tier")?;
        let candidate = fields.next().ok_or("missing candidate tier")?;
        let len = fields
            .next()
            .ok_or("missing tier width")?
            .parse::<usize>()
            .map_err(|error| format!("invalid tier width: {error}"))?;
        let quality = match fields.next() {
            Some("coarse") => ProbeQuality::Coarse,
            Some("precise") => ProbeQuality::Precise,
            _ => return Err("tier quality must be coarse or precise".to_owned()),
        };
        let iterations = fields
            .next()
            .ok_or("missing tier iterations")?
            .parse::<u32>()
            .map_err(|error| format!("invalid tier iterations: {error}"))?;
        if fields.next().is_some() || len == 0 || iterations == 0 {
            return Err("invalid tier-pair specification".to_owned());
        }
        Ok(Self {
            baseline,
            candidate,
            len,
            quality,
            iterations,
        })
    }
}

struct FormattingPairSpecification<'specification> {
    baseline: &'specification str,
    candidate: &'specification str,
    radix: u32,
    len: usize,
    quality: ProbeQuality,
    iterations: u32,
}

impl<'specification> FormattingPairSpecification<'specification> {
    fn parse(specification: &'specification str) -> Result<Self, String> {
        let mut fields = specification.split(',');
        let baseline = fields.next().ok_or("missing baseline tier")?;
        let candidate = fields.next().ok_or("missing candidate tier")?;
        let radix = fields
            .next()
            .ok_or("missing formatting radix")?
            .parse::<u32>()
            .map_err(|error| format!("invalid formatting radix: {error}"))?;
        let len = fields
            .next()
            .ok_or("missing tier width")?
            .parse::<usize>()
            .map_err(|error| format!("invalid tier width: {error}"))?;
        let quality = match fields.next() {
            Some("coarse") => ProbeQuality::Coarse,
            Some("precise") => ProbeQuality::Precise,
            _ => return Err("tier quality must be coarse or precise".to_owned()),
        };
        let iterations = fields
            .next()
            .ok_or("missing tier iterations")?
            .parse::<u32>()
            .map_err(|error| format!("invalid tier iterations: {error}"))?;
        if fields.next().is_some()
            || len == 0
            || iterations == 0
            || !(3..=36).contains(&radix)
            || radix.is_power_of_two()
        {
            return Err("invalid formatting tier-pair specification".to_owned());
        }
        Ok(Self {
            baseline,
            candidate,
            radix,
            len,
            quality,
            iterations,
        })
    }
}

fn multiplication_algorithm(name: &str) -> Result<multiplication::Algorithm, String> {
    match name {
        "schoolbook" => Ok(multiplication::Algorithm::Schoolbook),
        "karatsuba" => Ok(multiplication::Algorithm::Karatsuba),
        "toom3" => Ok(multiplication::Algorithm::ToomCook3),
        "toom4" => Ok(multiplication::Algorithm::ToomCook4),
        "toom6" => Ok(multiplication::Algorithm::ToomCook6),
        "toom85" => Ok(multiplication::Algorithm::ToomCook85),
        #[cfg(not(target_pointer_width = "16"))]
        "ssa" => Ok(multiplication::Algorithm::Ssa),
        _ => Err(format!("unknown multiplication tier {name}")),
    }
}

fn squaring_algorithm(name: &str) -> Result<squaring::Algorithm, String> {
    match name {
        "schoolbook" => Ok(squaring::Algorithm::Schoolbook),
        "karatsuba" => Ok(squaring::Algorithm::Karatsuba),
        "toom3" => Ok(squaring::Algorithm::ToomCook3),
        "toom4" => Ok(squaring::Algorithm::ToomCook4),
        "toom6" => Ok(squaring::Algorithm::ToomCook6),
        "toom85" => Ok(squaring::Algorithm::ToomCook85),
        #[cfg(not(target_pointer_width = "16"))]
        "ssa" => Ok(squaring::Algorithm::Ssa),
        _ => Err(format!("unknown squaring tier {name}")),
    }
}

fn formatting_algorithm(name: &str) -> Result<formatting::Algorithm, String> {
    match name {
        "schoolbook" => Ok(formatting::Algorithm::Schoolbook),
        "recursive" => Ok(formatting::Algorithm::Recursive),
        _ => Err(format!("unknown formatting tier {name}")),
    }
}

#[cfg(not(target_pointer_width = "16"))]
fn print_encoded(prefix: &str, values: &[u128]) {
    let encoded = values
        .iter()
        .map(u128::to_string)
        .collect::<Vec<_>>()
        .join(",");
    println!("{prefix}{encoded}");
}

fn print_division_score(prefix: &str, algorithm: division::Algorithm) {
    #[cfg(not(target_pointer_width = "16"))]
    {
        let values = DIVISION_SCORE_CELLS.map(|cell| score_division_cell(cell, algorithm));
        print_encoded(prefix, &values);
    }
    #[cfg(target_pointer_width = "16")]
    println!("{prefix}");
}

#[cfg(not(target_pointer_width = "16"))]
fn score_mul_cell(cell: ScoreCell) -> u128 {
    score_forced_mul_cell(cell, multiplication::Algorithm::Ssa)
}

#[cfg(not(target_pointer_width = "16"))]
fn score_forced_mul_cell(cell: ScoreCell, algorithm: multiplication::Algorithm) -> u128 {
    let left: Vec<Limb> = (0..cell.len_a)
        .map(|index| index.wrapping_mul(hash_a()) | 1)
        .collect();
    let right: Vec<Limb> = (0..cell.len_b)
        .map(|index| index.wrapping_mul(hash_b()) | 1)
        .collect();
    let mut destination = vec![0; cell.len_a.wrapping_add(cell.len_b)];
    let mut runner = multiplication::Tuner::new(algorithm, cell.len_a, cell.len_b);
    runner.run(&mut destination, &left, &right);
    median_batch_samples(
        || {
            runner.run(
                black_box(&mut destination),
                black_box(&left),
                black_box(&right),
            );
        },
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_sqr_cell(cell: ScoreCell) -> u128 {
    score_forced_sqr_cell(cell, squaring::Algorithm::Ssa)
}

#[cfg(not(target_pointer_width = "16"))]
fn score_forced_sqr_cell(cell: ScoreCell, algorithm: squaring::Algorithm) -> u128 {
    let value: Vec<Limb> = (0..cell.len_a)
        .map(|index| index.wrapping_mul(hash_a()) | 1)
        .collect();
    let mut destination = vec![0; cell.len_a.wrapping_mul(2)];
    let mut runner = squaring::Tuner::new(algorithm, cell.len_a);
    runner.run(&mut destination, &value);
    median_batch_samples(
        || {
            runner.run(black_box(&mut destination), black_box(&value));
        },
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_production_mul_cell(cell: ScoreCell) -> u128 {
    let left: Vec<Limb> = (0..cell.len_a)
        .map(|index| index.wrapping_mul(hash_a()) | 1)
        .collect();
    let right: Vec<Limb> = (0..cell.len_b)
        .map(|index| index.wrapping_mul(hash_b()) | 1)
        .collect();
    let mut destination = vec![0; cell.len_a.wrapping_add(cell.len_b)];
    let mut runner = MulBenchScratch::default();
    runner.run(&mut destination, &left, &right);
    median_batch_samples(
        || {
            runner.run(
                black_box(&mut destination),
                black_box(&left),
                black_box(&right),
            );
        },
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_production_sqr_cell(cell: ScoreCell) -> u128 {
    let value: Vec<Limb> = (0..cell.len_a)
        .map(|index| index.wrapping_mul(hash_a()) | 1)
        .collect();
    let mut destination = vec![0; cell.len_a.wrapping_mul(2)];
    let mut runner = SquareBenchScratch::default();
    runner.run(&mut destination, &value);
    median_batch_samples(
        || {
            runner.run(black_box(&mut destination), black_box(&value));
        },
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(not(target_pointer_width = "16"))]
fn score_division_cell(cell: ScoreCell, algorithm: division::Algorithm) -> u128 {
    let numerator: Vec<Limb> = (0..cell.len_a)
        .map(|index| index.wrapping_mul(hash_a()) | 1)
        .collect();
    let denominator: Vec<Limb> = (0..cell.len_b)
        .map(|index| index.wrapping_mul(hash_b()) | 1)
        .collect();
    let mut reference = division::Tuner::new(&numerator, &denominator);
    let mut runner = division::Tuner::new(&numerator, &denominator);
    reference.run(division::Algorithm::AlgorithmD);
    runner.run(algorithm);
    assert_eq!(
        runner.quotient_limbs(),
        reference.quotient_limbs(),
        "forced division candidate produced a different quotient"
    );
    assert_eq!(
        runner.remainder_limbs(),
        reference.remainder_limbs(),
        "forced division candidate produced a different remainder"
    );
    median_batch_samples(
        || black_box(&mut runner).run(algorithm),
        cell.iterations,
        cell.samples,
    )
    .as_nanos()
    .div_euclid(u128::from(cell.iterations))
}

#[cfg(all(not(target_pointer_width = "16"), target_pointer_width = "64"))]
const fn hash_a() -> Limb {
    0x9E37_79B9_7F4A_7C15
}

#[cfg(all(not(target_pointer_width = "16"), target_pointer_width = "64"))]
const fn hash_b() -> Limb {
    0xC2B2_AE3D_27D4_EB4F
}

#[cfg(all(not(target_pointer_width = "16"), target_pointer_width = "32"))]
const fn hash_a() -> Limb {
    0x9E37_79B9
}

#[cfg(all(not(target_pointer_width = "16"), target_pointer_width = "32"))]
const fn hash_b() -> Limb {
    0xC2B2_AE3D
}
