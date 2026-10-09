//! Shared batch lengths and declared Divan sampling settings.
//! These settings control execution; they do not establish timing uncertainty.

/// Operands per generated batch.
///
/// Benchmarks that loop over a batch report the aggregate duration for all [`SAMPLES`]
/// operations, maintaining parity across paired `mp` and `rug` evaluations.
pub const SAMPLES: u32 = 10;

/// Iterations per sample for fast-operation cases.
pub const SAMPLE_SIZE_FAST: u32 = 256;

/// Samples for fast comparison cases, matching Divan's default count.
pub const SAMPLE_COUNT_FAST: u32 = 100;

/// Iterations per sample for the default paired operation cases.
pub const SAMPLE_SIZE_WIDE: u32 = 32;

/// Samples for the default paired operation cases.
pub const SAMPLE_COUNT_WIDE: u32 = 50;

/// Iterations per sample for modular exponentiation and other costly cases.
pub const SAMPLE_SIZE_HEAVY: u32 = 4;

/// Samples for modular exponentiation and other costly cases.
pub const SAMPLE_COUNT_HEAVY: u32 = 20;

/// Sample size for complete deterministic GCD batches.
pub const SAMPLE_SIZE_GCD: u32 = 1;

/// Sample count for GCD-family operations and extended GCD evaluations.
pub const SAMPLE_COUNT_GCD: u32 = 15;
