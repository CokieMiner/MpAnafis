//! Native-limb NTT and production-tower comparisons with reusable storage.

#[cfg(feature = "rayon")]
use mp_ntt::RayonExecutor as Executor;
#[cfg(not(feature = "rayon"))]
use mp_ntt::SequentialExecutor as Executor;

mod inputs;
mod multiplication;
mod squaring;

use inputs::Inputs;

const SIZES: &[usize] = &[
    128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288,
    1_048_576, 2_097_152,
];

fn main() {
    divan::main();
}
