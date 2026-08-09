//! Where the SSA pointwise stage should leave the tower for a nested transform.
//!
//! `SSA_BASE_MODULUS_BITS` is the widest inner ring handled by the
//! multiplication tower; above it the pointwise stage nests another transform.
//! It was last swept before the inner-ring rounding fix, which cut the cost of
//! being on the nested side by up to a factor of two, so the crossover it
//! encodes almost certainly moved.
//!
//! This forces the transform across balanced widths spanning the range where the
//! nested branch is reachable. It is meant to be run once per candidate value
//! through `MP_TUNING_PROFILE`, because the constant is compiled in.

use core::hint::black_box;

use mp_anafis::tune_api::tier::{
    Limb,
    transform::{bench_ssa_mul_scratch_len, bench_ssa_mul_with_scratch},
};

use crate::shared::operands_pair;

const WIDTHS: [usize; 6] = [8_192, 32_769, 131_073, 262_145, 524_289, 1_048_577];

#[divan::bench(args = WIDTHS)]
fn forced_transform(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (larger, smaller, mut destination) = operands_pair(len, len);
    let mut scratch = vec![Limb::MIN; bench_ssa_mul_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_ssa_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(&mut scratch),
        );
    });
}
