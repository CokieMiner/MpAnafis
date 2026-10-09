//! Production, forced SSA, direct Fermat, and GMP squares on one width ladder.
//! Production rows retain dispatch and planning inside timing. Forced rows
//! prepare their selected root algorithm before timing and reuse its state.

#![expect(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_sqr with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t};
use mp_anafis::tune_api::{Limb, SquaringAlgorithm, SquaringBenchState, SquaringRunner};

use crate::shared::{
    assert_gmp_limb_width, gmp_equal_reference, operand, validate_and_warm_product,
    validated_gmp_count,
};

const WIDTHS: [usize; 9] = [1800, 2000, 2200, 2400, 2500, 2600, 2800, 3000, 3072];

#[divan::bench(args = WIDTHS)]
fn dispatched(bencher: divan::Bencher<'_, '_>, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let expected = gmp_equal_reference(&value, &value);
    let mut destination = vec![Limb::MIN; expected.len()];
    let mut reusable = SquaringBenchState::default();
    validate_and_warm_product(&expected, "production square", |probe| {
        reusable.prepare(probe, &value).run();
    });
    let mut prepared = reusable.prepare(&mut destination, &value);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}

#[divan::bench(args = WIDTHS)]
fn forced_transform(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure_forced(bencher, len, SquaringAlgorithm::SsaForced);
}

#[divan::bench(args = WIDTHS)]
fn direct_fermat(bencher: divan::Bencher<'_, '_>, len: usize) {
    measure_forced(bencher, len, SquaringAlgorithm::SsaDirectFermat);
}

#[divan::bench(args = WIDTHS)]
fn gmp_reference(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_gmp_limb_width() }
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let output_len = len.checked_mul(2).expect("square width fits usize");
    let mut destination = vec![Limb::MIN; output_len];
    let mut expected = vec![Limb::MIN; output_len];
    let mut oracle = SquaringBenchState::default();
    oracle.prepare(&mut expected, &value).run();
    let count = validated_gmp_count(len);
    validate_and_warm_product(&expected, "GMP square", |probe| {
        // SAFETY: GMP and Limb have matching size/alignment; value has count
        // initialized limbs and the disjoint probe holds exactly 2*count limbs.
        unsafe {
            gmp::mpn_sqr(
                probe.as_mut_ptr().cast::<limb_t>(),
                value.as_ptr().cast::<limb_t>(),
                count,
            );
        }
    });
    bencher.bench_local(|| {
        // SAFETY: two independently allocated, disjoint vectors; the destination
        // holds exactly the 2n limbs mpn_sqr writes for an n-limb operand.
        unsafe {
            gmp::mpn_sqr(
                black_box(destination.as_mut_ptr().cast::<limb_t>()),
                black_box(value.as_ptr().cast::<limb_t>()),
                black_box(count),
            );
        }
        let _output = black_box(&destination);
    });
}

fn measure_forced(bencher: divan::Bencher<'_, '_>, len: usize, algorithm: SquaringAlgorithm) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let expected = gmp_equal_reference(&value, &value);
    let mut destination = vec![Limb::MIN; expected.len()];
    let mut runner = SquaringRunner::new(algorithm, len);
    validate_and_warm_product(&expected, "forced square", |probe| {
        runner.prepare(probe, &value).run();
    });
    let mut prepared = runner.prepare(&mut destination, &value);
    bencher.bench_local(|| black_box(&mut prepared).run());
}
