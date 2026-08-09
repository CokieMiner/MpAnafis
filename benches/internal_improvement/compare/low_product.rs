//! Equal-width truncated low-product comparisons against GMP's `mpn_mullo_n`.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's internal mullo routine on exact disjoint limb spans"
)]

use core::{hint::black_box, mem::size_of};

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb,
    state::{MulBenchScratch, MulloBenchScratch},
};

use crate::shared::operands;

const LOW_PRODUCT_SIZES: [usize; 7] = [32, 64, 192, 384, 1_024, 4_096, 8_192];
const LOW_BASECASE_SIZES: [usize; 3] = [32, 64, 192];
const ROOT_SPLITS_4096: [usize; 9] = [256, 320, 341, 384, 409, 455, 512, 614, 735];

unsafe extern "C" {
    #[link_name = "__gmpn_mullo_n"]
    fn mpn_mullo_n(
        destination: *mut limb_t,
        left: *const limb_t,
        right: *const limb_t,
        len: size_t,
    );
}

#[divan::bench(args = LOW_PRODUCT_SIZES)]
fn mp_mullo(bencher: divan::Bencher, len: usize) {
    let (left, right, mut full_product) = operands(len);
    let mut full_scratch = MulBenchScratch::default();
    full_scratch.run(&mut full_product, &left, &right);

    let mut destination = vec![Limb::MIN; len];
    let mut low_scratch = MulloBenchScratch::default();
    low_scratch.run(&mut destination, &left, &right);
    let expected_low = full_product
        .get(..len)
        .expect("full Mp product contains its low half");
    assert_eq!(
        destination.as_slice(),
        expected_low,
        "Mp truncated product disagrees with its production full product"
    );

    bencher.bench_local(|| {
        low_scratch.run(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = LOW_BASECASE_SIZES)]
fn mp_basecase(bencher: divan::Bencher, len: usize) {
    let (left, right, _) = operands(len);
    let mut destination = vec![Limb::MIN; len];

    bencher.bench_local(|| {
        MulloBenchScratch::run_basecase(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = ROOT_SPLITS_4096)]
fn mp_forced_root_4096(bencher: divan::Bencher, small_len: usize) {
    let len = 4_096;
    let (left, right, mut full_product) = operands(len);
    let mut full_scratch = MulBenchScratch::default();
    full_scratch.run(&mut full_product, &left, &right);
    let expected_low = full_product
        .get(..len)
        .expect("full Mp product contains its low half");

    let mut destination = vec![Limb::MIN; len];
    let mut low_scratch = MulloBenchScratch::default();
    low_scratch.run_forced_root(&mut destination, &left, &right, small_len);
    assert_eq!(
        destination.as_slice(),
        expected_low,
        "forced-root low product disagrees with the production full product"
    );

    bencher.bench_local(|| {
        low_scratch.run_forced_root(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(small_len),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = LOW_PRODUCT_SIZES)]
fn mp_full_product(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = MulBenchScratch::default();

    bencher.bench_local(|| {
        scratch.run(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = LOW_PRODUCT_SIZES)]
fn gmp_mullo(bencher: divan::Bencher, len: usize) {
    assert_eq!(
        size_of::<Limb>(),
        size_of::<limb_t>(),
        "raw low-product comparison requires equal Mp and GMP limb widths"
    );
    let (mp_left, mp_right, _) = operands(len);
    let left: Vec<limb_t> = mp_left
        .into_iter()
        .map(|limb| limb_t::try_from(limb).expect("Mp limb fits GMP limb"))
        .collect();
    let right: Vec<limb_t> = mp_right
        .into_iter()
        .map(|limb| limb_t::try_from(limb).expect("Mp limb fits GMP limb"))
        .collect();
    let mut destination = vec![limb_t::MIN; len];
    let mut full_product = vec![limb_t::MIN; len.saturating_mul(2)];
    let gmp_len = size_t::try_from(len).expect("benchmark width fits GMP mp_size_t");

    // SAFETY: all vectors are independently allocated; each input contains
    // exactly gmp_len initialized limbs, destination has gmp_len writable
    // limbs, and full_product has 2*gmp_len writable limbs. The declaration
    // matches GMP 6.3's internal `void mpn_mullo_n(rp, xp, yp, n)` signature.
    unsafe {
        gmp::mpn_mul_n(
            full_product.as_mut_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            gmp_len,
        );
        mpn_mullo_n(
            destination.as_mut_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            gmp_len,
        );
    }
    let expected_low = full_product
        .get(..len)
        .expect("full GMP product contains its low half");
    assert_eq!(
        destination.as_slice(),
        expected_low,
        "GMP mpn_mullo_n disagrees with GMP's production full product"
    );

    bencher.bench_local(|| {
        // SAFETY: the validated disjoint spans above remain allocated and
        // initialized with the same exact widths for every timed iteration.
        unsafe {
            mpn_mullo_n(
                black_box(destination.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(right.as_ptr()),
                black_box(gmp_len),
            );
        }
        let _output = black_box(&destination);
    });
}
