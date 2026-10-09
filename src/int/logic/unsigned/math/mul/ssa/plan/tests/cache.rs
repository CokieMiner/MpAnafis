//! Plan retention, complete cache keys, eviction, and thread-local isolation.

use core::ptr::eq;
use std::{sync::Barrier, thread};

use super::{
    super::{
        super::product::{PointwiseMulStrategy, PointwiseSquareStrategy},
        PointwiseMulPlan, PointwiseSquarePlan, SSA_BASE_MODULUS_BITS,
    },
    FftPlan, Geometry, LIMB_BITS, SsaOperation, SsaPlan,
};

#[test]
fn geometry_cache_retains_power_of_two_and_colliding_irregular_widths() {
    let widths = [
        1 << 18,
        1 << 19,
        1 << 20,
        768 * LIMB_BITS,
        196_608 * LIMB_BITS,
    ];
    let plans = widths.map(FftPlan::new);
    // Irregular limb widths 0x300 and 0x30000 select the same folded-byte set.
    for (bits, plan) in widths.into_iter().zip(plans) {
        let cached = Geometry::cached(bits, SsaOperation::Multiply).expect("retained geometry");
        assert_eq!(
            (cached.transform_len, cached.chunk_bits, cached.inner_bits),
            (plan.transform_len, plan.chunk_bits, plan.inner_bits)
        );
    }
}

#[test]
fn crt_cache_keeps_complete_keys_and_replaces_evicted_entries() {
    for operation in [
        SsaOperation::Multiply,
        SsaOperation::Square,
        SsaOperation::Pair,
    ] {
        for required in [1024 * LIMB_BITS, 4096 * LIMB_BITS + 1] {
            if let Some(first) = SsaPlan::best_crt_half_width(required, operation) {
                assert_eq!(
                    SsaPlan::best_crt_half_width(required, operation),
                    Some(first)
                );
            }
        }
    }
    let low = 17 * LIMB_BITS;
    let high = low | 1_usize.rotate_right(2);
    let entries = [
        (low, SsaOperation::Multiply, 128),
        (high, SsaOperation::Multiply, 256),
        (low, SsaOperation::Square, 384),
        (high, SsaOperation::Square, 512),
        (low, SsaOperation::Pair, 640),
        (high, SsaOperation::Pair, 768),
    ];
    // Synthetic values exercise key retention without entering an arithmetic kernel.
    for (bits, operation, width) in entries {
        SsaPlan::cache_crt_half_width(bits, operation, width);
    }
    for (bits, operation, width) in entries {
        assert_eq!(SsaPlan::cached_crt_half_width(bits, operation), Some(width));
    }
    for bits in 100..180 {
        SsaPlan::cache_crt_half_width(bits, SsaOperation::Multiply, bits);
    }
    for (bits, operation, _) in entries {
        assert_eq!(SsaPlan::cached_crt_half_width(bits, operation), None);
    }
    assert_eq!(
        SsaPlan::cached_crt_half_width(179, SsaOperation::Multiply),
        Some(179)
    );
}

#[test]
fn crt_cache_replacements_remain_thread_local() {
    let barrier = Barrier::new(2);
    thread::scope(|scope| {
        for payload in [128, 256] {
            let gate = &barrier;
            let _worker = scope.spawn(move || {
                SsaPlan::cache_crt_half_width(usize::MAX, SsaOperation::Pair, payload);
                let _arrival = gate.wait();
                assert_eq!(
                    SsaPlan::cached_crt_half_width(usize::MAX, SsaOperation::Pair),
                    Some(payload)
                );
            });
        }
    });
}

#[expect(
    clippy::panic,
    reason = "The test requires transform strategies above the basecase boundary"
)]
#[test]
fn pointwise_plans_retain_shared_children_after_cache_replacements() {
    let bits = SSA_BASE_MODULUS_BITS
        .checked_mul(2)
        .expect("test ring fits");
    let (PointwiseMulStrategy::Transform(product), PointwiseMulStrategy::Transform(shared_product)) = (
        PointwiseMulPlan::from(bits).strategy,
        PointwiseMulPlan::from(bits).strategy,
    ) else {
        panic!("nested product required");
    };
    let (
        PointwiseSquareStrategy::Transform(square),
        PointwiseSquareStrategy::Transform(shared_square),
    ) = (
        PointwiseSquarePlan::from(bits).strategy,
        PointwiseSquarePlan::from(bits).strategy,
    )
    else {
        panic!("nested square required");
    };
    assert!(eq(product.as_ref(), shared_product.as_ref()));
    assert!(eq(square.as_ref(), shared_square.as_ref()));
    for shift in 1..=5 {
        let next = bits.checked_shl(shift).expect("test ring fits");
        let _product = PointwiseMulPlan::from(next);
        let _square = PointwiseSquarePlan::from(next);
    }
    assert_eq!(
        product.transform_mul_scratch(1),
        shared_product.transform_mul_scratch(1)
    );
    assert_eq!(
        square.transform_sqr_scratch(1),
        shared_square.transform_sqr_scratch(1)
    );
}
