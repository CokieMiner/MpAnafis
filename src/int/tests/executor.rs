//! Public arithmetic tests for execution and assignment policies.

use super::*;

#[test]
fn unsigned_assign_methods_match_default_arithmetic() {
    let left = MpUint::from(123_456_u64);
    let right = MpUint::from(7_890_u64);

    let mut product = MpUint::zero();
    product.assign_mul(&left, &right);
    assert_eq!(
        product,
        &left * &right,
        "destination-reusing product must match"
    );

    product.assign_square(&left);
    assert_eq!(
        product,
        &left * &left,
        "destination-reusing square must match"
    );
}

#[test]
fn signed_assign_methods_preserve_sign_and_value() {
    let left = MpInt::from(-123_456_i64);
    let right = MpInt::from(7_890_i64);

    let mut product = MpInt::zero();
    product.assign_mul(&left, &right);
    assert_eq!(
        product,
        &left * &right,
        "signed destination product must match"
    );

    product.assign_square(&left);
    assert_eq!(
        product,
        &left * &left,
        "signed destination square must match"
    );
}

#[cfg(feature = "rayon")]
#[test]
fn scoped_thread_pool_execution_matches_result() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("build thread pool");

    let left = MpUint::from(987_654_321_u64);
    let right = MpUint::from(123_456_789_u64);

    let pooled_res = pool.install(|| &left * &right);
    assert_eq!(pooled_res, &left * &right);
}
