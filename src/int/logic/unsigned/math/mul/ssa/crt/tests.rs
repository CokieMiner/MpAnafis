//! Retained CRT plan ownership, cache eviction, and shape validation.

use super::{CrtMulPlan, CrtSquarePlan, SSA_BNM1_BASECASE_LIMBS};

#[test]
fn crt_plans_reject_invalid_ring_widths() {
    assert!(CrtMulPlan::new(0).is_none());
    assert!(CrtSquarePlan::new(0).is_none());
    let width = SSA_BNM1_BASECASE_LIMBS
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_add(1))
        .expect("the configured CRT basecase leaves room for a recursive test");
    assert!(CrtMulPlan::new(width).is_none());
    assert!(CrtSquarePlan::new(width).is_none());
}

#[cfg(feature = "std")]
#[test]
fn crt_plan_storage_survives_cache_eviction() {
    let width = SSA_BNM1_BASECASE_LIMBS
        .checked_mul(2)
        .expect("a recursive CRT test width fits usize");
    let product = CrtMulPlan::new(width).expect("a doubled basecase admits a CRT split");
    let square = CrtSquarePlan::new(width).expect("a doubled basecase admits a CRT split");
    let product_shared = CrtMulPlan::new(width).expect("cached product geometry remains valid");
    let square_shared = CrtSquarePlan::new(width).expect("cached square geometry remains valid");
    assert_eq!(product.as_ptr(), product_shared.as_ptr());
    assert_eq!(square.as_ptr(), square_shared.as_ptr());

    // More distinct widths than retained slots evict both initial cache entries.
    // Live plans still own their levels while subsequent calls build replacements.
    for shift in 1..=5 {
        let next_width = width.checked_shl(shift).expect("test CRT widths fit usize");
        let _product = CrtMulPlan::new(next_width).expect("a power-of-two scaling admits CRT");
        let _square = CrtSquarePlan::new(next_width).expect("a power-of-two scaling admits CRT");
    }
    let product_rebuilt = CrtMulPlan::new(width).expect("evicted geometry remains valid");
    let square_rebuilt = CrtSquarePlan::new(width).expect("evicted geometry remains valid");
    assert_ne!(product.as_ptr(), product_rebuilt.as_ptr());
    assert_ne!(square.as_ptr(), square_rebuilt.as_ptr());
    assert_eq!(product.len(), product_rebuilt.len());
    assert_eq!(square.len(), square_rebuilt.len());
    assert_eq!(product.len(), product_shared.len());
    assert_eq!(square.len(), square_shared.len());
}
