//! Virtual-width overflow and packed destination interval admission.

use super::super::Toom8;

#[test]
fn destination_placement_rejects_unrepresentable_intervals() {
    let split = usize::MAX.div_euclid(16);
    let packed = split.checked_mul(3).expect("virtual packed width fits");
    assert!(Toom8::destination_points_fit(
        usize::MAX,
        split,
        packed,
        split
    ));
    assert!(!Toom8::destination_points_fit(
        usize::MAX,
        split,
        packed,
        usize::MAX
    ));
    assert!(!Toom8::destination_points_fit(
        usize::MAX,
        usize::MAX,
        packed,
        0
    ));
    assert!(!Toom8::destination_points_fit(31, 2, 9, 0));
}
