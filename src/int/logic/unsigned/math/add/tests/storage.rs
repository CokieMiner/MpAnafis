//! Exact-capacity addition and the high-column carry bound.

use super::{
    super::{InternalMpUint, Limb},
    values::check_arithmetic,
};

#[test]
fn high_column_bounds_reuse_exact_capacity_and_reserve_possible_carries() {
    for &width in if cfg!(miri) {
        &[5][..]
    } else {
        &[5, 8, 16, 64][..]
    } {
        let left = InternalMpUint::from_limbs(alloc::vec![1; width]);
        let right = InternalMpUint::from_limbs(alloc::vec![2; width]);
        let mut output = left.clone();
        let allocation = output.limbs().as_ptr();
        let capacity = output.capacity();
        output.add_assign(&right);
        assert_eq!(output.limbs(), alloc::vec![3; width]);
        assert_eq!(output.limbs().as_ptr(), allocation);
        assert_eq!(output.capacity(), capacity);
        let mut fused = InternalMpUint::with_capacity(width);
        let fused_allocation = fused.limbs().as_ptr();
        fused.assign_sum(&left, &right);
        assert_eq!(fused, output);
        assert_eq!(fused.capacity(), width);
        assert_eq!(fused.limbs().as_ptr(), fused_allocation);
        for right_top in [Limb::MAX - 2, Limb::MAX - 1, Limb::MAX] {
            for lower in [0, Limb::MAX] {
                let mut a = alloc::vec![lower; width];
                let mut b = alloc::vec![1; width];
                *a.last_mut().expect("nonempty operand") = 1;
                *b.last_mut().expect("nonempty operand") = right_top;
                check_arithmetic(
                    &InternalMpUint::from_limbs(a),
                    &InternalMpUint::from_limbs(b),
                );
            }
        }
    }
    for &width in if cfg!(miri) {
        &[5][..]
    } else {
        &[5, 8, 33, 257][..]
    } {
        let right = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        for inline_width in [1, 4] {
            let mut left = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; inline_width]);
            let expected = left.add(&right);
            left.add_assign(&right);
            assert_eq!(left, expected);
            assert_eq!(left.capacity(), width + 1);
        }
    }
}
