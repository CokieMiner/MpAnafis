//! Sparse evaluated magnitudes and unequal cancellation at child crossovers.

use alloc::vec;

use crate::int::logic::unsigned::math::mul::Schoolbook;

use super::super::{ArchKernels, ChildDemand, Limb, Multiplication, Toom8};

#[cfg_attr(
    miri,
    ignore = "512-limb sparse child products are covered by native execution"
)]
#[test]
fn sparse_evaluations_fit_the_parent_workspace() {
    let split = 512_usize;
    let evaluation_len = Toom8::evaluation_len(split);
    let output_len = Toom8::interpolation_value_len(split);
    let mut scratch = vec![Limb::MAX; ChildDemand::recursive_mul_scratch(split, evaluation_len)];
    let kernel = ArchKernels::selected_add_mul_limbs_unchecked();
    for (left_len, right_len) in [(512, 128), (128, 512), (511, 127), (32, 257), (0, 512)] {
        let mut left = vec![0; evaluation_len];
        let mut right = vec![0; evaluation_len];
        left.get_mut(..left_len)
            .expect("left evaluation fits")
            .fill(Limb::MAX);
        right
            .get_mut(..right_len)
            .expect("right evaluation fits")
            .fill(Limb::MAX);
        let mut expected = vec![0; output_len];
        Schoolbook::mul(&mut expected, &left, &right);
        let mut actual = vec![Limb::MAX; output_len];
        scratch.fill(Limb::MAX);
        Toom8::mul_evaluation(&mut actual, &left, &right, &mut scratch, split, kernel);
        assert_eq!(actual, expected, "sparse evaluation {left_len}x{right_len}");
    }
}

#[cfg_attr(
    miri,
    ignore = "wide cancellation products are covered by native execution"
)]
#[test]
fn unequal_negative_evaluations_preserve_the_complete_product() {
    for split in [511_usize, 512, 513] {
        let len = split.checked_mul(8).expect("test operand fits");
        // Equal adjacent parts cancel at -1; changing the constant part leaves
        // one full-width and one quarter-width negative evaluation.
        let mut left = vec![1; len];
        let mut right = vec![1; len];
        left.get_mut(..split).expect("full low part").fill(2);
        right
            .get_mut(..split.div_euclid(4))
            .expect("partial low part")
            .fill(2);
        let output_len = len.checked_mul(2).expect("test product fits");
        let mut expected = vec![0; output_len];
        Schoolbook::mul(&mut expected, &left, &right);
        let mut actual = vec![Limb::MAX; output_len];
        let mut scratch = vec![Limb::MAX; Multiplication::toom8_mul_scratch_len(len, len)];
        Toom8::mul(&mut actual, &left, &right, &mut scratch);
        assert_eq!(actual, expected, "cancelling parts of {split} limbs");
    }
}
