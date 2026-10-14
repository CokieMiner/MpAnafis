use super::{Limb, Ntt, NttPlan, NttWorkspace, SequentialExecutor, schoolbook::product};

#[test]
fn native_products_initialize_the_output_and_preserve_guards_and_failed_destinations() {
    let executor = SequentialExecutor::default();
    let mut workspace = NttWorkspace::new();
    for (left, right) in [
        (vec![], vec![1]),
        (vec![1], vec![]),
        (vec![0], vec![0]),
        (vec![Limb::MAX; 4], vec![Limb::MAX; 7]),
        (vec![3; 33], vec![7; 65]),
    ] {
        let plan = NttPlan::for_multiplication(left.len(), right.len());
        let expected = product(&left, &right);
        let mut dst = vec![Limb::MAX; expected.len().checked_add(2).expect("guard span")];
        let scratch_len = plan.as_ref().map_or(0, NttPlan::required_mul_scratch_limbs);
        let mut scratch = vec![Limb::MAX; scratch_len.checked_add(2).expect("scratch guard span")];
        assert!(Ntt::try_mul_with_scratch_and_executor(
            &mut dst,
            &left,
            &right,
            &mut scratch,
            &executor
        ));
        assert_eq!(dst.get(..expected.len()).expect("output prefix"), expected);
        assert_eq!(
            dst.get(expected.len()..).expect("output suffix"),
            [Limb::MAX; 2]
        );
        assert_eq!(
            scratch.get(scratch_len..).expect("scratch suffix"),
            [Limb::MAX; 2]
        );
        assert!(workspace.mul_with_executor(&mut dst, &left, &right, &executor));
        assert_eq!(
            dst.get(..expected.len()).expect("workspace product"),
            expected
        );
        if plan.is_some() {
            let original = dst.clone();
            assert!(!Ntt::try_mul_with_scratch_and_executor(
                &mut dst,
                &left,
                &right,
                &mut [],
                &executor
            ));
            assert_eq!(dst, original);
            let short = dst
                .get_mut(..expected.len().checked_sub(1).expect("positive output"))
                .expect("short prefix");
            let before = short.to_vec();
            assert!(!workspace.mul_with_executor(short, &left, &right, &executor));
            assert!(!Ntt::try_mul_with_scratch_and_executor(
                short,
                &left,
                &right,
                &mut scratch,
                &executor
            ));
            assert_eq!(short, before);
        }
    }
}
