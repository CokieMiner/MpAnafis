//! Property tests for addition ownership and assignment combinations.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "Addition operators are the API under test"
)]

use core::ops::{Add, AddAssign};

use proptest::prelude::*;

use super::strategies;

proptest! {
    #[test]
    fn prop_uint_ops_add(
        a in strategies::uint(32),
        b in strategies::uint(32),
        additional in 1_usize..=8,
    ) {
        let expected = Add::add(a.clone(), &b);
        assert_eq!(Add::add(&a, &b), expected);
        assert_eq!(Add::add(a.clone(), b.clone()), expected);
        assert_eq!(Add::add(&a, b.clone()), expected);

        let mut borrowed_assign = a.clone();
        AddAssign::add_assign(&mut borrowed_assign, &b);
        assert_eq!(borrowed_assign, expected);

        let mut owned_assign = a.clone();
        AddAssign::add_assign(&mut owned_assign, b.clone());
        assert_eq!(owned_assign, expected);

        let mut spare_rhs_for_add = b.clone();
        spare_rhs_for_add.reserve(additional);
        assert_eq!(Add::add(a.clone(), spare_rhs_for_add), expected);

        let mut spare_assign = a;
        let mut spare_rhs_for_assign = b;
        spare_rhs_for_assign.reserve(additional);
        AddAssign::add_assign(&mut spare_assign, spare_rhs_for_assign);
        assert_eq!(spare_assign, expected);
    }

    #[test]
    fn prop_int_ops_add(
        a in strategies::int(32),
        b in strategies::int(32),
        additional in 1_usize..=8,
    ) {
        let expected = Add::add(a.clone(), &b);
        assert_eq!(Add::add(&a, &b), expected);
        assert_eq!(Add::add(a.clone(), b.clone()), expected);
        assert_eq!(Add::add(&a, b.clone()), expected);

        let mut borrowed_assign = a.clone();
        AddAssign::add_assign(&mut borrowed_assign, &b);
        assert_eq!(borrowed_assign, expected);

        let mut owned_assign = a.clone();
        AddAssign::add_assign(&mut owned_assign, b.clone());
        assert_eq!(owned_assign, expected);

        let mut spare_rhs_for_add = b.clone();
        spare_rhs_for_add.reserve(additional);
        assert_eq!(Add::add(a.clone(), spare_rhs_for_add), expected);

        let mut spare_assign = a;
        let mut spare_rhs_for_assign = b;
        spare_rhs_for_assign.reserve(additional);
        AddAssign::add_assign(&mut spare_assign, spare_rhs_for_assign);
        assert_eq!(spare_assign, expected);
    }
}
