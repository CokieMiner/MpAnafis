//! Primitive checks for the float and modular GMP reference contracts.

use rug::Integer;

use crate::{BitReference, TheoryReference, float32, float64, modular};

#[test]
#[cfg_attr(
    miri,
    ignore = "Reference models use GMP native FFI unavailable to Miri"
)]
fn reference_models_preserve_rounding_domains_and_bit_orientation() {
    for value in [
        0_u64,
        1,
        (1 << 24) + 1,
        (1 << 24) + 3,
        (1 << 53) + 1,
        (1 << 53) + 3,
        u64::MAX,
    ] {
        let integer = Integer::from(value);
        // Primitive integer-to-float casts specify the independent nearest-even reference.
        assert_eq!(float64(&integer), Some(value as f64));
        assert_eq!(float32(&integer), Some(value as f32));
        assert_eq!(float64(&-integer.clone()), Some(-(value as f64)));
        assert_eq!(float32(&-integer), Some(-(value as f32)));
    }
    assert_eq!(float32(&(Integer::from(1) << 128)), None);
    assert_eq!(float64(&(Integer::from(1) << 1024)), None);
    let (a, b, m) = (Integer::from(3), Integer::from(5), Integer::from(7));
    assert_eq!(modular(&a, &b, &m, 1, 0), Some(Integer::from(5)));
    assert_eq!(modular(&a, &b, &m, 2, 0), Some(Integer::from(1)));
    assert_eq!(modular(&a, &b, &Integer::from(6), 4, 0), None);
    assert_eq!(modular(&a, &b, &Integer::new(), 3, 0), None);
    for value in [0_u16, 1, 2, 127, 128, 255, 256, u16::MAX] {
        let integer = Integer::from(value);
        assert_eq!(BitReference::reverse(&integer, 16), value.reverse_bits());
        assert_eq!(
            BitReference::swap_bytes(&integer, Some(16)),
            value.swap_bytes()
        );
        for shift in [0_u32, 1, 15, 16, 17, 31] {
            assert_eq!(
                BitReference::rotate(&integer, 16, usize::try_from(shift).unwrap(), true),
                value.rotate_left(shift)
            );
            assert_eq!(
                BitReference::rotate(&integer, 16, usize::try_from(shift).unwrap(), false),
                value.rotate_right(shift)
            );
        }
    }
    assert_eq!(TheoryReference::phi(0), None);
    assert_eq!(TheoryReference::phi(1), Some(Integer::from(1)));
    assert_eq!(TheoryReference::phi(36), Some(Integer::from(12)));
    let pseudoprime = Integer::from_str_radix("318665857834031151167461", 10).unwrap();
    assert!(TheoryReference::probably_prime(&pseudoprime, 12));
    assert!(!TheoryReference::probably_prime(&pseudoprime, 13));
}
