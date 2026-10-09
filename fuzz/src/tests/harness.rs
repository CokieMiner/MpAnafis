//! Complete selector matrices over values, signs, policies, and representation widths.

use crate::{signed, unsigned};

#[test]
#[cfg_attr(
    miri,
    ignore = "Rug/GMP comparisons use native FFI unavailable to Miri"
)]
fn every_dispatch_branch_matches_gmp_on_edge_values() {
    let cases = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [3, 5, 7],
        [128, 127, 6],
        [255, 2, 17],
        [255, 255, 255],
    ];
    assert_eq!(signed::OPERATIONS, unsigned::OPERATIONS);
    for (category, &count) in signed::OPERATIONS.iter().enumerate() {
        for operation in 0..count {
            let policies: &[u8] = match category {
                6 => &[0, 1, 2, 3],
                7 => &[0, 1, 2, 3, 4, 5, 6, 7],
                1 | 5 | 8 | 10 => &[0, 1],
                _ => &[0],
            };
            let modifiers: &[u8] = if [5, 7, 8].contains(&category) {
                &[0, 0x20]
            } else {
                &[0]
            };
            let parameters: &[u16] = if category == 0 {
                &[0]
            } else {
                &[0, 1, 7, 31, 63, 127, 255, 256, 257, 511, 0x4000, 0x8001]
            };
            for sign in [0, 0x40, 0x80, 0xc0] {
                for &policy in policies {
                    for &modifier in modifiers {
                        for &parameter in parameters {
                            for values in cases {
                                let [lo, hi] = parameter.to_le_bytes();
                                let data = [
                                    u8::try_from(category).unwrap(),
                                    operation,
                                    sign | policy | modifier,
                                    lo,
                                    hi,
                                    85,
                                    128,
                                    values[0],
                                    values[1],
                                    values[2],
                                ];
                                let context = format!(
                                    "category={category} operation={operation} flags={} parameter={parameter} values={values:?}",
                                    data[2]
                                );
                                // Native unwinding attaches controls to a failed assertion; campaigns abort immediately.
                                let result = std::panic::catch_unwind(|| {
                                    unsigned::run(&data);
                                    signed::run(&data);
                                });
                                assert!(result.is_ok(), "{context}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Rug/GMP comparisons use native FFI unavailable to Miri"
)]
fn general_matrix_crosses_sign_byte_limb_and_inline_boundaries() {
    for bytes in [1, 2, 8, 16, 32, 33, 65] {
        for pattern in 0..8 {
            let mut operand = vec![[0, 1, 0x7f, 0x80, 0xff, 0, 0xff, 0][pattern]; bytes];
            if pattern >= 5 {
                operand[0] = if pattern == 6 { 0x7f } else { 0x80 };
            }
            if pattern == 7 {
                operand[bytes - 1] |= 1;
            }
            for flags in [0, 1, 0x80, 0x81] {
                for (category, &count) in signed::OPERATIONS.iter().enumerate() {
                    for operation in 0..count {
                        let mut data = vec![
                            u8::try_from(category).unwrap(),
                            operation,
                            flags,
                            127,
                            0x40,
                            85,
                            128,
                        ];
                        data.extend_from_slice(&operand);
                        data.extend(operand.iter().rev());
                        data.extend_from_slice(&operand);
                        let result = std::panic::catch_unwind(|| {
                            unsigned::run(&data);
                            signed::run(&data);
                        });
                        assert!(
                            result.is_ok(),
                            "category={category} operation={operation} flags={flags} bytes={bytes} pattern={pattern}"
                        );
                    }
                }
            }
        }
    }
}
