//! Byte-decoder controls and complete payload partitioning.

use crate::{Input, signed, unsigned};

#[test]
fn header_controls_are_independent_and_payload_is_partitioned() {
    for length in 0..7 {
        assert!(Input::parse(&vec![0; length]).is_none());
    }
    assert_eq!(signed::OPERATIONS.len(), unsigned::OPERATIONS.len());
    for category in 0..u8::try_from(signed::OPERATIONS.len()).unwrap() {
        for operation in 0..=255 {
            let data = [category, operation, 0xc0, 0x34, 0x12, 85, 128, 3, 5, 7];
            let input = Input::parse(&data).unwrap();
            assert_eq!(
                usize::from(input.category),
                usize::from(category) % signed::OPERATIONS.len()
            );
            assert_eq!(input.operation, operation);
            assert_eq!(input.flags, 0xc0);
            assert_eq!(input.parameter, 0x1234);
            assert_eq!(input.left, [3]);
            assert_eq!(input.right, [5]);
            assert_eq!(input.modulus, [7]);
        }
    }
    for category in 0..=255 {
        let input_data = [category, 173, 0, 0, 0, 0, 0];
        let input = Input::parse(&input_data).unwrap();
        assert_eq!(
            usize::from(input.category),
            usize::from(category) % signed::OPERATIONS.len()
        );
        assert_eq!(input.operation, 173);
    }
    for length in [0, 1, 2, 3, 7, 255, 1024] {
        for first in [0, 1, 85, 128, 254, 255] {
            for second in [0, 1, 85, 128, 254, 255] {
                let mut data = vec![0, 0, 0, 0, 0, first, second];
                data.extend(vec![17; length]);
                let input = Input::parse(&data).unwrap();
                let left = length * usize::from(first) / 255;
                let right = (length - left) * usize::from(second) / 255;
                assert_eq!(input.left.len(), left);
                assert_eq!(input.right.len(), right);
                assert_eq!(input.modulus.len(), length - left - right);
                assert_eq!([input.left, input.right, input.modulus].concat(), data[7..]);
            }
        }
    }
}
