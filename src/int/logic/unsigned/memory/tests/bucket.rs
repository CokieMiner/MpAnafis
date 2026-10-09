//! Owned bucket transitions compared with an independent allocation model.

use alloc::vec::Vec;

use proptest::{collection, prelude::ProptestConfig, test_runner::TestRunner};

use super::BucketSlot;

#[test]
fn bucket_transitions_preserve_ownership_best_fit_and_occupied_index_ties() {
    let check = |actions: &[(u8, usize, usize)]| {
        let mut slot = BucketSlot::new();
        let mut retained = Vec::new();
        for &(operation, request, limit) in actions {
            if operation == 0 {
                let mut buffer = Vec::with_capacity(request);
                buffer.resize(request.min(3), request);
                let record = (buffer.capacity(), buffer.as_ptr(), buffer.clone());
                slot.push(buffer, limit);
                if retained.len() < limit {
                    retained.push(record);
                }
            } else {
                let (actual, expected) = if operation == 1 {
                    (slot.pop(), retained.pop())
                } else {
                    let best = retained
                        .iter()
                        .filter(|record| record.0 >= request)
                        .map(|record| record.0)
                        .min();
                    let chosen = best.and_then(|capacity| {
                        retained.iter().rposition(|record| record.0 == capacity)
                    });
                    (
                        slot.take_best_fit(request),
                        chosen.map(|index| retained.swap_remove(index)),
                    )
                };
                assert_eq!(actual.is_some(), expected.is_some());
                if let (Some(buffer), Some((capacity, pointer, contents))) = (actual, expected) {
                    assert_eq!(buffer.capacity(), capacity);
                    assert_eq!(buffer.as_ptr(), pointer);
                    assert_eq!(buffer.as_slice(), contents);
                }
            }
        }
        while let Some((capacity, pointer, contents)) = retained.pop() {
            let buffer = slot.pop().expect("retained entry is occupied");
            assert_eq!(buffer.capacity(), capacity);
            assert_eq!(buffer.as_ptr(), pointer);
            assert_eq!(buffer.as_slice(), contents);
        }
        assert!(slot.pop().is_none(), "the bucket is drained");
    };
    check(&[
        (0, 80, 16),
        (0, 64, 16),
        (0, 80, 16),
        (0, 80, 16),
        (2, 64, 16),
        (2, 75, 16),
        (2, 75, 16),
        (1, 0, 16),
    ]);
    check(&[(0, 64, 1), (0, 64, 1), (2, 65, 1), (2, 64, 1), (1, 0, 1)]);
    check(&alloc::vec![(0, 64, 16); 17]);
    let strategy = collection::vec(
        (0_u8..3, 0_usize..=256, 1_usize..=16),
        0..=if cfg!(miri) { 32 } else { 160 },
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |actions| {
            check(&actions);
            Ok(())
        })
        .expect("bucket state property");
}
