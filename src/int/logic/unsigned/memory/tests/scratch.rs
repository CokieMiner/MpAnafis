//! Initialized prefixes, capacity reuse, growth, reset, and allocation exchange.

use proptest::{
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::ScratchBuffer;

#[test]
fn scratch_copies_resets_and_swaps_preserve_initialized_values_and_capacity() {
    let check =
        |source_width: usize, requested_length: usize, destination_width: usize, fill: usize| {
            let length = requested_length.min(source_width);
            let mut source = ScratchBuffer::acquire(source_width);
            assert_eq!(source.len(), 0);
            assert!(
                source.capacity() >= source_width,
                "the request is fully reserved"
            );
            source.resize(length, fill);
            let cloned = source.clone();
            assert_eq!(cloned.as_slice(), source.as_slice());
            assert!(
                cloned.capacity() >= source.capacity(),
                "clone preserves reserved capacity"
            );
            if source.capacity() != 0 {
                assert_ne!(cloned.as_ptr(), source.as_ptr());
            }
            let mut destination = ScratchBuffer::acquire(destination_width);
            destination.resize(destination_width, usize::MAX);
            let pointer = destination.as_ptr();
            let capacity = destination.capacity();
            destination.clone_from(&source);
            assert_eq!(destination.as_slice(), source.as_slice());
            if capacity >= length {
                assert_eq!(destination.as_ptr(), pointer);
                assert_eq!(destination.capacity(), capacity);
            }
            let source_pointer = source.as_ptr();
            let destination_pointer = destination.as_ptr();
            ScratchBuffer::clone_into(&mut source, &mut destination);
            assert_eq!(source.as_slice(), destination.as_slice());
            assert_eq!(source.as_ptr(), destination_pointer);
            assert_eq!(destination.as_ptr(), source_pointer);
            let reusable = source.as_ptr();
            let reusable_capacity = source.capacity();
            source.reset_with_capacity(reusable_capacity);
            assert_eq!(source.len(), 0);
            assert_eq!(source.as_ptr(), reusable);
            let larger = reusable_capacity.checked_add(17).expect("bounded request");
            source.reset_with_capacity(larger);
            assert_eq!(source.len(), 0);
            assert!(
                source.capacity() >= larger,
                "reset reserves the larger request"
            );
            source.resize(larger, fill);
            assert!(
                source.iter().all(|limb| *limb == fill),
                "all active limbs are initialized"
            );
            let pointer_after_growth = source.as_ptr();
            let capacity_after_growth = source.capacity();
            source.clone_from(&ScratchBuffer::acquire(0));
            assert_eq!(source.len(), 0);
            assert_eq!(source.as_ptr(), pointer_after_growth);
            assert_eq!(source.capacity(), capacity_after_growth);
        };
    for width in [0_usize, 1, 8, 15, 16, 17, 63, 64, 65, 256] {
        for length in [0, width.div_euclid(2), width] {
            check(width, length, 256, usize::MAX);
        }
    }
    check(4096, 128, 256, usize::MAX);
    check(256, 256, 0, usize::MAX);
    let strategy = (0_usize..=256, 0_usize..=256, 0_usize..=256, any::<usize>());
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(width, length, destination, fill)| {
            check(width, length, destination, fill);
            Ok(())
        })
        .expect("scratch ownership property");
}
