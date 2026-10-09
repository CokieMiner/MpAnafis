//! Isolated thread-local selection, retention, allocation growth, and teardown.

use core::cell::RefCell;
use std::{thread, thread_local};

use alloc::vec::Vec;

use super::ScratchBuffer;

#[test]
fn pooling_selects_smallest_adequate_allocations_and_index_ties() {
    thread::spawn(|| {
        let small = ScratchBuffer::acquire(64);
        let middle = ScratchBuffer::acquire(80);
        let large = ScratchBuffer::acquire(110);
        let records = [
            (75, middle.as_ptr(), middle.capacity()),
            (100, large.as_ptr(), large.capacity()),
            (50, small.as_ptr(), small.capacity()),
        ];
        assert!(
            small.capacity() < middle.capacity() && middle.capacity() < large.capacity(),
            "ordered capacities share a bucket"
        );
        drop(small);
        drop(middle);
        drop(large);
        let held: Vec<_> = records
            .iter()
            .map(|(request, _, _)| ScratchBuffer::acquire(*request))
            .collect();
        for (actual, (_, pointer, capacity)) in held.iter().zip(records) {
            assert_eq!(actual.as_ptr(), pointer);
            assert_eq!(actual.capacity(), capacity);
            assert_eq!(actual.len(), 0);
        }
    })
    .join()
    .expect("isolated best-fit thread");
    thread::spawn(|| {
        let oldest = ScratchBuffer::acquire(80);
        let interior = ScratchBuffer::acquire(64);
        let middle = ScratchBuffer::acquire(80);
        let newest = ScratchBuffer::acquire(80);
        let pointers = [
            interior.as_ptr(),
            middle.as_ptr(),
            newest.as_ptr(),
            oldest.as_ptr(),
        ];
        let capacity = oldest.capacity();
        assert_eq!(middle.capacity(), capacity);
        assert_eq!(newest.capacity(), capacity);
        drop(oldest);
        drop(interior);
        drop(middle);
        drop(newest);
        // Removing the interior entry compacts [oldest, interior, middle, newest]
        // to [oldest, newest, middle]; ties select the highest occupied index.
        let selected = ScratchBuffer::acquire(64);
        let first = ScratchBuffer::acquire(capacity);
        let second = ScratchBuffer::acquire(capacity);
        let last = ScratchBuffer::acquire(capacity);
        for (actual, expected) in [selected, first, second, last].iter().zip(pointers) {
            assert_eq!(actual.as_ptr(), expected);
        }
    })
    .join()
    .expect("isolated occupied-index thread");
}

#[test]
fn recycling_and_clone_growth_reuse_available_capacity() {
    thread::spawn(|| {
        let tiny = ScratchBuffer::acquire(15);
        assert_eq!(tiny.capacity(), 15);
        for width in [64, 256] {
            let buffer = ScratchBuffer::acquire(width);
            let (pointer, capacity) = (buffer.as_ptr(), buffer.capacity());
            drop(buffer);
            let reused = ScratchBuffer::acquire(capacity);
            assert_eq!(reused.as_ptr(), pointer);
            assert_eq!(reused.capacity(), capacity);
        }
        let mut source = ScratchBuffer::acquire(256);
        source.resize(256, usize::MAX);
        let cached = ScratchBuffer::acquire(256);
        let (pointer, capacity) = (cached.as_ptr(), cached.capacity());
        drop(cached);
        let mut destination = ScratchBuffer::acquire(0);
        destination.clone_from(&source);
        assert_eq!(destination.as_slice(), source.as_slice());
        assert_eq!(destination.as_ptr(), pointer);
        assert_eq!(destination.capacity(), capacity);
    })
    .join()
    .expect("isolated recycled growth thread");
}

#[test]
fn pool_retention_keeps_only_the_bounded_occupied_prefix() {
    thread::spawn(|| {
        let buffers: Vec<_> = (0..17).map(|_| ScratchBuffer::acquire(64)).collect();
        let retained: Vec<_> = buffers
            .iter()
            .take(16)
            .map(|buffer| buffer.as_ptr())
            .collect();
        for buffer in buffers {
            drop(buffer);
        }
        let reused: Vec<_> = (0..16).map(|_| ScratchBuffer::acquire(64)).collect();
        for buffer in &reused {
            assert!(
                retained.contains(&buffer.as_ptr()),
                "the retained allocations are reused"
            );
            assert_eq!(buffer.len(), 0);
        }
        let fresh = ScratchBuffer::acquire(64);
        assert!(
            !retained.contains(&fresh.as_ptr()),
            "the seventeenth request needs a distinct allocation"
        );
    })
    .join()
    .expect("isolated retention thread");
}

#[test]
#[cfg(not(target_pointer_width = "16"))]
#[cfg_attr(
    miri,
    ignore = "Allocates millions of limbs to check the largest retention bucket; smaller allocation ownership tests run under Miri"
)]
fn upper_capacity_bound_bypasses_pooling_and_growth_stays_retainable() {
    thread::spawn(|| {
        let pooled = ScratchBuffer::acquire(1 << 20);
        let (pointer, capacity) = (pooled.as_ptr(), pooled.capacity());
        drop(pooled);
        let oversized = ScratchBuffer::acquire(1 << 21);
        assert!(
            oversized.capacity() >= 1 << 21,
            "oversized requests are fully reserved"
        );
        drop(oversized);
        let reacquired = ScratchBuffer::acquire(capacity);
        assert_eq!(reacquired.as_ptr(), pointer);
        assert_eq!(reacquired.capacity(), capacity);
    })
    .join()
    .expect("isolated upper-bound thread");
    thread::spawn(|| {
        let initial =
            ScratchBuffer::acquire((1_usize << 20).checked_add(100).expect("bounded capacity"));
        assert!(
            initial.capacity() < 1 << 21,
            "initial capacity is retainable"
        );
        drop(initial);
        let request = (1_usize << 20).checked_add(1000).expect("bounded capacity");
        let expanded = ScratchBuffer::acquire(request);
        assert!(
            expanded.capacity() >= request && expanded.capacity() < 1 << 21,
            "exact growth preserves the pool limit"
        );
        let (pointer, capacity) = (expanded.as_ptr(), expanded.capacity());
        drop(expanded);
        let reused = ScratchBuffer::acquire(capacity);
        assert_eq!(reused.as_ptr(), pointer);
        assert_eq!(reused.capacity(), capacity);
    })
    .join()
    .expect("isolated upper-bucket growth thread");
}

#[test]
fn scratch_acquisition_falls_back_during_thread_local_teardown() {
    struct TeardownClient;
    impl Drop for TeardownClient {
        fn drop(&mut self) {
            let buffer = ScratchBuffer::acquire(128);
            assert!(
                buffer.capacity() >= 128,
                "destroyed arena state permits independent acquisition"
            );
        }
    }
    thread_local! { static CLIENT: RefCell<Option<TeardownClient>> = const { RefCell::new(None) }; }
    thread::spawn(|| {
        CLIENT.with(|client| {
            *client.borrow_mut() = Some(TeardownClient);
        });
        let buffer = ScratchBuffer::acquire(128);
        assert!(
            buffer.capacity() >= 128,
            "register the arena after its teardown client"
        );
    })
    .join()
    .expect("thread teardown acquisition");
}
