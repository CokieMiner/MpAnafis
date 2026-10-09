//! Scratch prefix preservation, suffix initialization, and capacity growth.

use crate::int::logic::unsigned::math::mul::{Limb, MulScratch, ScratchBuffer};

#[test]
fn preparation_preserves_prefix_and_initializes_growth() {
    let mut scratch = MulScratch {
        buf: ScratchBuffer::acquire(128),
    };
    scratch.buf.extend_from_slice(&[7; 5]);
    let pointer = scratch.buf.as_ptr();
    for requested in [0_usize, 3, 5, 6, 17, 128, 1] {
        let old_len = scratch.buf.len();
        let prefix = scratch.buf.to_vec();
        scratch.prepare(requested);
        assert_eq!(scratch.buf.len(), old_len.max(requested));
        assert_eq!(scratch.buf.as_ptr(), pointer);
        let (preserved, extension) = scratch.buf.split_at(old_len);
        assert_eq!(preserved, prefix);
        assert!(extension.iter().all(|limb| *limb == 0));
        scratch.buf.fill(Limb::MAX);
    }
    let larger = scratch
        .buf
        .capacity()
        .checked_add(1)
        .expect("test growth fits");
    scratch.prepare(larger);
    assert_eq!(scratch.buf.len(), larger);
    assert!(scratch.buf.iter().all(|limb| *limb == 0));
}
