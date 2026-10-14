use core::num::NonZeroUsize;

use crate::{LIMB_BITS, Limb, ParallelExecutor, SequentialExecutor};

#[test]
fn standalone_limb_widths_and_sequential_execution_match_mpanafis_contracts() {
    assert_eq!(
        LIMB_BITS,
        usize::try_from(Limb::BITS).expect("native bit width")
    );
    let executor = SequentialExecutor;
    assert_eq!(executor.parallelism(), NonZeroUsize::MIN);
    assert_eq!(executor.join(|| 3, || 5), (3, 5));
}
