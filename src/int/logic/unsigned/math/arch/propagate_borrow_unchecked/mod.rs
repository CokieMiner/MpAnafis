//! Architecture-specific borrow propagation kernels.

#![allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]

use super::Limb;

select_arch_kernel! {
    function: propagate_borrow_unchecked;
    surface: direct;
    backends: [
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        s390x => all(not(miri), target_arch = "s390x"),
    ];
    x86_64: [baseline];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [];
}
