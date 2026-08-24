//! Architecture-specific two-limb-by-one-limb division kernels.
//!
//! `x86_64`, `x86`, and `s390x` expose a full-width hardware divide. Targets
//! with only limb-width division share one normalized Knuth two-half-limb
//! algorithm using ordinary limb division, which LLVM can schedule directly.
//! Remaining targets use the portable `DoubleLimb` implementation.

#![allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]

use super::Limb;

select_arch_kernel! {
    function: divrem_1_unchecked;
    surface: direct;
    backends: [
        half_limb => all(
            not(miri),
            any(
                all(target_arch = "aarch64", target_pointer_width = "64"),
                all(target_arch = "arm", target_pointer_width = "32"),
                all(target_arch = "csky", target_pointer_width = "32"),
                all(target_arch = "hexagon", target_pointer_width = "32"),
                all(target_arch = "loongarch32", target_pointer_width = "32"),
                all(target_arch = "loongarch64", target_pointer_width = "64"),
                all(target_arch = "m68k", target_pointer_width = "32"),
                all(target_arch = "mips", target_pointer_width = "32"),
                all(target_arch = "mips64", target_pointer_width = "64"),
                all(target_arch = "powerpc", target_pointer_width = "32"),
                all(target_arch = "powerpc64", target_pointer_width = "64"),
                all(target_arch = "riscv32", target_pointer_width = "32"),
                all(target_arch = "riscv64", target_pointer_width = "64"),
                all(target_arch = "sparc", target_pointer_width = "32"),
                all(target_arch = "sparc64", target_pointer_width = "64"),
                all(target_arch = "wasm64", target_pointer_width = "64"),
                all(target_arch = "xtensa", target_pointer_width = "32"),
            )
        ),
        s390x => all(not(miri), target_arch = "s390x", target_pointer_width = "64"),
        x86 => all(not(miri), target_arch = "x86", target_pointer_width = "32"),
    ];
    x86_64: [baseline];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [DoubleLimb, LIMB_BITS];
}
