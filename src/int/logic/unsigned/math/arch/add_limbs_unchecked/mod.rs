//! Addition into an initialized limb span.
//!
//! For `B = 2^Limb::BITS`, the result satisfies
//! `dst + carry * B^len = original_dst + src`, with `carry` in `{0, 1}`.
//! Source and destination spans are disjoint or exactly identical.

#![expect(
    unsafe_code,
    reason = "Raw pointer access and inline assembly implement the limb-span contract"
)]

use super::Limb;

select_arch_kernel! {
    function: add_limbs_unchecked;
    surface: direct;
    backends: [
        x86 => all(not(miri), target_arch = "x86", target_pointer_width = "32"),
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        arm => all(not(miri), target_arch = "arm", not(target_feature = "thumb-mode")),
        powerpc => all(not(miri), target_arch = "powerpc"),
        s390x => all(not(miri), target_arch = "s390x"),
        riscv64 => all(not(miri), target_arch = "riscv64"),
        riscv32 => all(not(miri), target_arch = "riscv32"),
        loongarch64 => all(not(miri), target_arch = "loongarch64"),
        loongarch32 => all(not(miri), target_arch = "loongarch32"),
        mips64 => all(not(miri), target_arch = "mips64"),
        mips => all(not(miri), target_arch = "mips"),
    ];
    x86_64: [baseline];
    powerpc64: [baseline];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        target_arch = "powerpc64",
    ];
    fallback_imports: [];
}

#[cfg(test)]
mod tests;
