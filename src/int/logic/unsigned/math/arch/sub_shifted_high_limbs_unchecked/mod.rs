//! Cross-limb shifted-high subtraction kernel.
//!
//! A target file exists exactly where the ISA can keep the borrow in hardware
//! across the whole span. That needs a carry/borrow chain plus a variable-count
//! shift, a merge, and loop control that all leave that chain intact. x86-64
//! satisfies it only with BMI2 (`shlx`/`shrx` and `lea` are flag-neutral, while
//! `shl`/`shr`/`shld` are not); `AArch64`, s390x, and POWER satisfy it natively.
//!
//! Everything else keeps the portable loop for a concrete reason rather than
//! for lack of a backend: RISC-V, `LoongArch`, MIPS, and wasm have no hardware
//! carry flag at all, so the borrow must be an explicit comparison value, which
//! is exactly what LLVM already emits; m68k `subx` exists but m68k shifts write
//! the X flag and break the chain; `ARM32` keeps flags through the body but has
//! no flag-free loop branch, so it would have to spill and reload the borrow
//! every iteration; `SPARC64` lacks a 64-bit borrow-consuming subtract (`subccc`
//! only reads `%icc.c`), while 32-bit `SPARC` has `subxcc` but lacks a flag-free
//! loop branch.

#![allow(
    unsafe_code,
    reason = "Raw limb pointers and target assembly implement the validated hot loop"
)]

select_arch_kernel! {
    function: sub_shifted_high_limbs_unchecked;
    kernel: SubShiftedHighKernel;
    surface: selector;
    backends: [
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
        s390x => all(not(miri), target_arch = "s390x"),
        powerpc => all(not(miri), target_arch = "powerpc", target_pointer_width = "32"),
    ];
    x86_64: [fallback, bmi2];
    powerpc64: [baseline];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
        all(target_arch = "powerpc64", target_pointer_width = "64"),
    ];
    fallback_imports: [];
    test_backends: [];
}
