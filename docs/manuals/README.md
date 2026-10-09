# Architecture manuals

This directory contains offline reference documentation for the architecture-specific
arithmetic kernels under [`src/int/logic/unsigned/math/arch/`](../../src/int/logic/unsigned/math/arch/).

These reference specifications guide instruction selection, carry/borrow semantics,
widening multiply idioms, inline assembly constraints, and hardware-feature availability
across the listed instruction sets. The
[kernel matrix](../int/kernel-matrix.md) records implemented backends and
selection gates separately from this reference collection.


| Architecture | Subdirectory | Reference Manuals | Primary Subsystem Scope |
| --- | --- | --- | --- |
| **ARM / AArch32** | [`arm/`](arm/) | `ARMv7-A-R-manual.pdf` | ARMv6/v7 `umaal`, paired-word multiplication, DSP extensions. |
| **AArch64** | [`arm/`](arm/) | `DDI0487G_b_armv8_arm.pdf` | 64-bit widening multiply (`mul`/`umulh`), carry propagation (`adcs`/`sbcs`). |
| **LoongArch64** | [`loongarch64/`](loongarch64/) | `LoongArch-Vol1-EN.pdf` | Loongson 64-bit basic architecture, arithmetic carry idioms. |
| **MIPS64** | [`mips/`](mips/) | `MD00087-2B-MIPS64BIS-AFP-6.06.pdf` | MIPS64 doubleword arithmetic, high/low multiplication register pairs. |
| **PowerPC / POWER** | [`powerpc/`](powerpc/) | `PowerISA_v3.1B.pdf` | Power ISA 3.1B, `mulhdu`/`mulld`, carry manipulation (`addc`/`adde`). |
| **RISC-V** | [`riscv/`](riscv/) | `riscv-unprivileged.pdf`, `riscv-privileged.pdf` | RV32/RV64 M-extension (`mulhu`/`mul`), Zba/Zbb bit manipulation. |
| **s390x** | [`s390x/`](s390x/) | `zarch-principles-of-operations.pdf` | IBM z/Architecture 64-bit arithmetic, condition codes, logical addition/subtraction. |
| **SPARC** | [`sparc/`](sparc/) | `sparcv9.pdf` | SPARC V9 64-bit integer instructions, condition register flags. |
| **x86 / x86-64** | [`x86/`](x86/) | `AMD64-APM-Vol-3.pdf`, `Intel-SDM-Vol-2.pdf` | ADCX/ADOX multi-precision carry chains, BMI2 (`mulx`), AVX2/IFMA vector butterfly. |
| **Xtensa** | [`xtensa/`](xtensa/) | `Xtensa-ESP32-ref.pdf`, `xtensa-isa-reference.pdf` | ESP32 Tensilica core, 32-bit multiply/accumulate primitives. |
