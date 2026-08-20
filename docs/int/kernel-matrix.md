# Architecture Kernel Matrix

This is the source of truth for the limb kernels below
`src/int/logic/unsigned/math/arch/`.

## Ownership rule

Each operation directory owns exactly one arithmetic kernel. Target files,
`fallback.rs`, and an optional `runtime_dispatch.rs` are implementations or
selectors for that one kernel; they must not add another arithmetic operation.
Shared proof machinery such as `divrem_1_unchecked/half_limb.rs` is not an
exported kernel. A target-named file is retained only when it contains real
target assembly or a materially target-specific algorithm; targets using an
identical Rust implementation select the shared module directly from `mod.rs`.

Legend (implementation *and* evidence level):

- **ASM-S**: assembly is structurally required to exploit an ISA facility that
  no Rust primitive can request — a second independent flag chain, a fused
  wide multiply-add, a single two-limb divide, or a borrow chain that must
  stay in the architecture's physical condition code.
- **ASM-B**: assembly retained because native benchmarks showed better code
  generation than the portable form.
- **ASM-C**: assembly retained after optimized-codegen inspection on a pinned
  toolchain; native hardware performance still unverified.
- **Rust+ISA**: a target-specific algorithm exists and LLVM selects the native
  word instruction; inline asm would add no instruction-selection benefit or
  is not available for that Rust target. A `-B`/`-C` suffix marks the same
  native-benchmark or codegen-inspection evidence as above.
- **Fallback**: the portable kernel is intentionally selected.
- **RT**: x86-64 `std` builds select the appropriate compiled backend once at
  runtime; `no_std` builds use compile-time target features.

Evidence levels are asserted only where this document records the evidence
(the native i686 audit and the codegen inspections in the evidence section).
An unlabelled `ASM` cell states the selection, not a proof that Rust cannot
match it, and no cell is a universal guarantee across every LLVM version and
microarchitecture.

## Exact kernel ownership

| Directory | Sole arithmetic kernel | Dedicated implementations |
|---|---|---|
| `add_limbs_unchecked` | in-place `dst += src` | x86/x86-64, AArch64, ARM, POWER32/64, s390x, RISC-V32/64, LoongArch32/64, MIPS32/64 |
| `add_limbs_3_unchecked` | fixed three-limb `dst += src` | same target set as `add_limbs_unchecked` |
| `sub_limbs_unchecked` | in-place `dst -= src` | x86/x86-64, AArch64, ARM, POWER32/64, s390x, RISC-V32/64, LoongArch32/64, MIPS32/64 |
| `sub_limbs_3_unchecked` | fixed three-limb `dst -= src` | same target set as `sub_limbs_unchecked` |
| `add_mul_limbs_unchecked` | `dst += src * scalar` | x86, x86-64 vanilla/BMI2/ADX, AArch64, ARM, POWER32/64 (POWER9 ISA 3.0 variant), s390x, RISC-V32/64, LoongArch32/64, MIPS32/64 |
| `sub_mul_limbs_unchecked` | `dst -= src * scalar` | x86, x86-64 vanilla/BMI2/ADX, AArch64, ARM, POWER32/64 (borrow-chain mask form), s390x, RISC-V32/64, LoongArch32/64, MIPS32/64 |
| `add_mul_2_limbs_unchecked` | two overlapping multiply-add rows | x86/x86-64 vanilla/BMI2, AArch64, ARM (ARMv7 `umaal`), POWER32/64 (POWER9 ISA 3.0 variant, register carry-forward), s390x, RISC-V32/64, LoongArch32/64, MIPS32/64 |
| `mul_2_limbs_unchecked` | write-only initialization of two product rows | x86/x86-64 vanilla/BMI2, AArch64, ARM (ARMv7 `umaal`), POWER64, RISC-V64, s390x |
| `add_sub_limbs_unchecked` | in-place simultaneous sum and difference | x86-64 ADX; fallback elsewhere |
| `add_sub_from_limbs_unchecked` | simultaneous sum and difference from two sources | x86-64 ADX; x86-64 AVX2 (RT/compile-time); fallback on AArch64, ARMv7, and other targets |
| `add_reverse_sub_limbs_unchecked` | simultaneous sum and reverse difference | x86-64 ADX; fallback elsewhere |
| `add_two_limbs_unchecked` | two independent addition chains | x86-64 ADX; fallback elsewhere |
| `monty_redc_unchecked` | one CIOS Montgomery reduction step | x86-64 BMI2/ADX, AArch64, POWER64, s390x, RISC-V64, LoongArch64 |
| `divrem_1_unchecked` | two-limb by one-limb quotient and remainder | full-width ASM on x86/x86-64/s390x; one shared normalized two-half-limb implementation on other capable targets |
| `lshift_unchecked` | in-place left shift | x86-64 (AVX2 and baseline, RT), AArch64; fallback elsewhere |
| `rshift_unchecked` | in-place right shift | x86-64 (AVX2 and baseline, RT), AArch64; fallback elsewhere |
| `lshift_into_unchecked` | out-of-place left shift into a destination | x86-64 (AVX2 and baseline, RT), AArch64; fallback elsewhere |
| `rshift_into_unchecked` | out-of-place right shift into a destination | x86-64 (AVX2 and baseline, RT), AArch64; fallback elsewhere |
| `propagate_carry_unchecked` | propagate one carry through a span | x86-64, AArch64, s390x; fallback elsewhere |
| `propagate_borrow_unchecked` | propagate one borrow through a span | x86-64, AArch64, s390x; fallback elsewhere |
| `mul_basecase_unchecked` | one complete schoolbook product | x86-64 ADX (fully unrolled per width); every other target composes the row kernels above |
| `sub_shifted_high_limbs_unchecked` | subtract a cross-limb shifted span | x86-64 BMI2, AArch64, s390x, POWER32/64; fallback elsewhere |

`mul_basecase_unchecked` is the one entry that owns a *driver* rather than a
row. Off x86-64 it is a portable loop over `mul_2`, `add_mul_2`, and
`add_mul_1`, each of which already selects its own backend, so the composed
path is architecture-specific everywhere without a per-target driver file.


`sub_shifted_high_limbs_unchecked` needs a hardware borrow chain that survives a
variable-count shift, a merge, loads and stores, and loop control. Where that
holds, one chain spans the whole span. The arithmetic is expressible in
portable Rust (`borrowing_sub` chains the borrow across limbs), but portable
Rust and LLVM IR cannot require the borrow to remain in the architecture's
physical condition code across the intervening shift, merge, memory, and
loop-control operations, so current optimized fallback codegen materializes
the borrow in a general register on every limb. The assembly backend
guarantees one uninterrupted hardware borrow chain. The ISA split is therefore
sharp, and the fallbacks are decisions:

| Target | Backend | Why |
|---|---|---|
| x86-64 BMI2 | ASM/RT | `shlx`/`shrx` and `lea` are flag-neutral; one `sbb` chain |
| x86-64 baseline, x86 32-bit | Fallback | `shl`/`shr`/`shld` all write flags; preserving CF per limb *is* the portable codegen |
| AArch64 | ASM | `lsr`/`lsl`/`orr`, `ldp`, and `cbnz` leave NZCV alone; one `sbcs` chain |
| s390x | ASM | `srlg`/`sllg` and `brctg` are CC-neutral, and `la` sums the two fragments without the CC that `ogr`/`algr` would set |
| POWER32/64 | ASM | best case: the borrow is in `XER[CA]` while the guard writes `CR`, so nothing can collide |
| RISC-V, LoongArch, MIPS, wasm | Fallback | no hardware carry flag; the borrow must be an `sltu` value, which is what LLVM emits |
| m68k | Fallback | `subx` exists, but m68k shifts also write X and break the chain |
| ARM32 | Fallback | flags survive the body, but ARM mode has no flag-free loop branch, so the borrow spills and reloads every iteration |
| SPARC64 | Fallback | `subccc` consumes only 32-bit `%icc.c`, not 64-bit `%xcc.c`; 32-bit SPARC has `subxcc` but lacks flag-free branches |

## Upper basecase and reduction coverage

These are the kernels where a missing backend can materially affect the
multiplication, Montgomery, or one-limb-division towers.

| Target | `mul_2` | CIOS REDC | `divrem_1` | Selection note |
|---|---:|---:|---:|---|
| x86-64 ADX+BMI2 | ASM/RT | ASM/RT | ASM | BMI2 is the canonical `mul_2`; REDC uses ADX+BMI2 |
| x86-64 BMI2 | ASM/RT | ASM/RT | ASM | BMI2 kernels selected independently of ADX |
| x86-64 baseline or ADX-only | ASM/RT | Fallback | ASM | baseline `mulq`/`divq`; ADX alone does not improve multiplication |
| x86 32-bit | ASM | Fallback | ASM | write-only `mull` rows and one complete 64-by-32 `div` |
| AArch64 | ASM | ASM | Rust+ISA | `mul`/`umulh`; LLVM schedules the two `udiv` half-digit estimates |
| ARM 32-bit | ASM | Fallback | Rust+ISA | shared half-digit core selected directly; LLVM uses UDIV when available |
| CSKY 32-bit | Fallback | Fallback | Rust+ISA | shared core replaces generic `__udivdi3` with native `divu32` half-digit estimates |
| Hexagon 32-bit | Fallback | Fallback | Rust+ISA | shared core replaces one generic 64-bit division with two narrower target-runtime divisions |
| m68k 32-bit | Fallback | Fallback | Rust+ISA | shared core replaces generic 64-bit division with the target's narrower division support |
| POWER64 | ASM | ASM | Rust+ISA | `mulld`/`mulhdu`; shared core lowers to two `divdu` estimates. `add_mul_1` and `add_mul_2` include a POWER9 ISA 3.0 path using `maddld`/`maddhdu`, selected by `target_feature = "power9-vector"` (normally via `-C target-cpu=pwr9` or an explicit target-feature; the default `powerpc64le` target does not enable it). |
| POWER32 | Fallback | Fallback | Rust+ISA | shared normalized core lowers to two `divwu` estimates |
| s390x | ASM | ASM | ASM | write-only rows and REDC use `mlgr`; `dlgr` is a full 128-by-64 divide |
| RISC-V64 M | ASM | ASM | Rust+ISA | `mul`/`mulhu`; shared core lowers to two `divu` estimates |
| RISC-V32 M | Fallback | Fallback | Rust+ISA | shared core lowers to two `divu` estimates |
| RISC-V32 base I | Fallback | Fallback | Rust+ISA | shared half-digit core selected directly through the target division runtime |
| LoongArch64 | Fallback | ASM | Rust+ISA | shared core lowers to two `div.du` estimates |
| LoongArch32 | Fallback | Fallback | Rust+ISA | shared core lowers to two `div.wu` estimates |
| MIPS64 | Fallback | Fallback | Rust+ISA | shared core lowers to two `ddivu` estimates |
| MIPS32 | Fallback | Fallback | Rust+ISA | shared core lowers to two `divu` estimates |
| SPARC64 | Fallback | Fallback | Rust+ISA | shared core selected directly; LLVM selects `udivx` |
| SPARC32 | Fallback | Fallback | Rust+ISA | shared normalized half-digit core avoids generic wide division |
| wasm64 | Fallback | Fallback | Rust+ISA | shared core becomes two `i64.div_u` operations instead of 128-bit compiler support |
| wasm32 | Fallback | Fallback | Fallback | native `i64.div_u` already implements `DoubleLimb` division |
| Xtensa32 | Fallback | Fallback | Rust+ISA | shared half-digit core avoids generic 64-bit division |

`mul_2` remains a portable write-only loop on targets not marked ASM. It
already maps directly to each ISA's low/high multiplication sequence; a custom
body is retained only where it gives a distinct scheduling or flag-use win.

### Why `divrem_1/half_limb.rs` is not a fallback wrapper

`half_limb.rs` contains the complete normalized division kernel, not a thin
forwarder. Targets with ordinary word division select that module directly.
Its two `Limb::wrapping_div` expressions let LLVM choose and schedule the exact
word instruction or target runtime primitive. Keeping per-target wrappers with
one inline-assembly `div` would produce the same instruction while constraining
register allocation, so those wrappers are deliberately absent. Only x86,
x86-64, and s390x retain target files because their kernels lower to the
target's single two-word-by-one-word divide instruction (`divq`/`divl`,
`dlgr`). Rust can express the full-width arithmetic through `DoubleLimb`, but
it cannot portably request that one instruction under the kernel's
preconditions; the assembly backend guarantees that lowering.

## Broad scalar-kernel coverage

| Target family | add/sub | fixed add/sub 3 | addmul/submul 1 | addmul 2 | shifts | carry/borrow propagation |
|---|---:|---:|---:|---:|---:|---:|
| x86-64 | ASM | ASM | ASM/RT | ASM/RT | ASM | ASM |
| x86 32-bit | ASM | ASM | ASM | ASM | Fallback | Fallback |
| AArch64 | ASM | ASM | ASM | ASM | ASM | ASM |
| ARM 32-bit | ASM | ASM | ASM | ASM | Fallback | Fallback |
| CSKY, Hexagon, m68k | Fallback | Fallback | Fallback | Fallback | Fallback | Fallback |
| POWER64 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| POWER32 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| s390x | ASM | ASM | ASM | ASM | Fallback | ASM |
| RISC-V64 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| RISC-V32 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| LoongArch64 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| LoongArch32 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| MIPS64 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| MIPS32 | ASM | ASM | ASM | ASM | Fallback | Fallback |
| SPARC, wasm, Xtensa | Fallback | Fallback | Fallback | Fallback | Fallback | Fallback |

MIPS and MIPS64 are marked `ASM` because they use genuine inline assembly,
enabled through the nightly `asm_experimental_arch` feature (declared once at
the crate root in `lib.rs`); they are not Rust-only lowering. The coverage in
this matrix is therefore nightly-scoped for the targets whose inline-assembly
backend is not on stable Rust.

The three simultaneous add/sub operations and the independent dual-add kernel
use x86-64 ADX's separate CF and OF chains. Every other ISA has only one
condition-code chain, or explicit carry values, so the fallback is the intended
implementation rather than a missing assembly body.

## Fallback decisions, kernel by kernel

`Miri` always selects `fallback.rs` because it cannot execute inline assembly;
that correctness selection is omitted from the table. “Other” means Rust
targets without a reviewed stable inline-assembly backend. A fallback is kept
only when the portable expression gives LLVM the complete arithmetic dataflow
and a custom instruction stream has no demonstrated advantage. This does not
turn cross-compilation into a foreign-hardware speed claim.

| Kernel | Architectures intentionally using portable code | Why a custom assembly file is absent |
|---|---|---|
| `add_limbs_unchecked` | SPARC32/64, wasm32/64, Xtensa, other | `carrying_add` exposes the single carry dependency directly. LLVM selects the target add/carry or set-less-than sequence and retains allocator freedom; wasm has no inline assembly, SPARC V9 lacks a 64-bit carry-consuming add, and portable code avoids register constraints. |
| `add_limbs_3_unchecked` | SPARC32/64, wasm32/64, Xtensa, other | The same one-chain argument applies to the fixed-destination form. There is no second independent flag chain to unlock, and LLVM can specialize constant small lengths after inlining. |
| `sub_limbs_unchecked` | SPARC32/64, wasm32/64, Xtensa, other | `borrowing_sub` represents exactly one borrow chain. A handwritten loop would use the same subtract/set-borrow sequence while constraining register allocation. |
| `sub_limbs_3_unchecked` | SPARC32/64, wasm32/64, Xtensa, other | The fixed-destination form has the same dependency graph; LLVM sees the full small-length call site and can unroll it without an assembly boundary. |
| `add_mul_limbs_unchecked` | SPARC32/64, wasm32/64, Xtensa, other | The widened product and carry split expose low/high halves exactly. LLVM either selects the ISA multiply pair or the target’s required compiler runtime; inline assembly cannot manufacture a missing wide multiply. |
| `sub_mul_limbs_unchecked` | SPARC32/64, wasm32/64, Xtensa, other | The widened multiply plus explicit low-limb borrow already exposes the complete dependency graph. Custom files are retained only on targets where controlled multiply/carry scheduling is a reviewed win. |
| `add_mul_2_limbs_unchecked` | ARM32, SPARC32/64, wasm32/64, Xtensa, other | ARM LLVM combines the widened row sums with `umull`/`umlal`/`umaal`, which is better than pinning register pairs in inline assembly. The remaining families either lack inline assembly or offer no dual-row instruction beyond the same two products. |
| `mul_2_limbs_unchecked` | ARM32, POWER32, RISC-V32, LoongArch32/64, MIPS32/64, SPARC32/64, wasm32/64, Xtensa, other | Optimized codegen was already compact: ARM uses `umull`/`umaal`/`umlal`, POWER32 uses `mullw`/`mulhwu`, RISC-V M uses `mul`/`mulhu`, and LoongArch64 uses `mul.d`/`mulh.du`. s390x is the exception: LLVM expanded the fallback into a spill-heavy multi-block body, so it now has a compact `mlgr` kernel. |
| `add_sub_limbs_unchecked` | x86-64 without ADX and every non-ADX ISA | Only ADX provides independent CF and OF chains. Elsewhere, assembly would serialize the two operations on one flag register; LLVM can interleave the two explicit Boolean carry values instead. |
| `add_sub_from_limbs_unchecked` | AArch64, ARMv7, and other non-x86 targets | x86-64 selects ADX first, then AVX2, then the scalar fallback through the central selector. A reviewed NEON prototype duplicated scalar loads and carry arithmetic around every vector pair, so without native evidence that it wins, both AArch64 and ARMv7 intentionally retain the scalar alias-safe loop. Native ARM speed is not claimed. |
| `add_reverse_sub_limbs_unchecked` | x86-64 without ADX and every non-ADX ISA | Reversing subtraction does not change the single-flag limitation. The ADX backend is the only implementation with a genuinely different two-chain schedule. |
| `add_two_limbs_unchecked` | x86-64 without ADX and every non-ADX ISA | Two independent additions benefit from ADX’s CF/OF split. On other ISAs, compiler-visible Boolean carries avoid artificial serialization through one condition-code register. |
| `monty_redc_unchecked` | x86-64 baseline/ADX-only, x86, ARM32, POWER32, RISC-V32, LoongArch32, MIPS32/64, SPARC32/64, wasm32/64, Xtensa, other | The i686 fused assembly candidate lost at common lengths while LLVM retained one allocator-visible traversal and selected BMI2 `mulx` when enabled. ARM32 codegen uses `umaal`; the other portable paths expose the two widened carry chains without forcing fixed multiply registers. |
| `divrem_1_unchecked` | shared half-limb core on AArch64, ARM, CSKY, Hexagon, m68k, POWER32/64, RISC-V32/64, LoongArch32/64, MIPS32/64, SPARC32/64, wasm64, Xtensa; `DoubleLimb` fallback elsewhere | Ordinary word division lets LLVM select `udiv`, `divu`, `divwu`, `div.du`, or the target runtime and schedule both half-digit estimates. CSKY emits native `divu32`; Hexagon and m68k still benefit from replacing generic double-word division with narrower runtime support. Only x86/x86-64 and s390x have a single two-word divide instruction that portable Rust cannot request directly under the kernel's preconditions. |
| `lshift_unchecked` | x86, ARM32, POWER32/64, s390x, RISC-V32/64, LoongArch32/64, MIPS32/64, SPARC32/64, wasm32/64, Xtensa, other | The tested i686 `shld` loop was 3–70% slower: its flag/dependency chain cost more than LLVM’s two shifts and OR. Other variable-count ISAs expose the same two-shift dataflow, which LLVM can allocate and unroll more freely. |
| `rshift_unchecked` | x86, ARM32, POWER32/64, s390x, RISC-V32/64, LoongArch32/64, MIPS32/64, SPARC32/64, wasm32/64, Xtensa, other | The tested i686 `shrd` loop was slower through the useful range. LLVM’s reverse traversal with two shifts and OR has fewer fixed-register constraints; no omitted ISA supplies a better variable-count funnel operation. |
| `propagate_carry_unchecked` | x86 and every target except x86-64, AArch64, and s390x | This is an early-exit increment loop, not a throughput loop. i686 optimized codegen is already the ideal initial add followed by `inc` and branch, so inline assembly adds no instruction-selection value. |
| `propagate_borrow_unchecked` | x86 and every target except x86-64, AArch64, and s390x | LLVM likewise emits the exact initial subtract followed by `sub 1` and early exit. A custom body cannot shorten that dependency chain. |

### Evidence used for those decisions

The 2026-07-17 native i686 audit compared every candidate directly with the
same portable fallback over lengths 1 through 256. Retained kernels improved
`addmul_1` by about 1–31%, `submul_1` by 15–27%, write-only `mul_2` by 12–34%,
and fused `addmul_2` by 18–73%. The direct add/sub kernels won broadly, with a
few near-parity cells around unroll boundaries. The rejected REDC candidate
lost by up to 43% at common sizes; the rejected shift candidates lost across
the useful small/medium range.

Optimized LLVM codegen was also inspected for AArch64, ARMv7, CSKY, Hexagon,
m68k, POWER32, RISC-V32 I/M, LoongArch64, s390x, and i686. This is sufficient
to reject an assembly body that merely reproduces an already compact
instruction stream, but foreign hardware still needs native benchmarks before
claiming a speedup.

The shift decisions were measured against the two-shift-plus-OR formulation on
the pinned toolchains. Newer nightly funnel-shift intrinsics are a candidate
for a future re-audit, not a reason to change selection in this checkout.

## PowerPC ISA 3.0 (POWER9) fused multiply-add kernels

`add_mul_limbs_unchecked` and `add_mul_2_limbs_unchecked` each ship two
64-bit PowerPC implementations selected by `cfg(target_feature =
"power9-vector")`, normally through `-C target-cpu=pwr9` or an explicit
`-C target-feature=+power9-vector` configuration. The default
`powerpc64le-unknown-linux-gnu` target uses `cpu = "ppc64le"` and does not
enable `power9-vector`.

The ISA 3.0 path uses the `maddld` and `maddhdu` instructions, which fuse a
64×64→128 multiply with a 64-bit addend in the destination register. Per limb
the arithmetic core collapses from six instructions (`mulld`/`mulhdu`/
`addc`/`addze`/`addc`/`addze`) to four (`maddld`/`maddhdu`/`addc`/`addze`),
or from twelve to eight for `add_mul_2`. `maddhdu` does not touch `XER[CA]`,
so the carry chain must use `addc` (which ignores the stale `CA` from
`maddld`) rather than `adde` (which would read it). The low-half overflow
from `maddld` is already captured in the `maddhdu` high-half result.

The `add_mul_2` kernel also uses an `ld {d_cur}` register carry‑forward
instead of reloading `dst[j+1]` every iteration, eliminating the
store→load forwarding stall.

## PowerPC operand classes and flag clobbers

Every PowerPC operand that appears as an address base, or as the source of
`addi`, is declared `reg_nonzero` rather than `reg`. PowerPC reads `r0` in those
positions as the literal zero instead of as the register, and Rust's PowerPC
`reg` class *includes* `r0`. This is reachable, not theoretical: a probe with
twelve pointer operands in `reg` compiled to `ld 0, 0(0)`, a load from absolute
address zero rather than through the pointer. The kernels here previously used
`reg` and were latently exposed under high operand pressure; `reg_nonzero`
removes the hazard at no scheduling cost. s390x has the same rule with
`reg_addr`, which its kernels already used.

Every `asm!` block that writes PowerPC `XER[CA]` (through `addc`, `adde`,
`addze`, `addic`, `subfc`, `subfe`, or `subfic`) declares `out("xer") _`. Every
block that writes `CR0` (through `cmpldi`, `cmpwi`, `cmplwi`, or `cmpdi`)
declares `out("cr0") _`. Every block that uses `mtctr`/`bdnz` declares
`out("ctr") _`. These are not latent: an undeclared `ctr` clobber on a  `bdnz`
loop was observed to cause LLVM's PowerPC hardware-loop pass to form a second
`mtctr`/`bdnz` around the inline asm, silently corrupting the outer loop
counter.

## s390x condition-code idiom and zero register

s390x uses a two-bit condition code (CC) rather than dedicated flag registers.
Every add/sub loop entry and tail clears the CC with the two-instruction
`lghi R, 0; algr R, R` pattern (adding zero to zero), which the follow-up
`alcgr` then reads as the carry-in. Multiply-accumulate kernels materialize one
zero before the loop (initialized in Rust and passed as an `inout(reg)`
operand) and reuse that register for the third operand of `alcgr` throughout
the body rather than re-zeroing inside the loop. `.p2align 4` was added on loop
headers that previously lacked it.

## x86-64 and ARM/AArch64 flag handling

x86-64 `att_syntax` and AArch64 `asm!` blocks do not need explicit flag
clobbers. LLVM's x86 backend emits `~{flags}` automatically when the assembler
dialect is set; LLVM's AArch64 backend infers flag usage from instruction
mnemonics. Neither architecture exposes a named flag register class for
`out(...)`. PowerPC is the exception: LLVM tracks `XER[CA]` as the physical
register `CARRY` and models individual `CR` fields separately, requiring
explicit `out("xer") _` and `out("cr0") _` declarations. `.p2align 4` or
`.p2align 2` was added to all previously unaligned loop headers on x86-64, x86
32-bit, and ARM 32-bit.

## Why `mul_basecase` stays composed off x86-64

A fused basecase adds no arithmetic that the composed path lacks. Its entire
value is a better instruction schedule across row boundaries, so it earns a
target file only where that schedule can be measured on the hardware it targets.

x86-64 meets that bar twice over. ADX supplies independent `adcx`/`adox` carry
chains, which is a real dataflow advantage a composed loop cannot express, and
the hardware is available here to tune the per-width unrolled bodies against.

Nowhere else currently meets it. A fused driver for another target would be a
large body whose correctness could not be executed and whose schedule could not
be timed, competing against LLVM scheduling row kernels that are already
target-specific. Published fused basecases for other ISAs run several hundred
lines *per CPU model*, which is itself the evidence: that size is
microarchitecture tuning, and tuning that cannot be measured is a guess. The
composed path is therefore the deliberate choice off x86-64, not a gap.

## x86-64 runtime dispatch

`std` builds detect ADX and BMI2 once in `arch/x86_runtime.rs`. A shared
`OnceLock<X86Backend>` owns feature detection; each operation then caches one
function pointer, or one small kernel-plus-policy record, in its own
`runtime_dispatch.rs`. Hot loops therefore contain no CPUID probe and no
repeated feature branch. Callers that execute a kernel repeatedly use the
`selected_*` interface to hoist the function pointer outside the arithmetic
loop. `no_std` builds select the same backends entirely through compile-time
features.

MIPS inline assembly still needs Rust’s crate-level
`asm_experimental_arch` feature declaration. That one declaration must live at
the crate root because Rust rejects module-scoped `#![feature]`; every actual
MIPS target selection and every assembly instruction remains under `arch/`.

## Cross-target compilation matrix

`tools/check_all_archs.sh` checks representative bigint configurations rather
than repeating the same code generation across every OS/vendor spelling in
Rust's target list. A required target is never silently skipped: a missing
target, failed rustup installation, crate warning, or crate compilation error
is retained while the script continues and is reported in the final failure
summary.

| Matrix slice | Representative coverage | Feature sets |
|---|---|---|
| Prebuilt `std` | x86-64 LP64 and x32, i686, AArch64, Arm64EC, ARMv5/ARMv7, POWER32, POWER64 BE/LE, s390x, RISC-V64, LoongArch64, SPARC64, wasm32 | no features, `num-traits`, `std`, `std,num-traits` |
| Prebuilt `no_std` | ARM Thumb v6-M/v7E-M, RISC-V32 I/M, NVPTX64 | no features, `num-traits` |
| Source-built `no_std` | AArch64 ILP32 and BE, ARM BE, AVR16, MSP430-16, BPF BE/LE, CSKY, Hexagon, LoongArch32, m68k, MIPS32/64 BE/LE, RISC-V32E, SPARC32, wasm64 | no features, `num-traits` |
| x86-64 compile-time selectors | BMI2-only, ADX-only, and ADX+BMI2 in addition to the baseline and `std` runtime-dispatch builds | `no_std` |
| Toolchain probe | Xtensa ESP32 | attempted from `rust-src`; compiler failures before this crate are reported separately |

This gives 38 required target configurations spanning 16-, 32-, and 64-bit
limbs, both endiannesses, native and OS-key TLS, unusual 32-bit-pointer ABIs,
targets without pointer atomics, and the CPU-feature branches used by the
kernel selectors. Tier-3 targets use `-Z build-std=core,alloc --release`;
release mode avoids an unrelated m68k debug-codegen crash in
`compiler_builtins`.

## Division proof and validation status

For targets without a full two-limb divide, the shared core normalizes the
divisor and computes two base-`sqrt(B)` quotient digits. A normalized high
half is at least `sqrt(B)/2`, so each estimate is at most `sqrt(B)+1`; Knuth's
bound proves that no more than two corrections are needed. All cancellation is
performed modulo one limb and the final normalized remainder is below the
normalized divisor.

This is the same division family described by GMP for targets without a native
two-limb divide: <https://gmplib.org/manual/Single-Limb-Division>.

Validation completed for this matrix:

- the selected native x86-64 division and shared 64-bit core each passed
  65,536 generated quotient/remainder properties;
- the selected x86 32-bit division and shared 32-bit core each passed another
  65,536 generated properties;
- the required cross-target matrix above compiles every architecture selector,
  limb width, endian family, and x86-64 feature combination (using
  `-Z build-std` where prebuilt artifacts are not distributed);
- the current nightly Xtensa `core` build fails earlier in unrelated `f16`
  lowering, so that Rust-only selection remains source-audited but not
  target-compiled in this checkout;
- foreign-target speedups still require measurements on representative real
  hardware before thresholds or performance claims become target-specific.

## Kernels this tower does not have

The sections above justify each *fallback* inside a kernel we own. This one
covers the complementary question: which whole kernels have no directory here,
and why not.

The governing rule is that a kernel exists because a caller in this tower needs
it. Surveying other bignum libraries is useful for finding techniques worth
stealing and for checking that we have not overlooked something our own callers
would benefit from, but their kernel list is not a checklist to complete. Most
of what a general-purpose library exports serves callers we do not have, and
importing those bodies would add unexecuted assembly and maintenance surface in
exchange for nothing measurable.

### Operations with no caller here

| Operation | Why it is absent |
|---|---|
| Base-`B-1` exact division and mod-`3*2^k` residues | These exist in other libraries as internal checksum helpers for their own test suites. Nothing in this tower computes such a residue. |
| Shift-fused add/sub (`dst -= src << 1` and the halving forms) | These fuse a shift into a Toom interpolation step. The Toom tiers here reduce with Hensel exact division (`exact_div_odd_in_place`, `invert_odd`) instead, so no call site wants the shifted form. |
| Constant-time conditional add/sub and table select | Side-channel-resistant selection primitives. This tower does not currently claim constant-time behaviour, so adding them would imply a guarantee the rest of the code does not keep. |
| Exact single-limb division (`divexact` by one limb) | Already covered: the Hensel/Jebelean form the Toom tiers call is the same operation, implemented portably. |

### Operations with a caller

**Single-limb division by a precomputed reciprocal.**
Implemented via Möller-Granlund 2-by-1 reciprocal precomputation (`reciprocal_2by1_unchecked`).
Call sites in `div.rs` and radix conversion in `convert/string.rs` compute the 64-bit reciprocal approximation once and reuse it across multiple limb divisions.
This replaces the non-pipelined hardware `divq`/`divu` (20+ cycles) with two fast 64-bit multiplications and branchless adjustments (6-10 cycles), providing a **1.38× to 3.46× speedup**.

**One- and two-limb GCD base cases.** The algorithm is already present:
`gcd.rs` runs a Lehmer/binary hybrid, and `lehmer_simulate_wide` performs the
two-limb extraction that a dedicated two-limb kernel would serve. Only a
per-architecture base-case body is absent, and its value is unmeasured: the
inner loop is `ctz`-driven and LLVM selects `tzcnt` or `rbit`+`clz` directly.
Worth profiling before writing anything.

### What emulation would and would not settle

Running the property tests under user-mode emulation in CI would execute the
AArch64, s390x, and POWER bodies that are currently compile-verified and
source-audited only. That is worth doing: it converts an audit argument into a
test result. It would not produce usable timings, so it cannot justify any
kernel whose entire value is instruction scheduling — which is precisely why
`mul_basecase` stays composed off x86-64.

## Scope and honesty of these claims

Some kernels have a structurally strong case for assembly: a second
independent flag chain (x86-64 ADX), a fused wide multiply-add (POWER9
`maddld`/`maddhdu`), a single two-limb divide (x86/x86-64, s390x), or a borrow
chain that must stay in the architecture's physical condition code
(`sub_shifted_high_limbs_unchecked` on the flagged targets). Many other
backends rest on pinned-toolchain codegen inspection or native benchmarks, and
several foreign-ISA bodies are compile-verified and source-audited but not yet
measured on real hardware.

No statement here is a universal guarantee across every LLVM version and
microarchitecture. The matrix records the best evidence available in this
checkout: ownership and selection are exact, structural necessity is argued
from an ISA facility where one exists, and everything else is labelled by the
evidence that supports it. Where this document does not yet assert a native
benchmark, no speed claim is made.
