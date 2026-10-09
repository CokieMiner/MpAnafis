# Low-level arithmetic kernel contracts and implementation matrix

This document specifies the low-level arithmetic kernel contracts, interfaces,
and architecture backend selection in `src/int/logic/unsigned/math/arch/`.

## 1. Architecture and admission evidence

Generic arithmetic calls the [ArchKernels facade](../../src/int/logic/unsigned/math/arch/kernels.rs)
and its [shift methods](../../src/int/logic/unsigned/math/arch/shifts.rs),
or obtains an operation-specific function pointer with a
[kernel signature](../../src/int/logic/unsigned/math/arch/signatures.rs). Operation modules declare
fallbacks and architecture providers. [Kernel selection](../../src/int/logic/unsigned/math/arch/kernel_selection.rs),
[backend providers](../../src/int/logic/unsigned/math/arch/backend_providers.rs), and
[x86 selectors](../../src/int/logic/unsigned/math/arch/x86_selectors.rs) implement
compile-time and runtime selection. Complete multiplication and squaring select
their backend outside their inner row loops.

Assembly admission distinguishes two kinds of justification:

- **ASM-S, structural rationale:** explicit flag lifetimes, independent ADX
  carry chains, or bounded double-limb division under caller-proved invariants.
- **ASM-B, benchmark evidence:** a repeatable benefit over the relevant Rust or
  assembly baseline, recording CPU, compiler, features, operand shapes, dispatch
  mode, and measurement method. SIMD intrinsics require the same evidence when
  replacing a production path.

## 2. Contract conventions

Let $W = \texttt{usize::BITS}$, $B = 2^W$, and
$[a]_n = \sum_{i=0}^{n-1} a[i]B^i$. Limb widths are 16, 32, and 64 bits.
Subscript zero denotes a value before mutation; a prime denotes its result.
`Limb` is `usize`. Scalar inputs are limbs unless narrower bounds are stated.

Pointers must meet the selected function's alignment, validity, and span
requirements. Reads require initialized memory. Write-only outputs need not
contain initialized values unless the interface explicitly requires them.
Callers prove representable sizes and valid allocations before pointer
arithmetic, including `n + 1`, `n + 2`, `2 * n`, `len_a + len_b`, and `offset + n`.

### 2.1 Aliasing

| Operation | Permitted aliasing |
| --- | --- |
| In-place addition/subtraction | Source and destination are disjoint or exactly equal; partial overlap is excluded. |
| Three-address addition/subtraction | Destination is disjoint from both sources; read-only sources may overlap. |
| Scalar and paired-row multiplication | Destination is disjoint from source, including any extra output limbs. |
| `add_two_limbs` | Each destination may exactly equal its corresponding source; otherwise that pair is disjoint. Destinations are mutually disjoint and disjoint from the other stream's source. Read-only sources may overlap. |
| `add_sub_limbs`, `add_reverse_sub_limbs` | The two read/write spans are disjoint. |
| `add_sub_from_limbs` | `sum` is disjoint from `source` and `difference`; `difference` is disjoint from `source` or exactly equal to it. |
| Montgomery step | Output is disjoint from both inputs; read-only inputs may overlap. |
| Out-of-place shifts, shifted-high subtraction | Source and destination are disjoint. |
| Overlapping left shift | One allocation covers the prefix and destination suffix. A nonnegative limb offset permits identical or overlapping spans; traversal is descending. |
| Complete multiplication/squaring | Follow the selected complete backend's disjoint-span contract. Multiplication requires `len_a >= 2` and `len_b > 0`; squaring requires disjoint input and output. |

### 2.2 Empty spans

Kernel return contracts for empty spans ($n = 0$):

- Addition, subtraction, scalar multiply-add, and butterflies return zero
  carry/borrow; multiply-subtract and paired multiply-add return `(0, 0)`.
- Carry/borrow propagation returns its incoming limb unchanged at `n = 0`.
  Shifted-high subtraction returns its incoming borrow unchanged.
- Ordinary shifts and the Montgomery step return zero at `n = 0`.
- Write-only paired multiplication returns without initializing its nominal two
  extra output limbs at `n = 0`. Squaring also performs no writes.
- Complete multiplication requires `len_a >= 2` and `len_b > 0`. Arithmetic
  callers handle zero operands before entering the kernel.

Shift-count and incoming-borrow bounds still apply to empty shift calls. Empty
handling does not relax explicit pointer requirements of a selected interface.

## 3. Arithmetic contracts

`selected_*` facade methods return pointers to the corresponding operation.
The signatures below retain the actual argument and return ordering.

### 3.1 Addition, subtraction, and propagation

| Signature | Exact relation |
| --- | --- |
| `add_limbs_unchecked(dst, src, n) -> Limb` | $D' + cB^n = D_0 + S$, $c \in \{0,1\}$. |
| `add_limbs_3_unchecked(dst, src1, src2, n) -> Limb` | $D' + cB^n = S_1 + S_2$; destination is write-only. |
| `sub_limbs_unchecked(dst, src, n) -> Limb` | $D' - bB^n = D_0 - S$, $b \in \{0,1\}$. |
| `sub_limbs_3_unchecked(dst, src1, src2, n) -> Limb` | $D' - bB^n = S_1 - S_2$; destination is write-only. |
| `propagate_carry_unchecked(dst, n, carry) -> Limb` | $D' + cB^n = D_0 + carry$. |
| `propagate_borrow_unchecked(dst, n, borrow) -> Limb` | $D' - bB^n = D_0 - borrow$. |

Propagation takes an explicit incoming carry/borrow in `{0, 1}`. This bound
applies across all backends, including empty calls. Processing can stop early
when the carry or borrow is resolved.

### 3.2 Scalar and paired-row multiplication

| Signature | Destination extent and relation |
| --- | --- |
| `add_mul_limbs_unchecked(dst, src, n, s) -> Limb` | `n` initialized limbs; $D' + cB^n = D_0 + Ss$. |
| `sub_mul_limbs_unchecked(dst, src, n, s) -> (Limb, Limb)` | `n` initialized limbs; returns `(c, b)` with $D' - (c+b)B^n = D_0 - Ss$. |
| `add_mul_2_limbs_unchecked(dst, src, n, s0, s1) -> (Limb, Limb)` | `n + 1` initialized limbs; $D' + c_0B^n + c_1B^{n+1} = D_0 + S(s_0 + Bs_1)$. |
| `mul_2_limbs_unchecked(dst, src, n, s0, s1)` | For `n > 0`, initializes all `n + 2` output limbs with $S(s_0 + Bs_1)$, independently of prior destination contents. |

For multiply-subtract, $c = \lfloor Ss/B^n \rfloor$ is the product carry and
$b \in \{0,1\}$ is the borrow from subtracting the low product limbs. They
are separate return values. For paired multiply-add, the caller still adds
`c0` to `dst[n]` and absorbs `c1` plus that addition's overflow into the next
limb. The returned carries are not already folded into the stored result.

### 3.3 Complete products

`mul_basecase_unchecked(dst, a, len_a, b, len_b)` writes the full product to
`len_a + len_b` limbs for `len_a >= 2` and `len_b > 0`. The arithmetic caller
handles smaller first operands. The complete backend owns row initialization,
accumulation, and fixed-width cases.

`sqr_basecase_unchecked(dst, src, n)` writes the full square to `2 * n` limbs.
The portable driver handles `n <= 8`, including specialized unrolled cases;
the selected BMI2/ADX square kernels handle larger inputs. Fixed-width facade
helpers also expose 2×2, 3×3, 4×4, and 8×8 products.

### 3.4 Independent sums and butterflies

Carries and borrows below are single bits.

| Signature | Relations and return values |
| --- | --- |
| `add_two_limbs_unchecked(dst_a, src_a, dst_b, src_b, n)` | $A' + c_aB^n = A_0 + S_a$; $D' + c_bB^n = D_0 + S_b$; returns `(ca, cb)`. |
| `add_sub_limbs_unchecked(sum, diff, n)` | $U' + cB^n = U_0 + V_0$; $V' - bB^n = U_0 - V_0$; returns `(c, b)`. |
| `add_reverse_sub_limbs_unchecked(sum, diff, n)` | $U' + cB^n = U_0 + V_0$; $V' - bB^n = V_0 - U_0$; returns `(c, b)`. |
| `add_sub_from_limbs_unchecked(sum, diff, src, n)` | $U' + cB^n = U_0 + S$; $V' - bB^n = U_0 - S$; returns `(c, b)`. |

The shared-source kernel reads the original sum and source, not the old
difference destination. An independent difference buffer is write-only.

### 3.5 Division and Montgomery reduction

`divrem_1_unchecked(limb, remainder_high, divisor) -> (Limb, Limb)` takes the
**low limb first**. With $v \ne 0$ and $h < v$, it returns `(q, r)` satisfying
$qv + r = hB + limb$, $q < B$, and $r < v$.

`monty_redc_step_unchecked(out, multiplicand, modulus, n, scalar, inverse)`
is the function in the `monty_redc_unchecked` module. For `n > 0`, the modulus
$N$ is odd and $inverse = -N[0]^{-1} \bmod B$. For incoming accumulator $T$
and multiplicand $Y$:

$$q = ((T[0] + scalar \cdot Y[0]) \cdot inverse) \bmod B,$$
$$T' + cB^n = (T + scalar \cdot Y + qN)/B.$$

Division by $B$ is exact. The function stores the low `n` result limbs and
returns $c \in \{0,1\}$. It performs one CIOS step, not a final canonical
reduction below $N$. Input and output spans each contain `n` limbs.

### 3.6 Shifts

Every shift takes $0 < s < W$. Results are truncated to the destination extent.

| Signature | Stored result and return value |
| --- | --- |
| `lshift_unchecked(limbs, n, s)` | $D' = D_0 2^s \bmod B^n$; returns `old[n-1] >> (W-s)` for `n > 0`. |
| `rshift_unchecked(limbs, n, s)` | $D' = \lfloor D_0/2^s \rfloor$; returns `(old[0] << (W-s)) mod B` for `n > 0`. |
| `lshift_into_unchecked(dst, src, n, s)` | Same left-shift result/carry, with a disjoint write-only destination. |
| `rshift_into_unchecked(dst, src, n, s)` | Same right-shift result/carry, with a disjoint write-only destination. |
| `lshift_overlapping_unchecked(limbs, n, offset, s)` | Writes the shifted original prefix to `limbs[offset..offset+n]`; returns the left-shift carry. |
| `sub_shifted_high_limbs_unchecked(dst, src, n, s, borrow)` | $D' - bB^n = D_0 - \lfloor S/2^{W-s} \rfloor - borrow$; input and output borrows are in `{0,1}`. |

Right-shift return bits are **left-aligned**, not an unshifted remainder.
Shifted-high subtraction uses `(src[i] >> (W-s)) | (src[i+1] << s)`, with an
implicit zero above the source. Overlapping left shift requires `offset + n`
initialized writable limbs in one allocation. The `*_into_small_unchecked`
facade methods use baseline small-input selection with the same contracts.

## 4. Backend matrix

The broad scalar set comprises 13 Rust target architectures: `x86_64`, `x86`,
`aarch64`, `arm`, `powerpc64`, `powerpc`, `s390x`, `riscv64`, `riscv32`,
`loongarch64`, `loongarch32`, `mips64`, and `mips`. Width and feature gates in
module declarations remain authoritative. Backends can use assembly,
intrinsics, or Rust composition.

ARM assembly providers require ARM instruction mode; Thumb targets use the
portable paths. The paired-row `umaal` providers additionally require ARMv6.
RISC-V multiply providers require the `m` extension; integer-only targets use
portable multiplication. These requirements belong in each operation's backend
availability predicate so fallback selection uses the same condition.

| Operation/module | Production implementations | Other selection / qualification |
| --- | --- | --- |
| `add_limbs_unchecked`, `add_limbs_3_unchecked` | Broad scalar set | Portable fallback elsewhere and under Miri. |
| `sub_limbs_unchecked`, `sub_limbs_3_unchecked` | Broad scalar set | Portable fallback elsewhere and under Miri. |
| `add_mul_limbs_unchecked` | Broad scalar set; x86-64 baseline/BMI2/ADX+BMI2; POWER64 baseline/POWER9 | ARM single-row backend uses `umull` and additions. |
| `sub_mul_limbs_unchecked` | Broad scalar set; x86-64 baseline/BMI2/ADX+BMI2 | POWER64 baseline; no dedicated POWER9 variant. |
| `add_mul_2_limbs_unchecked` | Broad scalar set; x86-64 baseline/BMI2; POWER64 baseline/POWER9 | ARM paired-row backend uses `umaal`. |
| `mul_2_limbs_unchecked` | x86-64 baseline/BMI2, x86, AArch64, ARM, POWER64, s390x, RISC-V64 | Portable fallback elsewhere. |
| `mul_basecase_unchecked` | Complete x86-64 baseline/BMI2/ADX+BMI2 compositions, including ADX fixed-width kernels | Direct composition over selected row kernels elsewhere. |
| `sqr_basecase_unchecked` | x86-64 BMI2 or ADX+BMI2 for `n > 8` | Portable driver otherwise, including small inputs. |
| `propagate_carry_unchecked`, `propagate_borrow_unchecked` | x86-64, AArch64, s390x | Portable fallback elsewhere. |
| `add_two_limbs_unchecked` | x86-64 ADX | Portable fallback elsewhere. |
| `add_sub_limbs_unchecked`, `add_reverse_sub_limbs_unchecked` | x86-64 ADX | Portable fallback elsewhere. |
| `add_sub_from_limbs_unchecked` | x86-64 ADX | Scalar fallback otherwise; AVX2 is test-only. |
| `divrem_1_unchecked` | x86-64 `divq`, x86 `divl`, s390x `dlgr` | Half-limb Rust algorithm on explicitly listed targets; double-limb Rust fallback otherwise and under Miri. |
| `monty_redc_unchecked` | x86-64 BMI2/ADX+BMI2, AArch64, POWER64, s390x, RISC-V64, LoongArch64 | Portable fallback on baseline/ADX-only x86-64 and other targets. |
| `sub_shifted_high_limbs_unchecked` | x86-64 BMI2, AArch64, POWER32/64, s390x | Portable fallback elsewhere. |
| `lshift_unchecked`, `rshift_unchecked` | x86-64 SSE2/AVX2; AArch64 | No dedicated BMI2 or AVX-512 in-place tier. |
| `lshift_into_unchecked`, `rshift_into_unchecked` | x86-64 SSE2/AVX2/AVX-512; AArch64 | Portable fallback elsewhere. |
| `lshift_overlapping_unchecked` | x86-64 SSE2/AVX2/AVX-512 | Portable fallback elsewhere. |

The [division module](../../src/int/logic/unsigned/math/arch/divrem_1_unchecked/mod.rs)
explicitly lists half-limb targets, including architectures beyond the broad
scalar set. Its Rust algorithm performs two limb-width divisions and bounded
quotient corrections. Hardware lowering depends on the compilation target.

## 5. Dispatch and instruction constraints

### 5.1 Independent arithmetic and SIMD selection

[x86_runtime.rs](../../src/int/logic/unsigned/math/arch/x86_runtime.rs) caches
two independent classifications with `OnceLock` in eligible `std` builds.
Operation modules map those classifications to function pointers.

| Arithmetic level | Meaning |
| --- | --- |
| `AdxBmi2` | ADX and BMI2; permits dual-carry multiplication kernels. |
| `Adx` | ADX without BMI2; permits ADX addition/butterflies, not `mulx` kernels. |
| `Bmi2` | BMI2 multiplication and shifted-high subtraction; no ADX selection. |
| `Baseline` | Baseline assembly or portable code, depending on the operation. |

SIMD levels are `Avx512`, `Avx2`, and `Sse2`. Runtime AVX-512 selection requires
**both AVX-512F and AVX2**: in-place shifts map the highest shared level to AVX2,
while out-of-place and overlapping shifts have dedicated AVX-512 providers.

Compile-time features can select direct backends. `no_std` uses compile-time
selection; Miri uses portable paths. Debug `MP_ANAFIS_TEST_BACKEND` overrides
request supported runtime tiers, falling back safely for unsupported requests.
They do not override a backend selected through compile-time target features.

### 5.2 Flag and division contracts

ADX `adcx` updates CF and `adox` updates OF. Intervening instructions must
preserve live flags; for instance, `dec` preserves CF while updating OF, so it
is valid only after live OF values are consumed or saved. Independent carry
chains remove flag dependencies between the chains; instruction scheduling and
throughput depend on the processor. See Intel's
[instruction-set reference, Volume 2A](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2a-manual.pdf).

A non-zero divisor with the high dividend limb below it ensures the double-limb
quotient fits into a single limb without overflow traps, enabling direct lowering
to hardware division instructions (such as x86 `divq`).

### 5.3 Experimental AVX2 shared-source butterfly

The AVX2 provider computes packed sums and differences and reconstructs carries
and borrows across SIMD lanes. Its test module loads it only on native x86-64
`std` builds. Production selects ADX or scalar backends. Tests verify carry
recurrence, arithmetic correctness, odd-limb tails, and exact
`difference == source` aliasing.

## 6. Verification

Correctness tests verify carry and borrow propagation, output disjointness,
permitted aliasing, empty-slice boundaries, and dispatch crossovers against
formal arithmetic invariants.

Each operation owns its tests in `tests.rs` or `tests/`. Shared `arch/tests/`
contains independent arithmetic recurrences, input patterns, limb-product
checks, and CPU-selection tests. Native tests call compiled backends only after
detecting their required CPU features. Miri uses portable backends with reduced
case budgets; assembly and CPU-detection tests are excluded by module gates.

Multi-architecture verification combines native compilation, matrix checks, and
emulation:

- `tools/check_all_archs.sh` validates compilation, code generation, and lints
  across the target and feature matrix, including 16-bit, big-endian, and
  restricted-extension profiles.
- QEMU user-mode emulation tests non-x86 kernels (ARM, AArch64, RISC-V, POWER,
  and s390x) against an independent arithmetic oracle, validating carry chains,
  aliasing, shifts, products, division, and Montgomery steps.
- Target profiles without ADX or BMI2 validate baseline assembly and portable
  Rust fallbacks.

```bash
cargo check --lib
cargo clippy --lib
cargo test --lib
python3 tools/structure_audit.py
python3 tools/import_audit.py
bash tools/check_all_archs.sh
cargo bench --bench public_api --features "std,rayon,_internal-tune"
```

Performance measurements require identical operands, preallocated buffers, CPU
pinning, and interleaved A/B/B/A sampling to isolate kernel execution from
allocation and cache transients.
