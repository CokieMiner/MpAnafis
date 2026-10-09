# Assembly Analyzer (`tools/asm_analyzer`)

The Assembly Analyzer is a static analysis, microarchitectural simulation, empirical measurement, and optimization toolkit for inline assembly kernels (`asm!`) in `MpAnafis` (`mp_anafis`). It provides automated pipeline simulation, static hazard detection, topological instruction scheduling search, pinned hardware measurement, side-by-side kernel diffing, and Linux `perf` PMU integration.

---

## 1. Overview and Architecture

The toolkit extracts assembly emitted by `rustc`, parses instructions, builds dependency graphs, and evaluates execution characteristics across target CPU models using analytical models, external simulation backends, and opt-in empirical micro-benchmarking. Analytical cycle values are model guidance, not measured performance claims.

```
tools/asm_analyzer/
├── __init__.py           # Package facade
├── __main__.py           # CLI argument parsing and routing
├── analyzer.py           # Abstract Base Class for simulator backends
├── asm_util.py           # Register classification, AT&T parsing, subprocess runners
├── diff_test.py          # Small native-verifier dispatcher
├── emitted_asm.py        # Decode rustc #APP/#NO_APP regions and enclosing symbols
├── extract.py            # Narrow extraction facade
├── extraction_harness.py # Render and compile temporary Rust extraction units
├── extraction_parser.py  # Parse asm! templates, operands, options, strings, and comments
├── hardware.py           # Read-only host cache, affinity, governor, and kernel provenance
├── kernel_source.py      # Repository discovery and real emitted-variant extraction
├── models.py             # CPU specifications, matrix configurations, backend models
├── regions.py            # Basic-block CFG construction and natural-loop selection
├── semantics.py          # Limb width, register budget, and model support by ISA
├── simulation.py         # Central backend matrix, validation, and diagnostics
├── targets.py            # ISA classification, Rust targets, compatible CPU policy
├── types.py              # Domain dataclasses (stats, reports, metrics, enums)
│
├── backends/             # Simulation & Benchmark Backends
│   ├── llvm_mca.py       # LLVM Machine Code Analyzer backend
│   ├── mca_driver.py     # Backend driver and factory utilities
│   ├── registry.py       # Backend construction and CPU-support filtering
│   ├── nanobench.py      # Active empirical micro-benchmark runner (pinned CPU cycles)
│   ├── nanobench_flow.py # Local control-flow and bounded countdown validation
│   ├── perf_runner.py    # Explicit perf workload timing; includes harness overhead
│   ├── osaca.py          # OSACA (RRZE-HPC) throughput model backend
│   └── uica.py           # uiCA (uops.info) cycle simulator backend
│
├── commands/             # CLI Subcommands
│   ├── analyze.py        # Single-file microarchitectural analysis
│   ├── check.py          # Backend and CPU capability probe
│   ├── diff.py           # Side-by-side kernel variant comparison
│   ├── optimize.py       # Batch search, hardware confirmation, source application
│   ├── pmu.py            # Linux perf hardware PMU counter integration
│   ├── search.py         # Topological instruction permutation search
│   ├── suggest.py        # Static anti-pattern and suggestion engine (Rules OPT001-OPT012)
│   └── sweep.py          # Repository-wide kernel sweep across target CPUs
│
├── differential/         # Same-ISA compiled differential execution
│   ├── common.py         # Shared deterministic driver and compile/run harness
│   ├── arm.py            # AArch64 and Arm32 ABI wrappers
│   ├── loongarch.py      # LoongArch32/64 ABI wrappers
│   ├── mips.py           # MIPS32/64 wrappers with HI/LO capture
│   ├── power.py          # PowerPC32/64 wrappers with CR/XER capture
│   ├── riscv.py          # RV32/RV64 ABI wrappers
│   ├── s390x.py          # s390x wrapper with condition-code capture
│   └── x86.py            # i686 and x86-64 wrappers with flags capture
│
├── consensus/            # Multi-backend consensus scoring
│   └── score.py          # Multi-backend consensus score calculation
│
├── features/             # Microarchitectural Static Analyzers
│   ├── extraction.py     # Feature aggregation and applicability assessments
│   ├── aarch64.py        # AArch64 LDP/STP pairs, MUL/UMULH, and ADCS chains
│   ├── branch_prediction.py # BTB density and loop entry alignment
│   ├── instruction_width.py # Exact target encodings through llvm-mc
│   ├── memory.py         # Memory operands: Loads, Stores, Read-Modify-Write, Alignment
│   ├── memory_hierarchy.py  # Working set and cache tier (L1D/L2/L3/DRAM) Roofline mapping
│   ├── multiplier.py     # Multi-ISA multiplier latency slack and pipelining
│   ├── ports.py          # Execution port pressure (Intel P0-P7, AMD ALU0-3, ARM, PowerPC, s390x, RISC-V)
│   ├── registers.py      # GPR allocation count and condition flag tracking
│   ├── short_loop.py     # Finite loop iteration latency & terminal misprediction model
│   ├── stlf.py           # Store-to-Load Forwarding partial overlap & straddle detector
│   ├── suggestions.py    # Actionable optimization advice generator (Rules OPT001-OPT012)
│   ├── uop_cache.py      # Decode width and µOp cache (DSB/Op-Cache) sizing
│   ├── vectorization.py  # AVX2 and AVX-512 IFMA vectorization feasibility
│   └── x86_32_loop.py    # 32-bit x86 stack loop control, flag preservation, and frame balance
│
├── report/               # Visualization & Formatting
│   ├── json_export.py    # Structured JSON report serialization
│   ├── markdown.py       # GitHub-flavored Markdown table formatters
│   └── terminal.py       # Rich ANSI terminal formatters with Unicode boxes
│
├── search/               # Dependency DAG & Instruction Permutation Search
│   ├── adapters.py       # Per-ISA parser and dependency-spec dispatch
│   ├── ast.py            # Instruction specifications and register/flag effect models
│   ├── dag.py            # Directed acyclic graph builder with dependency edges & search heuristics
│   ├── engine.py         # Topological permutation scheduler and simulator evaluator
│   ├── hardware.py       # Corrected screening and independent holdout
│   ├── memory_dependencies.py # Alias ranges and loop-carried stride analysis
│   ├── native_ast.py     # Conservative non-x86 instruction semantics
│   ├── source_apply.py   # Exact fail-closed emitted-to-Rust schedule mapping
│   └── statistics.py     # Exact paired sign test and Holm correction
│
└── tests/                # Unit and regression test suite
    ├── test_aarch64.py   # Tests for AArch64 feature detection
    ├── test_branch_prediction.py # Tests for BTB density and loop alignment
    ├── test_dag_heuristic.py # Tests for DAG scheduling heuristics and pruning
    ├── test_ifma_advisor.py # Tests for AVX-512 IFMA polynomial multiplication advice
    ├── test_memory.py    # Tests for RMW detection and memory access stats
    ├── test_memory_hierarchy.py # Tests for cache tier footprint mapping
    ├── test_multiplier.py# Tests for multiplier latency and pipeline stalls
    ├── test_nanobench.py # Tests for active nanobench execution backend
    ├── test_ports_multi_isa.py # Tests for multi-ISA dispatch port modeling
    ├── test_registers.py # Tests for GPR counting and ADX flag tracking
    ├── test_search_safety.py # Tests for DAG permutation safety constraints
    ├── test_short_loop.py# Tests for short loop finite iteration latency model
    ├── test_stlf.py      # Tests for STLF hazard and straddle detection
    ├── test_suggestions.py # Tests for automated optimization recommendations
    ├── test_terminal_diff.py # Tests for terminal diff and report formatters
    ├── test_uop_cache.py # Tests for µOp cache capacity and unroll bounds
    ├── test_vectorization.py # Tests for AVX2 and AVX-512 IFMA candidates
    └── test_x86_32_loop.py   # Tests for 32-bit x86 stack loop invariants and frame balancing
```

---

## 2. Microarchitectural Metrics & Analyzers

The analyzer extracts assembly across all listed target ISAs. Each JSON report marks every static section with `applicable`, `confidence`, and `rationale`; an unsupported model is reported as unavailable rather than evaluated with x86 defaults.

| Metric | Description |
|---|---|
| **Unroll Factor** | Uses repeated pointer/index stride for loops, destination lanes for straight-line blocks, and encoded widths for named fixed kernels. |
| **GPR Usage** | Counts distinct target registers and compares them with the conservative architecture budget stored in `semantics.py`. |
| **Memory (L / S / RMW)** | Classifies memory operations into loads, stores, and fused read-modify-writes. RMW presence is a resource tradeoff, not a hazard by itself. |
| **Memory Dependencies** | Proves same-address-form byte ranges disjoint, preserves every unknown alias, and reports cross-iteration overlap from pointer strides. |
| **Instruction Width** | Uses `llvm-mc --show-encoding` to retain the exact byte width of each emitted instruction on every assemblable target. |
| **32-Bit Loop Invariants** | Structurally checks the modeled x86-32 stack delta, live carry/borrow use, and stride matching. It is not a proof over arbitrary control flow. |
| **STLF Hazards** | Screens validated x86 and AArch64 operand forms for nearby partial overlaps and cache-line straddles. Penalty magnitudes are heuristic and CPU-dependent. |
| **Multiplier Slack** | Measures instruction distance between multiplication (`mulx`, `mulq`, `mul`, `madd`, `umaal`) and the first instruction consuming its product. Distinguishes baseline scalar fallbacks from pipelined multi-stream execution. |
| **ADCX / ADOX Pairing** | Identifies independent carry chains using `CF` for `ADCX` and `OF` for `ADOX`. Throughput depends on the target model and instruction dependencies. |
| **Short Loop Latency** | Provides a generic finite-trip heuristic for selected CFG loops. Its branch penalty is not a target-specific prediction. |
| **Cache Hierarchy & Roofline** | Maps a caller-provided operand size to generic cache tiers. Without that input, the report explicitly covers only the emitted iteration footprint. |
| **CPU Execution Model** | Retains per-CPU LLVM instruction latency, reciprocal throughput, dispatch width, and named execution-resource pressure. Generic static heuristics remain separately labeled. |

---

## 3. CLI Commands and Usage

The main CLI entrypoint is `tools/asm_analyzer.py` (or `python3 -m asm_analyzer`).
The module form requires `PYTHONPATH=tools` when invoked from the repository root.

### 3.1. Repository Audit & Optimization Advice (`audit` or `suggest`)

Scans assembly code for microarchitectural anti-patterns and outputs remediation suggestions with severity levels (`CRITICAL`, `WARNING`, `INFO`):

```bash
# Audit a single kernel (accepts .rs source files directly)
python3 tools/asm_analyzer.py audit src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs

# Audit an assembly file
python3 tools/asm_analyzer.py suggest path/to/kernel.s
```

Rules evaluated:
- **`OPT001-RMW-TRADEOFF`**: Neutral comparison point for fused memory arithmetic versus a register split.
- **`OPT002-MUL-SLACK-STALL`** / **`OPT002-BASELINE-MUL-SLACK`**: Multiplier product consumed with zero slack (or baseline scalar fallback).
- **`OPT003-ALIGN-FALLTHROUGH`**: Straight fall-through execution into `.p2align` NOP padding.
- **`OPT004-HIGH-GPR-PRESSURE`**: Excessive GPR allocation approaching architecture limits.
- **`OPT005-UOP-CACHE-OVERFLOW`**: Inner loop exceeding CPU Decoded Stream Buffer (DSB) capacity.
- **`OPT006-BTB-DENSITY-HIGH`**: Code window exceeding 3 branches per 64 bytes.
- **`OPT007-AVX512-IFMA-OPPORTUNITY`**: Vectorization opportunities for wide matrix/polynomial operations.
- **`OPT008-STLF-FORWARDING-HAZARD`**: Store-to-load forwarding partial overlap reloads.
- **`OPT009-IFMA-REDUNDANT-RADIX`**: Redundant radix-$2^{52}$ IFMA acceleration feasibility for wide multi-limb inputs.
- **`OPT010-STACK-IMBALANCE`**: Stack frame pointer delta $\ne 0$ at function return / exit paths.
- **`OPT011-FLAG-CLOBBER-LOOP-CONTROL`**: Live condition flag clobbered by stack loop counter arithmetic without prior mask capture.
- **`OPT012-LOOP-STRIDE-MISMATCH`**: Pointer displacement advance does not match loop counter decrement step.

### 3.2. Single Kernel Analysis (`analyze`)

Analyzes a single AT&T assembly file (`.s`) or Rust source file containing an `asm!` block across target CPUs:

```bash
python3 tools/asm_analyzer.py analyze src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/aarch64.rs --cpu neoverse-n1
```

### 3.3. Repository Sweep (`sweep`)

Scans architecture-specific kernels (`src/int/logic/unsigned/math/arch/`) and compiles a comparison matrix across modeled CPU targets. Every direct `asm!` block is reported independently, while macro sources are compiled whole so every generated function is analyzed under its emitted symbol:

```bash
# Markdown table output
python3 tools/asm_analyzer.py sweep --markdown

# Structured JSON output
python3 tools/asm_analyzer.py sweep --json

# Run on a specific kernel directory
python3 tools/asm_analyzer.py sweep src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/
```

### 3.4. Kernel Diffing (`diff`)

Computes a side-by-side analytical comparison between two kernel variants. Reported relative changes are labeled as modeled deltas, not speedups:

```bash
python3 tools/asm_analyzer.py diff \
    src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64.rs \
    src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs
```

### 3.5. Topological DAG Permutation Search (`search`)

Constructs a target-specific instruction dependency DAG, schedules each basic-block
interior without moving labels or branches, differentially validates the selected
region when running on the same ISA, and ranks full backend/CPU cells
by normalized regret:

```bash
python3 tools/asm_analyzer.py search src/int/logic/unsigned/math/arch/mul_2_limbs_unchecked/x86_64.rs --candidates 50
```

LLVM-MCA reports both a steady-state `Block RThroughput` resource bound and
simulated `Total Cycles`. Normal analysis retains the former as `cycles`;
schedule search ranks by `simulated_cycles = Total Cycles / iterations` because
it responds to instruction order. Search JSON records the selected cost metric
for every CPU/backend cell in `provenance.static_cost_metrics`.

The candidate budget bounds alternatives in addition to the original schedule.
Exact enumeration has a deterministic state budget and falls back to heuristic
scheduling when exhausted. Automatic search preserves register identities;
the separate renaming helper requires explicit scratch-register and live-out
contracts. MIPS search keeps each branch delay slot fixed and analyzes the full
block so region selection cannot discard a delay-slot instruction. JSON records
whether validation used native differential execution or static analysis only.

With `--backend nanobench`, candidates are measured against the original in
A/B/B/A order. JSON retains every sample plus CPU affinity, cache topology,
frequency governor, SMT siblings, and kernel provenance. `nanoBench` requires a
single-CPU affinity and read/write access to that CPU's MSR device. Its fixed
counter reports core cycles on Intel and reference-clock `RDTSC` ticks on AMD;
the latter remains valid for pinned interleaved comparisons and is labeled in
the retained provenance rather than presented as literal core cycles. Hardware
screening uses 11 paired rounds per candidate and an exact one-sided sign test.
Holm-Bonferroni correction controls the family-wise error across all screened
candidates. The corrected winner then receives 11 fresh holdout rounds that
were not used for selection. The original remains selected unless both stages
retain at least nine non-tied valid pairs, show at least 2% median improvement,
and pass their significance gate at 0.05.

Measurement snippets must have local control flow. Supported backward branches
use an unmodified, positively bounded countdown register; counter resets inside
the loop, indirect exits, returns, and unresolved relocations are rejected.
Pointer-changing snippets use one execution per sample to avoid cumulative
pointer drift. These checks do not prove memory bounds for arbitrary assembly.

### 3.6. Batch Kernel Optimization (`optimize`)

Discovers every eligible architecture kernel under a file or directory,
performs target-compatible static screening, and optionally runs native
differential and hardware confirmation. Foreign ISAs remain static-only on the
current machine; the same command activates their compiled verifier when run on
matching hardware.

Automatic hardware confirmation requires nanoBench and an identified host CPU.
The `perf` backend measures complete workloads including launcher and loop
overhead, so it cannot confirm or automatically apply a schedule. Ambiguous CPU
brand strings are reported as unknown rather than mapped by product-number
prefix. Static screening and same-ISA differential verification remain available
when hardware confirmation is unavailable.

```bash
# Static batch screening
python3 -m asm_analyzer optimize src/int/logic/unsigned/math/arch --json

# Corrected hardware selection and fresh holdout on the native x86-64 host
sudo env HOME="$HOME" PATH="$PATH" PYTHONPATH="$PWD/tools" \
  taskset -c 0 python3 -B -m asm_analyzer optimize \
  src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs \
  --backend llvm-mca --cpu znver5 --hardware --json

# Atomically apply only an independently confirmed, exactly mapped schedule
sudo env HOME="$HOME" PATH="$PATH" PYTHONPATH="$PWD/tools" \
  taskset -c 0 python3 -B -m asm_analyzer optimize path/to/kernel.rs \
  --backend llvm-mca --cpu znver5 --hardware --apply-confirmed --json
```

`--apply-confirmed` requires `--hardware`, verifies that the source did not
change during the run, and accepts only single-line Rust `asm!` templates whose
emitted instructions and confirmed schedule have the same multiset and a unique
region mapping. Macro-expanded templates or blocks with interleaved standalone
comments are reported for manual application instead of being guessed.
Each batch record also retains `static_original`, `static_winner`, and
`static_tie_count`, so a deterministic tie cannot be mistaken for evidence that
the original schedule is better.

Source application also rejects escaped or multi-statement templates, multiple
strings on one physical line, empty templates, and `concat!` templates. It accepts
standalone normal and raw strings with an exact emitted-region mapping.

### 3.7. Hardware PMU Profiling (`pmu`)

Records real hardware performance counters under Linux `perf`:

```bash
python3 tools/asm_analyzer.py pmu -- cargo bench --bench public_api \
  --features std,rayon -- 'int::unsigned::arithmetic::operators::add'
```

Failed commands and runs without measured counters return a nonzero status.

### 3.8. Simulator Capability Check (`check`)

Probes the system for available simulation tools (`llvm-mca`, `osaca`, `uica`, `nanobench`) and validates supported CPU targets:

```bash
python3 tools/asm_analyzer.py check
```

---

## 4. Supported Targets and Simulator Backends

### CPU Targets & ISA Triples

- **x86_64**: AMD Zen (`znver1` through `znver5`), Intel Core (`skylake`, `icelake-server`, `alderlake`, `coffee-lake`, `ice-lake`)
- **x86 32-bit**: Skylake scheduling model in `i386` mode
- **AArch64**: ARM Neoverse N1/V1/V2, ThunderX2, A64FX, Cortex-A72, and Apple M1
- **ARM 32-bit**: Cortex-A9 in ARMv7 mode
- **PowerPC**: POWER9 and POWER10 in 64-bit mode; POWER9 in 32-bit mode
- **IBM Z**: z15, z16 (`s390x`)
- **RISC-V**: Rocket RV32/RV64, SiFive U74, and SiFive P550 representatives
- **MIPS**: generic MIPS32r2 and MIPS64r2 scheduling models
- **LoongArch**: exact LA32/LA64 extraction and encoding; LLVM 23 currently exposes CPU names but no usable instruction scheduling model

### Simulator & Benchmark Backends

| Backend | Tool | Mechanism |
|---|---|---|
| `llvm-mca` | LLVM Machine Code Analyzer | Target-specific static throughput/resource model. |
| `osaca` | Open Source Architecture Code Analyzer | Port-pressure and loop-carried dependency bounds from machine profiles. |
| `uica` | uiCA (uops.info) | Analytical Intel x86 pipeline model. |
| `nanobench` | Empirical Cycle Runner | High-resolution empirical hardware cycle measurement on pinned CPU cores. |

### Trust boundaries

- Rust templates are never sent directly to a backend. Extraction must produce real compiler-emitted assembly or the command fails.
- Loop selection uses a basic-block control-flow graph and resolved backward edges, including numeric local labels and multi-operand conditional branches.
- Static report sections distinguish structural facts from heuristic estimates and explicitly mark ISA-specific sections unavailable when they do not apply.
- Sweep, analyze, diff, and search retain backend failures instead of silently computing medians from missing requested cells.
- CPU matrices are filtered by the source ISA. Foreign-ISA extraction also requires the corresponding installed Rust target.
- Schedule search supports every listed ISA while keeping block boundaries, labels, and branches fixed. Unknown instructions remain barriers; memory order is relaxed only for proved-disjoint byte ranges; unparsed lines fail closed; and native rewrites are accepted only after compiled differential execution succeeds on matching hardware.
- Batch hardware selection uses corrected multi-candidate screening and an independent holdout. Static model winners are never described or applied as measured winners.
- Rustup-distributed foreign targets are used directly. MIPS targets, whose prebuilt `core` is not distributed by rustup, are compiled through nightly Cargo `-Z build-std=core` with a shared analyzer-only target cache.
- `llvm-mca`, OSACA, and uiCA results are uncalibrated analytical estimates. Only an authorized, pinned `nanobench` run can support a hardware performance claim.

---

## 5. Development and Testing

Package facades declare explicit exports. Feature aggregation and backend
construction reside in their dedicated implementation files. Tests are isolated
by parser, feature, backend, search, and reporting responsibility.

The regression suite exercises feature extraction, predictions, execution
boundaries, and suggestion rules:

```bash
PYTHONPATH=tools python3 -m unittest discover -s tools/asm_analyzer/tests -v
```
