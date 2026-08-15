# Assembly Analyzer (`tools/asm_analyzer`)

The Assembly Analyzer is a static analysis, microarchitectural simulation, empirical calibration, and optimization toolkit for inline assembly kernels (`asm!`) in `MpAnafis` (`mp_anafis`). It provides automated pipeline simulation, static hazard detection, topological instruction scheduling search, empirical cycle calibration, side-by-side kernel diffing, and Linux `perf` PMU integration.

---

## 1. Overview and Architecture

The toolkit extracts assembly emitted by `rustc`, parses instructions, builds dependency graphs, and evaluates execution characteristics across target CPU models using analytical models, external simulation backends, and empirical micro-benchmarking.

```
tools/asm_analyzer/
├── __init__.py           # Package facade
├── __main__.py           # CLI argument parsing and routing
├── analyzer.py           # Abstract Base Class for simulator backends
├── asm_util.py           # Register classification, AT&T parsing, subprocess runners
├── diff_test.py          # Differential correctness testing for candidate rewrites
├── extract.py            # Temporary crate driver to extract rustc assembly from asm!
├── models.py             # CPU specifications, matrix configurations, backend models
├── types.py              # Domain dataclasses (stats, reports, metrics, enums)
│
├── backends/             # Simulation & Benchmark Backends
│   ├── llvm_mca.py       # LLVM Machine Code Analyzer backend
│   ├── mca_driver.py     # Backend driver and factory utilities
│   ├── nanobench.py      # Active empirical micro-benchmark runner (pinned CPU cycles)
│   ├── osaca.py          # OSACA (RRZE-HPC) throughput model backend
│   └── uica.py           # uiCA (uops.info) cycle simulator backend
│
├── commands/             # CLI Subcommands
│   ├── analyze.py        # Single-file microarchitectural analysis
│   ├── calibrate.py      # Active calibration against empirical cycle timings
│   ├── check.py          # Backend and CPU capability probe
│   ├── diff.py           # Side-by-side kernel variant comparison
│   ├── pmu.py            # Linux perf hardware PMU counter integration
│   ├── search.py         # Topological instruction permutation search
│   ├── suggest.py        # Static anti-pattern and suggestion engine (Rules OPT001-OPT012)
│   └── sweep.py          # Repository-wide kernel sweep across target CPUs
│
├── consensus/            # Statistical Consensus & Calibration
│   ├── confidence.py     # Variance estimation and confidence interval scoring
│   ├── dataset.py        # Append-only JSONL empirical calibration dataset
│   ├── error_model.py    # Per-CPU/backend bias regression corrections
│   └── score.py          # Multi-backend consensus score calculation
│
├── features/             # Microarchitectural Static Analyzers
│   ├── aarch64.py        # AArch64 LDP/STP pairs, MUL/UMULH, and ADCS chains
│   ├── branch_prediction.py # BTB density and loop entry alignment
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
│   ├── ast.py            # Instruction specifications and register/flag effect models
│   ├── dag.py            # Directed acyclic graph builder with dependency edges & search heuristics
│   └── engine.py         # Topological permutation scheduler and simulator evaluator
│
└── tests/                # Unit & Regression Test Suite (44 Tests)
    ├── test_aarch64.py   # Tests for AArch64 feature detection
    ├── test_branch_prediction.py # Tests for BTB density and loop alignment
    ├── test_calibrate.py # Tests for active calibration CLI subcommand
    ├── test_consensus_calibration.py # Tests for empirical error modeling and bias correction
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

The analyzer evaluates assembly blocks across all major target ISAs (x86_64, AArch64, ARM, PowerPC, s390x, RISC-V, LoongArch, MIPS):

| Metric | Description |
|---|---|
| **Unroll Factor** | Estimates the loop unrolling factor based on limb memory access offsets and stride patterns. |
| **GPR Usage** | Counts distinct general-purpose registers. Warns when register pressure approaches architecture limits (e.g. >14 on x86_64, >28 on AArch64, >5 on x86-32). |
| **Memory (L / S / RMW)** | Classifies memory operations into Loads, Stores, and Read-Modify-Write (RMW) operations. Distinguishes streaming writes (`movq %r8, ({dst})`) from true in-memory RMW stalls (`addq %rax, ({dst})`). |
| **32-Bit Loop Invariants** | Verifies net stack delta = 0 on all exits, guarantees live carry/borrow flags are captured before stack counter arithmetic, and checks stride matching. |
| **STLF Hazards** | Evaluates Store-to-Load Forwarding: detects partial-overlap stalls, cross-iteration reload penalties, and cacheline boundary straddles. |
| **Multiplier Slack** | Measures instruction distance between multiplication (`mulx`, `mulq`, `mul`, `madd`, `umaal`) and the first instruction consuming its product. Distinguishes legacy serial fallbacks from pipelined multi-stream execution. |
| **ADCX / ADOX Pairing** | Identifies parallel dual-carry chains utilizing independent condition flags (`CF` for `ADCX`, `OF` for `ADOX`) to achieve 2 additions per cycle on ADX targets. |
| **Short Loop Latency** | Models finite loop execution costs for $N \in [1, 2, 3, 4, 8, 16]$ limbs, combining prologue, loop body cycles, epilogue, and terminal branch misprediction penalties (~16 cycles). |
| **Cache Hierarchy & Roofline** | Maps memory footprint across L1D ($<32\text{ KB}$), L2 ($<512\text{ KB}$), L3 ($<32\text{ MB}$), and DRAM tiers with arithmetic intensity bounds. |
| **Multi-ISA Port Pressure** | Models execution port dispatch pressure: Intel Port 0/1/5/6, AMD ALU0-3, ARM M1/M2/X-series integer pipelines, PowerPC execution units, s390x dual pipes, and RISC-V dispatchers. |

---

## 3. CLI Commands and Usage

The main CLI entrypoint is `tools/asm_analyzer.py` (or `python3 -m asm_analyzer`).

### 3.1. Repository Audit & Optimization Advice (`audit` or `suggest`)

Scans assembly code for microarchitectural anti-patterns and outputs remediation suggestions with severity levels (`CRITICAL`, `WARNING`, `INFO`):

```bash
# Audit a single kernel (accepts .rs source files directly)
python3 tools/asm_analyzer.py audit src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs

# Audit an assembly file
python3 tools/asm_analyzer.py suggest path/to/kernel.s
```

Rules evaluated:
- **`OPT001-RMW-HAZARD`**: In-memory read-modify-write arithmetic locking execution ports.
- **`OPT002-MUL-SLACK-STALL`** / **`OPT002-LEGACY-MUL-SLACK`**: Multiplier product consumed with zero slack (or baseline pre-BMI2 fallback).
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

### 3.3. Active Empirical Calibration (`calibrate`)

Calibrates simulator predictions against empirical micro-benchmarks (`nanobench`) using bias regression and variance modeling:

```bash
# Run calibration on local CPU and record to dataset
python3 tools/asm_analyzer.py calibrate --cpu znver3 --runs 10
```

### 3.4. Repository Sweep (`sweep`)

Scans architecture-specific kernels (`src/int/logic/unsigned/math/arch/`) and compiles a comparison matrix across simulated CPU targets:

```bash
# Markdown table output
python3 tools/asm_analyzer.py sweep --markdown

# Structured JSON output
python3 tools/asm_analyzer.py sweep --json

# Run on a specific kernel directory
python3 tools/asm_analyzer.py sweep --path src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/
```

### 3.5. Kernel Diffing (`diff`)

Computes a side-by-side comparison between two kernel variants:

```bash
python3 tools/asm_analyzer.py diff \
    src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64.rs \
    src/int/logic/unsigned/math/arch/add_mul_limbs_unchecked/x86_64_adx.rs
```

### 3.6. Topological DAG Permutation Search (`search`)

Constructs an instruction dependency DAG and searches for topological permutations that minimize cycle latency:

```bash
python3 tools/asm_analyzer.py search src/int/logic/unsigned/math/arch/mul_2_limbs_unchecked/x86_64.rs --max-candidates 50
```

### 3.7. Hardware PMU Profiling (`pmu`)

Records real hardware performance counters under Linux `perf`:

```bash
python3 tools/asm_analyzer.py pmu -- cargo bench --bench addition
```

### 3.8. Simulator Capability Check (`check`)

Probes the system for available simulation tools (`llvm-mca`, `osaca`, `uica`, `nanobench`) and validates supported CPU targets:

```bash
python3 tools/asm_analyzer.py check
```

---

## 4. Supported Targets and Simulator Backends

### CPU Targets & ISA Triples

- **x86_64**: AMD Zen (`znver2`, `znver3`, `znver4`, `znver5`), Intel Core (`skylake`, `icelake-server`, `alderlake`)
- **AArch64**: ARM Neoverse (`neoverse-n1`, `neoverse-v1`, `cortex-a78`), Apple Silicon
- **ARM 32-bit**: Cortex-A7, Cortex-A15 (`armv7-a`)
- **PowerPC**: POWER9, POWER10 (`powerpc64le`, `powerpc`)
- **IBM Z**: z15, z16 (`s390x`)
- **RISC-V**: RV64GC, RV32GC
- **LoongArch / MIPS**: LoongArch64, MIPS64 / MIPS32

### Simulator & Benchmark Backends

| Backend | Tool | Mechanism |
|---|---|---|
| `llvm-mca` | LLVM Machine Code Analyzer | Multi-target out-of-order pipeline simulation with dispatch port tracking. |
| `osaca` | Open Source Architecture Code Analyzer | Critical path and throughput bound analysis based on machine profiles. |
| `uica` | uiCA (uops.info) | Cycle-accurate pipeline simulation for Intel x86 microarchitectures. |
| `nanobench` | Empirical Cycle Runner | High-resolution empirical hardware cycle measurement on pinned CPU cores. |

---

## 5. Development and Testing

The unit test suite validates all feature extractors, predictors, calibration models, and suggestion rules:

```bash
PYTHONPATH=tools python3 -m unittest discover -s tools/asm_analyzer/tests -v
```
