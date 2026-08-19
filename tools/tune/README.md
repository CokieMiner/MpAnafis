# Hardware autotuner

`mp-tune` generates one complete machine-local integer tuning profile. The
tuner and build script consume the schema and architecture defaults from
`build_support/tuning.rs`.

Run it on an idle pinned core:

```sh
taskset -c 2 cargo run --release --bin mp-tune \
  --features _internal-tune
```

The tuner verifies that its launcher restricted it to one logical CPU and
prints a warning otherwise. On Linux it records the logical CPU, topology core
ID, and maximum frequency/capacity metadata exposed by sysfs. Results from
different core classes use separate report and score-cache directories.
`taskset` or the platform equivalent selects the measurement CPU.

Tune one core class at a time. A generated profile describes the core on which
the library is expected to do its heavy work; averaging fast and compact cores
would produce thresholds optimal for neither. On a hybrid Linux machine,
inspect the classes first and run separately if both profiles are useful:

```sh
lscpu -e=CPU,CORE,MAXMHZ,MINMHZ
taskset -c <cpu> cargo run --release --bin mp-tune \
  --features _internal-tune
```

## Phases

A complete run has six phases, ordered by algorithm family:

1. **Multiplication Toom tier.** Rebuild and measure the Toom-8.5 paired
   reconstruction boundary, then tune adjacent multiplication tiers from
   Schoolbook through Toom-Cook 8.5. Two implementations run on identical
   operands, verify identical output, and only a win that survives a later
   guard cell is accepted.
2. **Squaring Toom tier.** Tune the independent square ladder from Schoolbook
   through Toom-Cook 8.5. It uses the Toom-8.5 reconstruction choice from phase
   1 and measures square crossover values independently. Only the
   Toom-6 and Toom-8.5 searches extend to the 32,768-limb transform horizon;
   early square tiers retain the compact ladder.
3. **Division.** Rebuild and tune Burnikel-Ziegler block geometry and the Newton
   reciprocal basecase in their direct forced domains. Then rebuild every
   dispatch-threshold candidate and score the production divider, because the
   Burnikel threshold controls both outer entry and the recursive Algorithm-D
   handoff. Burnikel and Newton thresholds are coordinate-tuned with their
   ordering invariant enforced. Quotients and remainders must agree before
   timing.
4. **SSA and transforms.** Rebuild and tune SSA kernel, layout, and geometry
   constants. SSA candidates first use the complete size ladder with reduced
   repeats; only the three best and noise-adjacent candidates receive precise
   scores. Per-ring pins time only cells with that exact top-level ring. Finally,
   tune multiplication and squaring Toom-8.5-to-SSA crossovers separately.
5. **Radix formatting.** Tune the schoolbook-to-recursive string formatting crossovers. Since the recursive algorithm relies on optimized basecase arithmetic, this tuning is deferred until all arithmetic and transform crossovers are stable. Radices are grouped because schoolbook extraction and recursive leaf costs vary substantially. Decimal extracts limb-sized chunks with `10^19`, while small non-power-of-two radices require many more digit divisions.
6. **End-to-end validation.** The production dispatcher — not forced tiers —
   is scored with the tuned profile and with the architecture defaults on a
   ladder crossing multiplication, squaring, transform, and division
   crossovers. The tuned profile must beat the defaults by the host's noise
   margin; otherwise the run is rejected, nothing is installed, and the
   rejected profile is preserved for inspection.

### Measurement discipline

- Crossover comparisons alternate A/B/B/A and B/A/A/B paired slots, so slow
  frequency drift and a persistent first-position advantage cancel instead of
  biasing the winner.
- A calibration step times a stable mid-sized product repeatedly, estimates
  the host's coefficient of variation, and derives the acceptance margin from
  it (three sigma, floor 2%). A host whose timing spreads cannot accept
  differences it cannot distinguish.
- Every measurement is persisted under `target/tune/<cpu-and-core>/`: a JSON
  report with each knob's chosen value and outcome, and a score cache keyed by
  the worker mode, rendered profile, source tree, toolchain, compiler flags,
  timing-calibration bucket, and core identity. Repeated runs reuse compatible
  scores without comparing stale raw timings from another build or power state.

## Constant policy

Presence in `TuningProfile` means that generated builds must reproduce the
value; it does not automatically make the field a sound autotuning dimension.
The default tuner handles each constant according to the code path it controls:

| Constant | Usage | Default tuner policy |
| --- | --- | --- |
| **Formatting** | | |
| `RADIX_DECIMAL_RECURSIVE_THRESHOLD` | Schoolbook to recursive format for radix 10 | Tune directly on sustained crossovers. |
| `RADIX_SMALL_RECURSIVE_THRESHOLD` | Schoolbook to recursive format for radices 3..=9 | Tune directly on sustained crossovers of endpoints. |
| `RADIX_LARGE_RECURSIVE_THRESHOLD` | Schoolbook to recursive format for radices 11..=36 | Tune directly on sustained crossovers of endpoints. |
| **Multiplication Toom tier** | | |
| `KARATSUBA_THRESHOLD` | Schoolbook to Karatsuba dispatch | Tune directly on adjacent forced tiers. |
| `TOOM_COOK_THRESHOLD` | Karatsuba to Toom-3 dispatch | Tune directly on adjacent forced tiers. |
| `TOOM_COOK_4_THRESHOLD` | Toom-3 to Toom-4 dispatch | Tune directly on adjacent forced tiers. |
| `TOOM_COOK_6_THRESHOLD` | Toom-4 to Toom-6 dispatch | Tune directly on adjacent forced tiers. |
| `TOOM_COOK_85_THRESHOLD` | Toom-6 to Toom-8.5 dispatch | Tune directly on adjacent forced tiers. |
| `TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS` | Selects two real Toom-8.5 reconstruction kernels | Tune by rebuild on forced Toom-8.5 multiplication and square cells. |
| `TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS` | Selects guard expansion or a full two-guard-limb point product | Tune by rebuild on forced Toom-8.5 multiplication cells. |
| **Square Toom tier** | | |
| `SQR_KARATSUBA_THRESHOLD` | Schoolbook to Karatsuba square dispatch | Tune directly and independently from multiplication. |
| `SQR_TOOM_COOK_THRESHOLD` | Karatsuba to Toom-3 square dispatch | Tune directly. |
| `SQR_TOOM_COOK_4_THRESHOLD` | Toom-3 to Toom-4 square dispatch | Tune directly. |
| `SQR_TOOM_COOK_6_THRESHOLD` | Toom-4 to Toom-6 square dispatch | Tune directly. |
| `SQR_TOOM_COOK_85_THRESHOLD` | Toom-6 to Toom-8.5 square dispatch | Tune directly. |
| **Division** | | |
| `BURNIKEL_ZIEGLER_THRESHOLD` | Algorithm D to Burnikel-Ziegler dispatch and Burnikel recursive basecase | Tune by rebuild on production division after recursion geometry settles, capturing both directions of the cutoff. |
| `NEWTON_RAPHSON_THRESHOLD` | Burnikel-Ziegler to Newton-Raphson dispatch | Tune by rebuild on production division with the selected Burnikel cutoff fixed. |
| `BURNIKEL_ZIEGLER_BLOCK_LIMBS` | Burnikel-Ziegler recursion block width | Tune by rebuild on forced Burnikel-Ziegler division cells, never SSA cells. |
| `NEWTON_RAPHSON_BASECASE_LIMBS` | Newton reciprocal basecase cutoff | Tune by rebuild on forced Newton-Raphson division cells, never SSA cells. |
| **SSA and transforms** | | |
| `NTT_THRESHOLD` | Enables unfinished multi-prime NTT dispatch | Keep registered and force to zero; do not tune until NTT is complete. |
| `SSA_THRESHOLD` | Conventional multiplication tower to SSA | Tune directly after compiled SSA constants settle. |
| `SQR_SSA_THRESHOLD` | Conventional square tower to SSA | Tune separately from multiplication after compiled SSA constants settle. |
| `TRANSFORM_MIN_SMALLER_LIMBS` | Rejects transforms whose shorter operand is too small | Retain the architecture default. It needs a dedicated multi-width unbalanced-shape grid; balanced crossover timing cannot identify it. |
| `TRANSFORM_MAX_OPERAND_RATIO` | Rejects transforms at excessive operand ratios | Retain the architecture default for the same reason; a ratio grid must compare forced SSA with production blocking. |
| `SSA_BASE_MODULUS_BITS` | Chooses tower versus nested-transform pointwise products | Tune by rebuild on the complete forced-SSA size ladder. |
| `SSA_BNM1_BASECASE_LIMBS` | Chooses direct versus recursive Mersenne-ring multiplication | Tune by rebuild on forced SSA cells. |
| `SSA_NEGACYCLIC_FACTOR3_THRESHOLD` | Enables factor-3 negacyclic decomposition | Tune by rebuild on forced SSA cells, including a practical disable candidate. |
| `SSA_NEGACYCLIC_FACTOR5_THRESHOLD` | Enables factor-5 negacyclic decomposition | Tune by rebuild on forced SSA cells, including a practical disable candidate. |
| `SSA_COEFFICIENT_VISIT_OVERHEAD` | Planner model coefficient | Retain the architecture value. End-to-end coordinate timing cannot isolate it from the other planner coefficients. |
| `SSA_BASECASE_COST_WEIGHT_16THS` | Planner interpolation coefficient | Retain until fitted against isolated lower-tower measurements. |
| `SSA_NESTED_COST_PENALTY_16THS` | Planner correction for nested memory/cache cost | Retain until an isolated model-fit scores prediction error across rings; do not let it compensate for base modulus. |
| `SSA_DIRECT_SHIFT_MAX_LIMBS` | Direct versus decomposed Fermat shift kernel | Tune by rebuild on forced SSA cells. |

## Profile resolution

`build.rs` uses the first complete source available:

1. the path in `MP_TUNING_PROFILE`;
2. the ignored local `src/int/tuned_thresholds.rs`;
3. the conservative architecture profile in `build_support/tuning.rs`.

Partial profiles are rejected. This makes adding a new hardware-sensitive
constant fail loudly until the schema, defaults, rendering, and tuner are
updated together.

`MP_TUNING_PROFILE` is primarily an internal candidate mechanism, but it is
also useful for reproducible A/B builds:

```sh
MP_TUNING_PROFILE=/absolute/path/profile.rs cargo build --release
```

Do not commit a generated local profile as a portable default. Promote a value
to an architecture profile only after repeatable measurements on multiple
machines in that architecture family.

## Tuning in stages

The phases can be rerun separately:

```sh
taskset -c 2 cargo run --release --bin mp-tune \
  --features _internal-tune -- --tiers-only

taskset -c 2 cargo run --release --bin mp-tune \
  --features _internal-tune -- --compiled-only

taskset -c 2 cargo run --release --bin mp-tune \
  --features _internal-tune -- --toom-only

taskset -c 2 cargo run --release --bin mp-tune \
  --features _internal-tune -- --division-only
```

`--tiers-only` tunes the conventional towers, transform crossovers, formatting, and
division with the architecture-profile compiled constants. `--compiled-only` tunes the
compile-time constants alone. `--toom-only` is the short rebuild pass for the
compiled Toom-8.5 choices. `--division-only` tunes only division recursion
geometry and the coupled production dispatch thresholds. All four modes skip
the end-to-end validation gate, preserve their result as a rejected candidate
for inspection, and do not replace the local profile. Unchanged families retain
their architecture-profile values. Only a complete run can pass the
production-dispatch validation gate and install a local override.

The compiled-only phase builds and measures optimized binaries for each
compile-time kernel or geometry candidate. Planner model coefficients retain
their architecture values because this phase has no isolated fitting objective
for them.
