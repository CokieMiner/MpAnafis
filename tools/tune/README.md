# Hardware autotuner

`mp-tune` measures arithmetic kernels and selects performance thresholds for the
current hardware. Profile fields are defined in
[`build_support/`](../../build_support/); units and measured operations are listed
in [coverage.md](coverage.md).

## Execution

Serial tuning on one CPU:

```sh
taskset -c 2 cargo run --release --bin mp-tune --features _internal-tune
```

Parallel tuning on an explicit CPU set:

```sh
taskset -c 2-5 cargo run --release --bin mp-tune \
  --features "_internal-tune,rayon" -- --parallel-only=2-5
```

Serial runs require one CPU and `RAYON_NUM_THREADS` absent or equal to one.
Parallel runs require Rayon; on Linux, affinity must match the requested CPU set.
Use idle CPUs with stable frequency; tune each core class separately.

## Modes

Modes are mutually exclusive and follow Cargo's `--` separator.

| Mode | Operation |
| --- | --- |
| No mode | Complete serial tuning, validation, and installation. |
| `--tiers-only` | Arithmetic entries, division, modular arithmetic, GCD, and conversion. |
| `--compiled-only` | Reconstruction, transform, product, and division policies. |
| `--division-only`, `--modular-only` | Shared product policies and the selected arithmetic family. |
| `--toom-only`, `--gcd-only` | The selected arithmetic family. |
| `--formatting-only`, `--parsing-only` | The selected conversion direction. |
| `--parallel-only=<cpu-list>` | Parallel SSA tuning and production product validation. |
| `--validate-only` | Installation checks on `MP_TUNING_START`. |
| `--validate-parallel-only=<cpu-list>` | Parallel production checks on `MP_TUNING_START`. |
| `--check` | Worker builds, numerical results, and output protocols. |
| `--profile-for <arch>` | Install a built-in target profile without measurement. |

`MP_TUNING_START` selects the complete starting profile. Otherwise, tuning starts
from built-in defaults. Partial searches save the candidate and report; complete
and validation-only runs install a profile after successful validation.

```sh
MP_TUNING_START=/absolute/path/candidate.rs taskset -c 2 \
  cargo run --release --bin mp-tune --features _internal-tune -- --gcd-only
```

## Source organization

| Module | Responsibility |
| --- | --- |
| [`main.rs`](main.rs), [`arguments.rs`](arguments.rs) | Command selection and dispatch. |
| [`session/`](session/) | Session state, phase order, and installation decisions. |
| [`tiers/`](tiers/), [`crossovers.rs`](crossovers.rs) | Algorithm crossover searches. |
| [`compiled/`](compiled/) | Candidate grids, coordinate searches, and coupled updates. |
| [`measure/`](measure/) | Batch calibration and paired timing statistics. |
| [`harness/`](harness/) | Candidate builds and worker captures. |
| [`worker/`](worker/) | Operand fixtures, result verification, and timed execution. |
| [`platform/`](platform/) | Hardware metadata and CPU affinity. |
| [`store/`](store/) | Context identity, caches, reports, and profile publication. |
| [`validation/`](validation/) | Regression limits and production validation. |

Candidate grids: [`compiled/knobs.rs`](compiled/knobs.rs).
Regression limits: [`validation/criteria.rs`](validation/criteria.rs).
Profile publication: [`store/profile.rs`](store/profile.rs).

## Search order

Multiplication → squaring → transforms → products → division → modular powers →
GCD → formatting → parsing → validation. Dependent choices are revisited when
their inputs change.

Serial tuning searches 67 fields; three parallel fields use the explicit CPU set.
Mathematical and safety bounds remain fixed.

- Coordinate searches use at most three passes and two grid expansions per
  coordinate. An enabled boundary value can extend the grid by halving or doubling.
- Crossover searches screen by bisection, then confirm every width within four
  limbs of the candidate, its four immediate successors, and remaining ladder guards.

## Measurements and acceptance

Workers verify results, warm kernels, calibrate batches, and reuse buffers.
Both candidates use identical operands and scoring weights.

Cached scores screen candidates. Fresh A/B/B/A and B/A/A/B slots confirm changes;
pilot samples set the confirmation budget and are discarded. The
[confidence implementation](measure/confidence.rs) defines sample counts and
median-ratio bounds, which assume independent, stationary observations.

Direct product objectives time cyclic residues and prepared Montgomery products.
Complete consumer calls provide additional regression checks.

Installation requires production validation against defaults. Production
regression limits are 0.5% aggregate, 1.5% per family, and 3% per cell. Failed or
inconclusive required checks reject installation.

The [reserved catalog](worker/profile/holdout.rs) adds 78 cells across ten
families to final validation. These cells are excluded from search and cached
scoring. `--score-holdout` runs them directly; `MP_TUNING_CELLS` selects indices.

Results apply to the measured candidates and operand sizes.

## Profiles and artifacts

`build.rs` selects the first available complete profile:

1. `MP_TUNING_PROFILE`.
2. The ignored local `src/int/tuned_thresholds.rs`.
3. Built-in defaults from `build_support/`.

Profiles use the current schema's decimal `usize` constants. Incomplete,
duplicate, unknown, or invalid fields are rejected.

```sh
MP_TUNING_PROFILE=/absolute/path/profile.rs cargo build --release
```

`target/tune/` stores context metadata, cached scores, executables, raw captures,
decisions, and reports. Reports are saved before atomic profile installation.
Source or build-context changes stop acceptance. Local profiles remain untracked.
`--check` captures are stored in `target/tune/checks/`; parallel validation saves
a separate receipt for its CPU set and worker counts.
