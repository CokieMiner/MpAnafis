# Benchmark driver and reports

Run `python3 tools/bench.py --help` from the repository. The driver discovers the
actual Cargo executable and its registered Divan cases. Selection uses full-path
globs; an unmatched selection is an error.
Execution and plans target `public_api`. The shared report and plot generator
also accepts `internal_improvement` captures.

```sh
# Check Rust declarations without compiling.
python3 tools/bench.py check --source-only

# Build, discover, and validate all registered function/scenario groups.
python3 tools/bench.py check --features std,rayon,_internal-tune --require-comparison

# Generate a reviewable plan and an executable script for any selection.
python3 tools/bench.py plan --case 'int::unsigned::arithmetic::operators::add' \
  --arg 256 --arg 1024 --rounds 3 \
  --output target/bench-results/add-plan.json --script target/bench-results/run-add.sh \
  --results target/bench-results/add-results

# Execute the plan. A fresh, empty output directory is required.
python3 tools/bench.py run --plan target/bench-results/add-plan.json \
  --output target/bench-results/add-results

# Or select and run directly; --smoke runs each case once without timing claims.
python3 tools/bench.py run --case 'int::signed::arithmetic::helpers::mul_add' \
  --arg 256 --smoke --output target/bench-results/mul-add-smoke

# Rebuild reports and optional plots from recorded measurements.
python3 tools/bench.py report target/bench-results/add-results/measurements.json \
  --output target/bench-results/add-report --plot

# Export only size-sweep figures and compact context for Git.
python3 tools/bench.py publish target/bench-results/add-results \
  --name add-review-2026-09-20 \
  --description 'State the measured question, conditions, and interpretation.'

# Report internal tiers or engine comparisons with the same plotter.
python3 tools/bench.py report target/bench-results/internal-capture.txt \
  --suite internal_improvement --output target/bench-results/internal-report --plot
```

No per-function Python script is needed. `--case` and `--arg` are repeatable.
`--binary PATH` explicitly reuses an existing executable; its features are the
caller's responsibility. Otherwise Cargo builds `public_api` with
`--features std,rayon`. Add `_internal-tune` to link FLINT comparisons
on supported Linux hosts. Rug/GMP comparisons require 64-bit x86.

## Execution and data integrity

Each function/scenario is isolated in a subprocess. Comparison rounds use
Mp/comparator/comparator/Mp ordering separately for every registered reference.
Reports retain each comparator and identify the fastest measured reference for
each argument and configuration. Defaults select one allowed CPU and one
Rayon worker; `--threads N` selects N available CPUs, or `--cpus ID,ID,...`
specifies affinity explicitly. Worker count is independent of Divan contention
threads. Linux execution uses `taskset`; explicit affinity is checked against
the process's permitted CPUs. Platforms without CPU-affinity support run unpinned.

Arguments, sampling, worker count, CPU affinity, and per-run timeout are captured
in the plan. Inherited `DIVAN_*` and `NEXTEST` overrides are removed for listing
and execution. Timeouts and interruptions
terminate the subprocess group on POSIX. Raw output and partial measurements are
retained when a run fails.

A run may declare its own `arguments` array to override the plan's default
selection. Each filter must match that exact array, and execution and export
verify the corresponding measured widths. This permits a plan containing
different registered ladders, such as bit widths and factorial inputs, without
applying one function's sizes to another.

Results contain:

- `run.json`: exact commands, plan, compiler version, executable hash, Git
  revision and dirty status, platform, timestamps, and completion status.
- Numbered stdout/stderr logs for every subprocess.
- `measurements.json`: all raw Divan summaries, preserving full function,
  scenario, argument, engine, run, and configuration identities.
- `summary.json`, `summary.csv`, and `report.md`: median of run medians,
  observed ranges, and comparator/Mp ratios.
- With `--plot`, one time-versus-size PNG per function/scenario and configuration.
  Paths use `plots/<suite>/<full-case-path>/<configuration>.png`; `report.md`
  links each plot. A curve requires at least two distinct numeric arguments for
  the same engine. Single-size comparisons and shaped arguments stay in tables.
  Curves use logarithmic axes when values are positive and show observed
  run-median ranges. Missing sizes are not synthesized or extrapolated.
  Plotting requires matplotlib; all other commands use Python's standard library.

Configurations include the binary hash, host, CPU model, platform, compiler, sampling, affinity,
Rayon workers, declared features, and relevant environment overrides. Different
configurations never merge into one comparison. Imports of raw Divan text have
unspecified configuration; only combine logs known to share the same environment.
The driver rejects duplicate measurements, invalid timing ranges, non-finite
values, malformed trees, and missing requested arguments. Listing and smoke-test
output cannot be mistaken for measured results.

Times are per **benchmark iteration**. Some cases process a ten-operand batch;
others process one operand. Reports do not silently divide by an assumed batch
size or infer throughput from a width. Ranges are observed run ranges, not
confidence intervals. CPU pinning and A/B/B/A ordering reduce variability but
do not establish statistical significance.

Internal reports retain engine/tier names, normal versus huge ladders, operand
shapes, and worker labels such as `512x256-limbs/8-workers`. They do not combine
different worker budgets or replace parallel measurements with serial data.
Their measurement table and plots do not infer cross-policy speedup ratios.
The execution planner remains specific to the symmetric public API suite.

## Documentation records

Repository working outputs must reside under ignored `target/bench-results/`.
An explicit working directory outside the repository is also accepted. The
driver resolves symlinks before checking plans, scripts, runs, and report paths.
`publish` exports a completed public run into
`docs/int/benchmarks/public_api/<name>/`. A different repository publication root
must remain under `docs/int/benchmarks/`; an explicit external root is accepted.

The exporter checks that every saved measurement agrees with the corresponding
raw stdout, and that the planned functions, engines, and requested arguments are
present. It requires CPU pinning, complete A/B/B/A rounds, matching worker and
sampling settings, chronological timestamps, and exact commands applying the
planned binary, affinity, filters, and sample counts. Unpinned or incomplete
captures remain available for local reports.
It exports size-sweep PNGs, a scope README, and a compact manifest with
host/compiler/plan metadata and source/figure checksums. Raw captures and numeric
reports remain in the working directory. They are deliberately excluded from
Git; preserve or share the complete run separately when reproducibility matters.
The manifest fingerprints evidence but does not embed or recover it.
Export requires matplotlib. It rejects single-size runs, smoke runs, incomplete
evidence, mismatched measurements, and existing record names. A failed export
leaves no partial record.

Keep records immutable. Use a new name for a new experiment and a separate
working directory when regenerating a report. A dirty worktree is reported
explicitly; its commit identifier alone cannot reproduce uncommitted changes.

The [records index](../../docs/int/benchmarks/README.md) describes what belongs in
Git. `.gitignore` admits only README files, measurement manifests, and figures
under each record's `plots/` tree. Validation runs and incomplete legacy captures
remain local under `target/bench-results/`.
There are no per-operation plotting entry points: all current report/plot work
uses `tools/bench.py report --plot` or `publish`.

## Choosing a size sweep

Omitting `--arg` runs the selected function's complete registered ladder; repeated
`--arg` values select exact registered points. The tool does not invent missing
widths. Choose one function/scenario per comparison and state its size units:
most arguments are bits, whereas factorial uses an integer input value and
internal kernels may use limbs.

The current additive ladder spans 256–16,777,216 bits, multiplication spans
256–16,777,216 bits, division spans 8–16,777,216 bits, and GCD spans
64–33,554,432 bits. Extended GCD, inversion, and unsigned Jacobi span
64–4,194,304 bits. These are registered widths, not a record of completed
measurements. Other operations have their own limits in
`benches/public_api/int/ladders.rs`. The shared hexadecimal
operand generator requires positive multiples of four bits. Primality and
modular exponentiation use separate ladders suited to their domains and cost.

Nested scenario benchmarks (such as algebraic distributions registered under an
operation) form distinct function/scenario groups. You can target an entire
group, individual operations, or specific scenario paths using Divan path
filters:

```sh
# Plan a sweep over an operation and all its nested scenarios:
python3 tools/bench.py plan --features std,rayon,_internal-tune \
  --case 'int::unsigned::theory::common_divisors::gcd*' \
  --arg 1024 --arg 4096 --arg 16384 --arg 65536 \
  --samples 15 --sample-size 1 --rounds 2 --cpus 2 \
  --output target/bench-results/gcd-sweep-plan.json

# Run the planned execution matrix:
python3 tools/bench.py run --plan target/bench-results/gcd-sweep-plan.json \
  --output target/bench-results/gcd-sweep
```

## Adding cases

The Rust organization and shared templates are documented in
[benches/public_api/README.md](../../benches/public_api/README.md).
`check --source-only` verifies handwritten engine names, sampling, argument
ladders, counters, and reset strategy. Full `check` also verifies compiled
registration, numeric categories, and comparison selection.

```sh
python3 -m unittest discover -s tools/benchmark/tests
python3 -m unittest discover -s tools/audit/tests
```

The package separates selection (`catalog.py`), plan validation (`plan.py`),
execution (`runner.py`), raw-evidence validation (`evidence.py`), reports
(`report.py`), and publication (`publish.py`). `paths.py` enforces artifact
placement; `cli.py` routes these operations. Tests follow the same responsibilities.
