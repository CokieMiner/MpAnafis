# Internal arithmetic benchmarks

This target measures multiplication and squaring through the `_internal-tune`
interface. It separates production algorithm selection from explicit tier
selection. The [public API target](../public_api/README.md) measures public methods
and their allocation and precision contracts.

## Layout

```text
main.rs                    platform gates and Divan entry point
compare/
  mod.rs                   comparison registry
  production.rs            equal-width production multiplication
  unbalanced.rs            production multiplication across operand shapes
  flint.rs                 FLINT bindings and scoped worker budget
crossovers/
  mod.rs                   crossover registry
  multiply.rs              individually selected multiplication tiers
  square.rs                production, forced SSA, direct Fermat, and GMP squares
shared/
  mod.rs                   explicit shared exports
  operands.rs              deterministic limbs and checked output sizing
  gmp.rs                   reference products and GMP layout/count checks
  validation.rs            untimed result check and one-call warmup
  cases.rs                 operand shape and worker labels
  sizes.rs                 production comparison ladders and shape matrices
```

Registries contain module declarations and explicit exports. Case entry points
are ordinary functions, followed by their measurement helpers and ABI boundary
checks. Shared case builders receive the worker budget explicitly. Crossover
widths stay with their experiment; shared production ladders live in
`shared/sizes.rs`. FLINT bindings belong to `compare/`.

## What each group measures

| Group | Inputs | Measured operation |
| --- | --- | --- |
| `compare::production` | Equal limb counts | Mp production tower with reusable buffers, GMP `mpn_mul_n`, or FLINT product. |
| `compare::unbalanced` | Explicit longer/shorter limb counts, including balanced controls | Mp production tower with reusable buffers, GMP `mpn_mul`, or FLINT product. |
| `crossovers::multiply` | Equal limb counts chosen for each tier | Prepared schoolbook, Karatsuba, or Toom runner; SSA product with production planning. |
| `crossovers::square` | One shared width ladder | Production square with reusable buffers, prepared forced SSA or direct Fermat, or GMP `mpn_sqr`. |

Inputs are deterministic, with the high bit of the top limb set. Every row
checks one untimed output against an independent implementation before timing.
Mp results use a GMP reference; timed GMP calls use an Mp reference. FLINT results
use GMP. One timed iteration computes one product or square.

Output buffers are allocated before timing. The untimed validation and warmup
also provision reusable numeric scratch. Production multiplication and square
rows run dispatch, scratch sizing, and any SSA plan construction on every call.
Forced Karatsuba and Toom rows select their root tier before timing; recursive
children retain normal dispatch. Forced SSA square rows retain the transform
plan, whereas `crossovers::multiply::ssa` uses production transform planning on
each run. Plan metadata allocations remain included wherever planning is timed.
Allocations internal to GMP or FLINT remain included in their calls.

FLINT here is an independent internal engine comparison, alongside GMP. The
public API suite's FLINT fallback rule concerns public method equivalents.

## Sizes, workers, and sampling

All size arguments count **limbs**, not bits. The supported host uses 64-bit
limbs. Balanced comparison labels use `256-limbs/1-workers`; rectangular labels
use `400x100-limbs/1-workers`. Crossover arguments are numeric limb counts.

Mp cases use the ambient Rayon worker budget, or one worker without `rayon`.
GMP and serial FLINT rows always label one worker. Parallel FLINT rows use the
ambient budget and are registered only when it exceeds one. The scoped FLINT
guard restores the previous global budget after each case. Set
`RAYON_NUM_THREADS` explicitly for worker comparisons; listing cases resolves
the ambient pool as well.

Standard rows use Divan's default sampling unless overridden on the command line.
Rows ending in `_huge` use three samples of one iteration and remain distinct
from the standard ladder. The balanced huge ladder reaches 33,554,432 limbs;
the rectangular huge matrix reaches 4,194,304 limbs for its longer operand.

## Platforms and execution

Cargo requires `_internal-tune` for this target. Crossover cases require 64-bit
x86, where the Cargo development dependencies provide GMP. Production comparison
cases additionally require Linux and the system FLINT library. Unsupported
architectures compile an entry point that reports the target requirement and
exits without attempting a benchmark. `rayon` enables parallel execution.

```sh
cargo clippy --bench internal_improvement --features _internal-tune,rayon
cargo bench --bench internal_improvement --features _internal-tune,rayon --no-run
RAYON_NUM_THREADS=1 cargo bench --bench internal_improvement \
  --features _internal-tune,rayon -- --list
```

Execute a selected balanced check, including each engine's untimed validation:

```sh
RAYON_NUM_THREADS=1 cargo bench --bench internal_improvement \
  --features _internal-tune,rayon -- --test \
  '^internal_improvement::compare::production::(mp|gmp_serial|flint_serial)::256-limbs/1-workers$'
```

For measurements, invoke the built executable with `--bench` and exact filters,
following the CPU pinning and A/B/B/A interleaved execution protocols documented in the
[benchmark driver guide](../../tools/benchmark/README.md). Keep commands, worker settings,
and raw captures under `target/bench-results/`.

The shared reporter processes captured Divan output:

```sh
python3 tools/bench.py report target/bench-results/internal-capture.txt \
  --suite internal_improvement --output target/bench-results/internal-report
```

The reporter retains full paths, shapes, workers, and sampling ladders. Its
numeric size plots apply to crossover rows; shape/worker labels remain tables.
The generic execution planner and checked figure exporter target `public_api`.
