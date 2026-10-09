# Public API benchmark structure

Cases are organized by domain, category, function, engine, and numeric argument:

```text
int/{signed,unsigned}/
  arithmetic/{operators,helpers,policies}
  bitwise/{operators,inspection,shifts,manipulation,single_bit}
  comparison
  conversion/{strings,bytes,primitives}
  division/{quotient,rounding,predicates,shapes}
  modular
  theory/{common_divisors,roots,primality,special}
  sign  (signed only)
```

A category contains the operations applicable to that domain. Each measured
function or operand scenario has its own Divan module, containing `mp` and
`rug` (GMP) entries where equivalent operations exist. Jacobi also registers
FLINT, and Euler's totient uses FLINT because Rug/GMP does not provide it.
FLINT is gated by `_internal-tune`, Linux, x86-64, and 64-bit pointers.
Engine registration is target-dependent; the Mp cases remain available without
Rug. This is a benchmark inventory, not a claim that every public API signature
has performance coverage.

## Shared execution shapes

`int/support/templates.rs` defines three symmetric templates:

- `paired_bench!`: immutable prepared operands, untimed operand/result
  equivalence checks, then the same black-boxed batch loop for each engine.
- `paired_assign!`: a prepared reusable destination on both sides; assignments
  overwrite it for each operand pair. Allocation preparation is outside timing.
- `paired_mutate!`: a fresh clone outside each timed invocation, followed by a
  mutation. This preserves the intended input state for operations such as
  `abs_assign`.

Case declarations supply the function name, argument ladder, operand setup,
and operation for each library. Every engine inherits identical sample settings.
`paired_bench!` accepts an additional FLINT setup and operation; each comparator
gets its own A/B/B/A round in the driver.
Category-specific policy macros provide bounded success/failure cases using the
same paired shape. The FLINT totient cases use explicit declarations with matching
setup and timing, checked by the source audit.

```rust,ignore
paired_bench!(add, ADDITIVE,
    mp: mp_uint_pairs => |(a, b): &(MpUint, MpUint)| a + b,
    rug: rug_uint_pairs => |(a, b): &(Integer, Integer)| Integer::from(a + b),
);
```

The setup returns a batch, typically ten generated pairs or one prepared operand.
One timed iteration processes the entire batch. Deterministic seed and width
selection produce numerically identical inputs. Verification, conversion for
verification, and input generation are outside timing. Black boxes protect the
operands and outputs. Separate cases measure shift distances, arithmetic
policies, sign predicates, and primitive conversions; a case does not aggregate
different API functions.

`mod.rs` files contain module declarations and explicit reexports. Category files
contain case declarations followed by their local setup and reference helpers.
Shared code in `int/support/` separates these responsibilities:

- `templates.rs` declares paired cases; `measurement.rs` implements timed loops
  and receiver reset.
- `verification.rs` checks numeric contracts; `outcome.rs` encodes values for
  comparison. `comparison.rs` handles hashes and borrowed selections.
- `operands.rs`, `division.rs`, `gcd.rs`, and `shapes.rs` construct deterministic
  inputs; `sampling.rs` defines shared sampling settings.
- `rug_ops.rs` implements composed Rug references; `flint.rs` owns the FLINT
  integer and its C bindings.

## Operand shapes and scenario distributions

Baseline cases use seeded, exact-width magnitudes. Named scenarios vary operand
geometry and algebraic structure:

- **Geometric shapes**: Asymmetric operand geometries and residue classes
  (such as unbalanced operand ratios, exact multiples, or maximal remainders).
  Separate cases isolate these inputs from equal-width random pairs.
- **Algebraic scenarios**: Nested benchmark submodules registered under an
  operation to evaluate specialized mathematical structures (such as
  near-equal pairs, shared factors, extreme trailing zeros, or structured
  factorizations). Their generators establish the stated structure before timing.

Both shapes and scenarios adhere to the same execution discipline as standard
cases: symmetric pairing across engines, deterministic input streams, identical
sample settings, and untimed verification. Specific input distributions,
geometric ratios, and scenario generators are documented in their respective
support modules under `int/support/`.

## Comparison contracts

References implement the public result contract. A composed reference includes
the required computation in its timed expression; it is not labeled as a
dedicated GMP primitive:

- Fused multiply-add computes `a * b + c` on both sides.
- Widening and carrying operations split products at the same bounded width;
  signed halves use signed two's-complement interpretation.
- Signed byte conversion produces the same minimal two's-complement encoding.
- Float conversion composes ties-to-even integer rounding with GMP's truncating
  export and returns `None` on overflow.
- Signed modular exponentiation follows the public magnitude-based contract.
- Unsigned extended-GCD coefficients use the public modular-residue convention.
  Untimed validation compares GCDs exactly and verifies both coefficient bounds
  and congruences using GMP. Nonunique valid representatives need not match.
- Prime search is exclusive on both sides. Primality fixtures compare
  classifications, while Mp and GMP retain different witness policies.
- Bit reversal uses Rug byte export/import and primitive byte bit reversal.
- Montgomery multiplication includes Mp domain setup and compares
  `a*b*R^-1 mod m` with a composed Rug reference that includes radix inversion.
- Division and modular references preserve optional result wrappers.
- Minimum, maximum, and clamp return borrowed operands on both sides. Untimed
  checks encode the selection; timing includes no clone or encoding.
- Hash checks establish input identity and equal-value hashes within each
  library. The two libraries' hash encodings need not agree.

These benchmarks exercise public production dispatch. The
[internal suite](../internal_improvement/README.md) covers forced tiers. For CPU
pinning, execution order, sampling controls, and measurement schemas, refer to the
[benchmark driver documentation](../../tools/benchmark/README.md).

Working measurements belong under `target/bench-results/`. Reviewed curves follow
the [publication conventions](../../docs/int/benchmarks/README.md).

## Verification

```sh
python3 tools/bench.py check --source-only
python3 tools/structure_audit.py
python3 tools/import_audit.py
cargo clippy --bench public_api --features std,rayon,_internal-tune
python3 tools/bench.py check --features std,rayon,_internal-tune --require-comparison
```

The source audit checks engine names, symmetric attributes, and matching timing
and reset strategies. The compiled catalog rejects bundled-function cases,
missing Mp entries, and redundant FLINT alternatives. The shared templates check
actual numeric equivalence when a case executes; source audits do not prove
arithmetic equivalence.
