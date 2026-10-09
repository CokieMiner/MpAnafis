# Integer fuzzing

This independent Cargo workspace contains two libFuzzer targets backed by a
shared Rust harness and Rug/GMP reference models.

| Target | Public type | Surface |
| --- | --- | --- |
| `math_ops` | `MpUint` | All 159 stable inherent methods, unsigned conversions and traits |
| `int_ops` | `MpInt` | All 169 stable inherent methods, signed conversions and traits |

The [API inventory](../docs/int/api-inventory.md) defines this stable surface.
Precision metadata, scoped contexts, parse diagnostics, and optional numeric
traits are also exercised. Global precision is verified in an isolated native
subprocess. The explicitly unstable `_internal-tune` interface is outside this
workspace's coverage. Rug/GMP is the only external mathematical comparator;
totient uses an independent finite-domain coprime-counting reference.

## Source organization

```text
fuzz_targets/                 libFuzzer entrypoints
src/lib.rs                    registry and shared reference reexports
src/input.rs                  independent header and operand decoding
src/policy.rs                 result intervals, wrapping, saturation, errors
src/bit_reference.rs          GMP bit residues, scans, reversal, rotation
src/reference.rs              values, modular arithmetic, bytes, float rounding
src/theory_reference.rs       totient and fixed-base Miller–Rabin models
src/{signed,unsigned}/
  mod.rs                      explicit case reexports
  dispatch.rs                 selector counts and category dispatch
  support.rs                  independent Mp and GMP operand construction
  arithmetic.rs               unlimited arithmetic and fused assignment
  division.rs                 bounded division modes and divisibility
  bitwise.rs                  infinite/zero extension and power-of-two scaling
  conversion.rs               all radices, byte input/output, casts, parse errors
  theory.rs                   powers, roots, gcd/lcm, primes, totient, factorial
  modular.rs                  bounded/unlimited modular operations
  bounded.rs                  independent operand widths and policy families
  combined.rs                 widening/carrying products and word decomposition
  properties.rs               scans, counts, bit edits/ranges, transforms, ordering
  metadata.rs                 constructors, capacity, precision, scoped contexts
  traits.rs                   ownership forms, conversions, iterators, formatting
src/tests/
  mod.rs                      test registry
  input.rs                    control independence and complete partitioning
  harness.rs                  all-selector value and representation matrices
  reference.rs                reference rounding, domains, and bit orientation
  policies.rs                 expected panics and assignment rollback
  precision.rs                isolated global state and scoped restoration
  coverage.rs                 source/inventory/case-name agreement
  conversion.rs               large radix/byte and float-input fixtures
  primes.rs                   native boundaries and wide primality/search fixtures
  gcd.rs                      common prefixes, asymmetric sizes, tier boundaries
  modular.rs                  sparse exponents and large moduli
```

Registries contain declarations and reexports. Production imports pass through
those facades; shared items use plain `pub`. Tests remain in their own directory.
The target entrypoints call ordinary library drivers, so native tests execute
the same decoding, policy models, and assertions as campaigns.

Generated `corpus/`, `artifacts/`, `coverage/`, and `target/` data are ignored.
Retain reproducible defects in ordinary regression tests. The decoder now has
eleven categories; existing corpus inputs remain readable, but selectors can
exercise different cases from the earlier six-category layout.

## Input and work limits

Each input starts with seven control bytes:

| Byte | Meaning |
| --- | --- |
| 0 | Category modulo 11 |
| 1 | Operation modulo that category's declared count |
| 2 | Flags: `0x80`/`0x40` operand signs, `0x01`/`0x02` unlimited operands, `0x04` unlimited carry, `0x20` carry/modulus sign or bit value |
| 3–4 | Little-endian scalar parameter |
| 5 | Left operand's share of remaining bytes, scaled by 255 |
| 6 | Right operand's share of the remaining tail, scaled by 255 |

The payload contains three big-endian magnitudes: left, right, and modulus/carry.
Empty magnitudes represent zero. Partition arithmetic is checked, and every
payload byte belongs to an operand. Category-specific flags select the relevant
precision and sign controls.

| Category | Cases per target | Checks |
| --- | ---: | --- |
| 0 | 6 | Arithmetic, quotient/remainder, fused assignment, midpoint |
| 1 | 3 | Truncated, Euclidean, floor, ceiling, divisibility |
| 2 | 5 | AND/OR/XOR, shifts, `mul_2exp`/`div_2exp` |
| 3 | 5 | Radices 2–36, arbitrary bytes, primitive/float exports, extrema, invalid parsing |
| 4 | 10 | GCD/LCM, extended GCD, roots, powers, primality/search, Jacobi, totient/factorial |
| 5 | 7 | Add/subtract/multiply/power/inverse/reduction/Montgomery |
| 6 | 6 | Checked, try, wrapping, saturating, overflowing, strict arithmetic and left shifts |
| 7 | 3 | Widening products, additive carries, mixed precision, exact word reconstruction |
| 8 | 4 | Counts/scans, edits/ranges, predicates/order, explicit-width transformations |
| 9 | 4 | Storage, constructors, contexts, destination-precision assignment |
| 10 | 5 | Operators, primitive conversions, shifts, value/iterator traits, optional numeric traits |

Both targets have 58 selectors. Binary policies vary left widths through
1–512 bits and right widths through 1–256 bits, independently of unlimited
precision. Carry widths vary through 1–256 bits. Width transforms use 1–512
bits; scans, edits, ranges, and scaling use bounded scalar counts. Arithmetic,
encoding, and modular operands retain the payload's size.

Power exponents are 0–15, modular exponents have magnitude below 256, root
degrees are 0–10, and factorial inputs are 0–128. Totient projects to magnitudes
below 1024. Prime search projects to 16-bit values; configurable primality uses
at most 512-bit inputs and up to 64 prime bases. Larger primality inputs project
to a native scalar in campaigns. Dedicated tests retain wider prime/search,
sparse-exponent, GCD, and conversion cases, including 32768-byte radix inputs.
These bounds control per-input work; they are not library domain restrictions.

## Reference contracts and permitted differences

Uniquely specified results use independent arithmetic, residue, rounding, or
encoding models. Checks include output values, precision, option/error results,
and overflow flags. Saturation clamps to the mathematical interval; wrapping
reduces modulo `2^P` and decodes signed residues. Signed `MIN % -1` models the
documented bounded overflow even though the mathematical remainder is zero.
Division by zero follows each policy's documented result.

Implementation-dependent results use their mathematical relations:

- Signed extended GCD checks the nonnegative gcd and `a*x + b*y = gcd(a,b)`.
  Unsigned coefficients check `a*x = gcd (mod b)` and `b*y = gcd (mod a)`.
  Coefficient tuples are not compared with GMP's chosen tuple.
- Capacity must accommodate the used limbs and requested reservation; allocator
  growth and the capacity retained after shrinking are not fixed.
- Wide `is_prime` results are probable-prime classifications. GMP and the
  library can use different tests. Wide fixtures check search ordering and the
  library's qualification, proven GMP classifications, acceptance of published
  [Mersenne primes](https://www.mersenne.org/primes/) at 127 and 521 bits, square
  rejection, and the documented base-two requirement. Configurable fixed-base
  Miller–Rabin is modeled with GMP modular exponentiation and `gcd(n,1999!)`
  for screening.
  Native `u64` primality has the exact documented contract.
- Positive totient inputs may return `None` when factorization cannot complete;
  any returned value must equal the independent coprime count. Zero and
  negative signed inputs must be rejected.
- Debug output is checked as diagnostic output, without fixing its text.
  Equality/hash checks compare equal values within a type and allow different
  hashes between signed and unsigned types.

Byte encodings, radix alphabets, precision combination, signed root domains,
Montgomery radix, and floating-point rounding have explicit public contracts
and are checked accordingly. All four binary ownership forms, owned/borrowed
assignment, twelve primitive shift types, supported primitive conversions,
owned/borrowed iterator reductions, formatting, and optional `num-traits`
methods are exercised.

The native inventory guard compares public source methods with documentation
and case identifiers, including type-checked policy macro arguments. It detects
missing method names; selector matrices provide runtime checks. Neither is a
proof of exhaustive path coverage, all architecture backends, allocation
behavior, or every tunable algorithm crossover.

## Verification and campaigns

From the repository root:

```sh
cargo fmt --manifest-path fuzz/Cargo.toml --check
cargo test --locked --manifest-path fuzz/Cargo.toml --lib
cargo test --locked --manifest-path fuzz/Cargo.toml --lib --all-features
cargo clippy --locked --manifest-path fuzz/Cargo.toml --all-targets --all-features -- -D warnings
cargo +nightly miri test --manifest-path fuzz/Cargo.toml --lib --features std
cargo +nightly fuzz build --all-features
cargo +nightly fuzz run math_ops --all-features -- -max_total_time=60 -max_len=1024
cargo +nightly fuzz run int_ops --all-features -- -max_total_time=60 -max_len=1024
```

Nightly and `cargo-fuzz` are required for campaigns. Rug builds GMP through
`gmp-mpfr-sys`, so its native build prerequisites must be available. Campaigns
use cargo-fuzz's default AddressSanitizer and coverage instrumentation.
The [fuzz workflow](../.github/workflows/fuzz.yml) enables all public features
for its CI smoke runs, weekly five-hour campaigns, and manual runs. Native CI
also checks default and all-feature harness tests.

The fuzz crate forwards `std`, `num-traits`, and `rayon`. The first enables scoped
context cases, the second enables numeric trait calls through a direct optional
dependency, and the third enables library parallel dispatch. The default
feature set verifies the library's `core`/`alloc` configuration on the host.

libFuzzer installs a panic hook that aborts before unwinding. Campaigns call
panicking methods only when their reference preconditions hold. Expected
panics, invalid counts/widths, and receiver rollback are verified by native
tests. Global precision runs in a child process so concurrent tests retain
their own ambient state. Miri ignores GMP comparisons, host-filesystem inventory
discovery, and subprocess tests with explicit reasons; decoder, rollback,
invalid-domain, and scoped-restoration tests remain enabled.

`-max_len` limits the whole input in bytes, including its header and operands.
Short campaigns verify harness execution and do not establish exhaustive
coverage. Replay saved inputs with `cargo +nightly fuzz run <target>
<artifact-path> --all-features`.
