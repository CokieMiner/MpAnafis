# Multiplication Tower Plan

This document describes the multiplication tower as it exists, the work that
can still materially improve it, and the evidence required before a new
algorithm enters production.

Scope: unsigned full multiplication, squaring, lopsided products, partial
products, transform backends, scratch ownership, dispatch, and tuning.

## 1. Non-negotiable contract

- Every backend computes the exact mathematical product for every admitted
  input. A probabilistic check cannot turn an approximate product into the
  default result.
- Public validation happens before dispatch. Proved internal kernels are
  infallible and allocation-free once their scratch owner has been prepared.
- One plan variant names one algorithm. Execution and scratch sizing must agree
  on that exact plan.
- Recursive Toom children use a lower tier ceiling, so a failed split cannot
  redispatch to itself.
- Multiplication and squaring have independent thresholds and scratch models.
- Thresholds and hardware-sensitive kernel choices come from the generated
  tuning profile. Generic algorithm files contain no personal-machine values.
- Multi-prime NTT is archived in `discarded/ntt/`; production large-integer
  transform multiplication is driven by the cache-oblivious Fermat-ring SSA engine.

## 2. Implemented tower

| Family | Status | Role |
|---|---|---|
| Schoolbook | production | Quadratic basecase; architecture kernels own the hot limb loops (`mul_basecase_unchecked`, `mul_2_limbs_unchecked`, `add_mul_2_limbs_unchecked`, unrolled `sqr_1..6, 8`). |
| Karatsuba / Toom-2 | production | First subquadratic balanced tier and recursive child tier; stack scratch buffer for shallow depths. |
| Toom-3 | production | Five point products; also retained where higher splits collapse. |
| Toom-4 | production | Seven point products for the middle balanced range. |
| Toom-6 / Toom-6.5 | production | Balanced six-way and adjacent half-step unbalanced split. |
| Toom-8 / Toom-8.5 | production | Highest conventional tier; balanced square and adjacent half-step product. |
| Toom-3×2 | production | Fractional split for selected ratios from 3:2 toward 3:1. |
| Toom-4×3 | production | Fractional split for selected ratios from 4:3 toward 2:1. |
| Lopsided blocking | production | Partitions a long operand into blocks that land on an efficient balanced tier. |
| Low product | production | Computes only the required low limbs for consumers (division, Barrett reduction, reciprocal iteration). |
| Schönhage-Strassen (SSA) | production | Exact large multiplication over Fermat rings $2^N + 1$, including dedicated squaring, odd-factor negacyclic factor-3 & factor-5 splits, $8 \times 8$ matrix cache-blocking, and parallel pointwise execution. |

The production selector offers large transforms before conventional shape
tests, then fractional Toom, lopsided blocking, and finally the balanced tower.
This matters: a transform has no Toom split ratio and must not become
unreachable merely because a shape fails a conventional split predicate.

The conventional tower is mature, not exhausted. Generic interpolation and
scratch work now tends to produce memory wins or low-single-digit timing
changes. Larger gains require one of three things:

1. a better large-product algorithm class (such as Multi-Prime NTT);
2. a benchmark-proven missing operand shape; or
3. a genuinely faster architecture kernel.

## 3. Current engineering constraints

These are design constraints established by the implementation and its focused
benchmarks. Revisit one only with a new mechanism and isolated evidence.

- Carry-propagating add/sub and cross-limb shifts are serial dependency chains.
  SIMD instruction-count reductions alone have not translated into wall-time
  wins for the Fermat representation.
- SSA obtains cheap roots of unity from the limb-aligned Fermat modulus. A
  redundant 52-bit radix would destroy that property; radix 2^52 belongs only
  inside a separate IFMA NTT backend.
- Recursive cache-oblivious transforms remain the portable default. A
  cache-blocked four-step layout stays tuner-selectable, but repeated full
  matrix transposes are not a generic improvement.
- SSA geometry is constrained by roots of unity. Increasing a transform
  exponent can force wider coefficients and a much larger matrix, so geometry
  search must price memory as well as arithmetic.
- Sparse shift/add interpolation is not a portable replacement for limb
  multiplication. It is useful only when an architecture supplies fast
  `addlsh`/`sublsh`-style kernels and a focused benchmark selects them.
- More Toom variants are not automatically better. They increase selector,
  scratch, interpolation, testing, and tuning surface, and their useful shape
  regions shrink when the transform crossover moves downward.

## 4. Priority roadmap

### 4.1 Production Fermat-Ring SSA Transform

The primary large-integer multiplication transform is the cache-oblivious
Fermat-ring SSA engine ($Z/(2^N+1)$). Because twiddle multiplications by $\omega = 2^k$
are pure bit-shifts and limb rotations ($0$ arithmetic multiplications), SSA eliminates
the 3-prime FMA and CRT overhead of floating-point NTTs on multi-megabyte integers.

Key architectural features:
1. **Cache-oblivious 4-way radix-4 bisection**: Subtransforms recursively decompose
   down to L1/L2 cache lines.
2. **2D Matrix Transposition with $8 \times 8$ Cache Blocking**: Prevents L3 cache line
   thrashing on multi-megabyte working sets.
3. **Negacyclic Factor-3 and Factor-5 Decompositions**: Eliminates power-of-two padding
   cliffs with exact $3 \cdot 2^k$ and $5 \cdot 2^k$ transforms.
4. **SIMD ADX Dual-Carry Butterfly**: Executes `adcxq` for sum and `notq` + `adoxq` for difference
   simultaneously in physical CPU condition codes.
5. **Caller-Owned Reusable Scratch**: Guarantees zero heap allocation on the hot path.

### 4.2 Multi-Prime Floating-Point NTT (Archived Reference)

The 3-prime 50-bit floating-point Harvey NTT engine (with AVX2 FMA butterflies,
Truncated Fourier Transform codelets, and Garner CRT reconstruction) is preserved
in `discarded/ntt/` for reference and future multidimensional polynomial exploration.

### 4.3 Improve partial products

`LowProduct` exists. Add high and middle products only through real consumers:

```text
low product     division, reciprocal iteration, reduction
high product    reciprocal and quotient approximation
middle product  Newton iteration, conversion, polynomial-style consumers
```

The goal is to avoid computing limbs that a caller discards. Do not add public
surface merely to complete a taxonomy; first integrate the internal consumer,
prove the requested limb range, and benchmark the whole caller.

### 4.5 Prepared transforms

Repeated multiplication by one large operand can reuse its forward transform.
Any prepared form must be explicit state owned by the caller and tied to one
backend, geometry, limb width, and tuning profile. Do not introduce implicit
address- or hash-keyed caches.

FLINT exposes precomputed transform state for polynomial operations, showing
the useful ownership model even though it is not a general integer API; see the
preconditioned interfaces in [FLINT `fft_small`](https://flintlib.org/doc/fft_small.html).

### 4.6 Unbalanced Toom variants, benchmark-gated

Potential additions are **Toom-4×2**, **Toom-5×3**, and **Toom-6×3**. GMP ships
several unequal-degree Toom functions, and its developers describe them as a
way to fill regions of the two-dimensional operand-shape map. GMP also notes
that faster FFT multiplication reduces the useful region for higher Toom
variants. See [GMP's multiplication overview](https://gmplib.org/manual/Multiplication-Algorithms)
and [developer shape discussion](https://gmplib.org/devel/).

Before implementing one:

1. benchmark production, Toom-3×2, Toom-4×3, lopsided blocking, and GMP on a
   dense two-dimensional shape grid;
2. identify a contiguous region where the missing split is expected to win by
   more than the host noise margin;
3. prototype only that split and require it to own the predicted region;
4. reject it if a lower NTT crossover removes the region.

An unbalanced tier without an owned region is permanent complexity, not a
future optimisation.

### 4.7 SSA maintenance

SSA remains a production backend and a fallback competitor even after NTT is
enabled. Continue tuning its geometry, direct-shift cutoff, factorised
negacyclic products, and optional four-step layout. Prefer exact per-ring pins
only where the cost model is repeatably wrong; sparse pins must not become a
substitute for a coherent model.

### 4.8 Schönhage-Nussbaumer Convolution (Fallback for non-AVX)

While Multi-prime NTT CRT is the ultimate target for modern processors equipped
with SIMD (AVX2/AVX-512), standard CPUs without wide vector units suffer from
heavy root-of-unity computation costs in standard SSA.

Nussbaumer transforms replace expensive modular root computations with recursive
polynomial decompositions, acting effectively as bit-shifts. This must be implemented
as the primary fallback for the "Valley of Death" (131K - 262K bits) on architectures
that lack AVX, ensuring we close the final performance gap against GMP on all hardware.

### 4.9 Public API & Specification Parity

The core multiplication engine supports the full public API defined in `docs/int/spec.md`.
To achieve complete specification parity, the following public multiplication methods
are prioritized for integration across `MpUint` and `MpInt`:

1. **`widening_mul` / `try_widening_mul`**: Double-width product returning `(MpUint, MpUint)`
   representing the lower and upper words.
2. **`carrying_mul` / `try_carrying_mul`**: Double-width product with an additive carry parameter.
3. **`carrying_mul_add`**: Double-width multiply-accumulate with two additive terms.
4. **`mul_2exp`**: Direct shift-multiplication by $2^n$ ($self \times 2^n$) paired with power-of-two division methods.

## 5. Research catalogue

These ideas are mathematically valid or useful on another workload, but they
are not scheduled production work.

### Single-Fermat exact product

Choose one modulus `2^N + 1` large enough that the modular result is the exact
integer product. This removes the current dual `B^n-1`/`B^n+1` CRT merge but
uses a wider ring. It is an isolated forced-backend experiment, not an assumed
improvement. GMP documents this form in its
[FFT multiplication](https://gmplib.org/manual/FFT-Multiplication.html).


### Verified complex FFT

A complex FFT may have excellent throughput, but rounding error makes it a
different correctness problem. An acceptable backend needs conservative chunk
bounds, a rigorous error estimate, exact reconstruction, an independent check,
and deterministic fallback to an exact NTT. A probabilistic modular check alone
does not satisfy the crate's default exactness contract.

### Parallel CPU and GPU transforms

Prime-major NTT work is naturally parallel across primes and transform blocks.
CPU threading should be added only above a size that amortises scheduling and
memory-bandwidth costs. GPU work needs batched or extremely large products to
amortise transfers and belongs in an optional companion crate, not the portable
integer core.

### Fürer-type and Harvey-van der Hoeven multiplication

Fürer-type algorithms improve asymptotic complexity by arranging especially
cheap roots of unity; later work gives bounds of the form
`O(n log n K^log* n)`. Harvey and van der Hoeven ultimately proved
`O(n log n)` integer multiplication. These results define the theoretical
ceiling, not a practical in-memory backend for this library. See
[Even faster integer multiplication](https://arxiv.org/abs/1407.3360) and the
authors' `O(n log n)` work referenced from
[their research page](https://web.maths.unsw.edu.au/~davidharvey/research/nlogn/).

## 6. Deliberately not planned

- Toom-10, Toom-12, or an open-ended balanced Toom staircase.
- A compatibility wrapper preserving an obsolete tier or dispatch path.
- Redundant radix inside the Fermat SSA representation.
- SIMD rewrites of serial carry chains without isolated wall-time evidence.
- Repeated full-matrix transposes as the portable transform strategy.
- Approximate complex FFT as a default exact multiplication backend.
- Automatic prepared-operand caches hidden behind ordinary multiplication.

## 7. Acceptance gates

Every new tier or kernel must pass all applicable gates:

1. **Mathematical proof:** coefficient bounds, exact reconstruction, carry and
   borrow bounds, transform order, scratch layout, and supported limb widths.
2. **Capability:** a side-effect-free admission predicate distinct from the
   performance crossover.
3. **Correctness:** static regressions plus properties over widths, operand
   order, sparse/dense values, dirty destinations, and threshold neighbours.
4. **Memory:** exact caller-owned scratch sizing; no allocation in the timed
   reusable `run` path.
5. **Performance:** CPU-pinned identical operands, reusable buffers, A/B/B/A
   sampling, forced-tier results, then production-dispatch results.
6. **Shape ownership:** an unbalanced tier must win a contiguous shape region,
   not isolated cells.
7. **Portability:** 16-, 32-, and 64-bit proofs and the architecture matrix.
8. **Tuning:** a complete generated-profile field for every hardware-sensitive
   choice, with provisional thresholds active in recursive children.

Do not infer a crossover from stale tables or another library. Rebuild and
measure the current checkout.

## 8. Tuning ownership

`build_support/tuning.rs` owns the complete typed profile. `tools/tune/` owns
measurement policy. Production algorithms consume generated constants and do
not know how tuning is performed.

The tuner order is:

1. compiled Toom kernel choices;
2. multiplication tower with each accepted lower threshold rebuilt into the
   recursive children;
3. independent square tower under the same rule;
4. division;
5. SSA geometry and transform crossovers;
6. end-to-end production validation.

See `tools/tune/README.md` for commands and profile resolution.

## 9. Primary references

- [GMP multiplication algorithms](https://gmplib.org/manual/Multiplication-Algorithms)
- [GMP higher-degree Toom'n'half](https://gmplib.org/manual/Higher-degree-Toom_0027n_0027half)
- [FLINT word-prime FFT and integer multiplication](https://flintlib.org/doc/fft_small.html)
- [FLINT Schönhage-Strassen FFT](https://flintlib.org/doc/fft.html)
- [Harvey and Roche, in-place TFT](https://arxiv.org/abs/1001.5272)
- [Harvey, cache-friendly TFT](https://arxiv.org/abs/0810.3203)
- [Harvey, van der Hoeven, and Lecerf, Fürer-type multiplication](https://arxiv.org/abs/1407.3360)
