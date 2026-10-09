# MpAnafis: Roadmap to Complete `MpUint`

## 0. Master Architecture & Delivery Matrix

```
Phase 0 (Exposure) ──┐
Phase 1 (T0 APIs)  ──┴──► Phase 2 (Keystone Engines)
                            ├── 2.1 Product/Remainder Trees ──┬──► Phase 3.3-3.4 (Batch Mod/GCD)
                            │                                └──► Phase 5 (Combinatorics)
                            ├── 2.2 Garner's CRT ────────────┬──► Phase 4 (Modular Solving)
                            ├── 2.3 Factorization Extraction ─┴──► Phase 3.1-3.2 (Divisors/Group)
                            └── 2.4 Coupled RootRem ─────────────► Phase 2.5 (Power Detection)

Parallel Tracks:
- Phase 6: Ecosystem Integrations (Serde, Rand, Zeroize)
- Pillar 9: Performance Engines (Batch Sieve, Static Parallelism, CtUint, Arch Kernels)
```

---

## Phase 0: Immediate Exposure of In-Tree Foundations

*Goal: Expose production-grade internal engines through public inherent APIs with zero new algorithmic risk.*

- [ ] **0.1 Public Exact Division (`div_exact` / `checked_div_exact`)**
  - **Engine**: Wrap internal `div_exact_into` ([`src/int/logic/unsigned/math/div/divisibility.rs`](src/int/logic/unsigned/math/div/divisibility.rs)) utilizing Hensel single-limb modular inversion and truncated Newton exact quotient.
  - **Contract**: `div_exact(&self, rhs: &Self) -> Self` panics if remainder is non-zero; `checked_div_exact` returns `Option<Self>`.
- [ ] **0.2 Reusable Modular Context Handles**
  - **`MontgomeryContext`**: Encapsulates modulus $M$, precomputed $R \pmod M$, $R^2 \pmod M$, and modular inverse $M' \equiv -M^{-1} \pmod B$ to avoid setup recomputation in iterative loops.
  - **`BarrettContext`**: Encapsulates modulus $M$ and precomputed reciprocal $\mu = \lfloor B^{2k} / M \rfloor$.
  - **`PrecomputedDivisor`**: Encapsulates divisor $D$ and normalized Möller-Granlund 3/2 reciprocal or Newton reciprocal block.
- [ ] **0.3 Public Inherent Limb Serialization**
  - Expose read-only `to_limbs_le(&self) -> Vec<usize>`, `to_limbs_be(&self) -> Vec<usize>`.
  - Expose validated constructors `from_limbs_le(limbs: &[usize]) -> Self`, `from_limbs_be(limbs: &[usize]) -> Self`.
- [ ] **0.4 Public Extended GCD Bézout Exposure (`gcd_cofactors`)**
  - Expose canonical Bézout cofactors from `div/extended.rs` returning `(gcd, x, y)` such that $a \cdot x - b \cdot y = \gcd(a, b)$ for unsigned operands.

---

## Phase 1: High-Consensus & T0 Standard APIs

*Goal: Implement consensus utility and rounding methods requiring no novel math engines.*

- [ ] **1.1 Round-to-Nearest Division Family**
  - Implement `div_round`, `rem_round`, `div_rem_round` (ties to even, matching IEEE-754 / standard rounding).
  - Add checked counterparts: `checked_div_round`, `checked_rem_round`.
- [ ] **1.2 Fused & Averaging Arithmetic**
  - `average(&self, other: &Self) -> Self`: Infallible floor average $\lfloor(a + b)/2\rfloor$ without intermediate addition overflow.
  - `average_ceil(&self, other: &Self) -> Self`: Ceil average $\lceil(a + b)/2\rceil$.
  - `mul_shr_round(&self, other: &Self, shift: usize) -> Self`: Fixed-point rounding product.
- [ ] **1.3 Bitwise Windowing & Metric Operations**
  - `keep_bits(&self, count: usize) -> Self`: Low-order bit masking ($a \bmod 2^k$).
  - `split_bits(&self, at: usize) -> (Self, Self)`: Deconstruct integer into $(high, low)$ at bit boundary.
  - `hamming_distance(&self, other: &Self) -> usize`: Bitwise XOR popcount `(self ^ other).count_ones()`.
- [ ] **1.4 Power-of-Two Modular Arithmetic**
  - Implement `*_mod_2exp`, `add_mod_2exp`, `sub_mod_2exp`, `mul_mod_2exp`.
  - `invert_2exp(&self, k: usize) -> Option<Self>`: Modular inverse modulo $2^k$ via Hensel lifting from seed modulo 8.
  - `balanced_mod(&self, modulus: &Self) -> MpInt`: Symmetric centered remainder in $[-\lfloor(m-1)/2\rfloor, \lfloor m/2 \rfloor]$.
- [ ] **1.5 Number Theory Micro-Utilities**
  - `is_congruent(&self, other: &Self, modulus: &Self) -> bool`: Verifies $(a \equiv b \pmod m)$.
  - `remove_factor(&self, factor: &Self) -> (Self, usize)`: Valuation extraction returning cofactor $a / f^k$ and exponent $k$.
  - `gcd_slice(values: &[Self]) -> Self`: Sequential fold over slice; empty slice returns 0.
  - `cbrt(&self) -> Self`: Cube root sugar over `nth_root(3)`.
  - `ceil_isqrt(&self) -> Self`, `ceil_nth_root(&self, n: u32) -> Self`.
  - `to_f64_exp(&self) -> (f64, i32)`: Decompose large integer into normalized mantissa and binary exponent.
  - `prev_prime(&self) -> Option<Self>`: Backward prime scan using bit-sieve and Baillie-PSW.
  - `kronecker_symbol(&self, other: &Self) -> i8`, `legendre_symbol(&self, prime: &Self) -> i8`.
  - Associated constants: `MpUint::ZERO`, `MpUint::ONE`, `MpUint::TWO`, `MpUint::TEN`, `two()`, `ten()`.

---

## Phase 2: Keystone Algorithmic Engines

*Goal: Build the fundamental asymptotic utilities that unlock high-level algebra and combinatorics.*

- [ ] **2.1 Balanced Product Tree & Remainder Tree Utility (`src/int/logic/unsigned/math/trees/`)**
  - **Product Tree**: Balanced binary tree multiplying integer leaves in $O(M(N) \log k)$ time, preventing polynomial degree imbalance and maximizing SSA/Toom locality.
  - **Remainder Tree**: Top-down remainder evaluation reducing numerator $X$ modulo tree nodes using Newton division.
  - Scratch allocation partitioned across contiguous buffers.
- [ ] **2.2 Garner's Mixed-Radix Chinese Remainder Theorem (`chinese_remainder`)**
  - Reconstruct integer $X$ from residues $r_i \pmod{m_i}$ for pairwise coprime moduli in $O(M(N) \log k)$ time.
  - Compute mixed-radix coefficients $v_i$ using precomputed modular inverses $c_{ij} = m_j^{-1} \pmod{m_i}$.
- [ ] **2.3 Standalone Factorization Engine Extraction (`src/int/logic/unsigned/math/factor/`)**
  - Extract siloed Pollard-Brent rho out of `totient.rs` into dedicated subsystem.
  - Public APIs: `factor(&self) -> Vec<(MpUint, usize)>`, `prime_factors(&self) -> Vec<MpUint>`.
  - Implement cascading factor ladder:
    1. Wheel-30 trial division for primes $\le 5,132$.
    2. Square cofactor detection via `isqrt`.
    3. Pollard-Brent rho with Montgomery representation and cycle batching (128 steps).
    4. Pollard's $p-1$ stage 1 and stage 2 (Baby-Step Giant-Step).
    5. Lenstra Elliptic Curve Method (ECM) using Montgomery parameterization and Suyama's curves.
- [ ] **2.4 Coupled Zimmermann $n$-th Root with Remainder (`nth_root_rem`)**
  - Coupled Newton iteration on $f(y) = y^{-n} - x$ yielding floor root $s = \lfloor x^{1/n} \rfloor$ and exact remainder $r = x - s^n$ simultaneously.
  - Remainder computed directly from the final Newton step refinement via `LowProduct`, avoiding an independent $s^n$ evaluation.
- [ ] **2.5 Exact Prime-Power & Perfect-Power Detection**
  - `is_perfect_power(&self) -> bool`, `is_prime_power(&self) -> bool`.
  - Algorithm: Candidate prime exponent filtering $\le \log_2(x)$, modular residue screening, and coupled `nth_root_rem` verification.

---

## Phase 3: Factor-Dependent Number Theory

*Goal: Advanced arithmetic and group-theoretic structure functions built on factorization and trees.*

- [ ] **3.1 Multiplicative Functions & Divisor Structures**
  - `divisors(&self) -> Vec<Self>`: Cartesian product expansion over canonical prime factor powers.
  - `divisor_count(&self) -> Self`: $d(n) = \prod (e_i + 1)$.
  - `divisor_sum(&self) -> Self`: $\sigma_1(n) = \prod \frac{p_i^{e_i+1} - 1}{p_i - 1}$ via exact division.
  - `is_squarefree(&self) -> bool`: Checks all $e_i = 1$.
  - `radical(&self) -> Self`: $\operatorname{rad}(n) = \prod p_i$.
  - `is_smooth(&self, b: &Self) -> bool`: Checks all prime factors $p_i \le B$.
  - `moebius_mu(&self) -> i8`: $\mu(n) = (-1)^k$ if squarefree, else 0.
  - `carmichael_lambda(&self) -> Self`: $\lambda(n) = \operatorname{lcm}(\lambda(p_i^{e_i}))$ with power-of-2 branch rules.
- [ ] **3.2 Multiplicative Group Order & Generator**
  - `multiplicative_order(&self, modulus: &Self) -> Option<Self>`: Order of $a$ modulo $m$ by factoring $\lambda(m)$ and testing divisors.
  - `primitive_root(&self) -> Option<Self>`: Existence verification for moduli in $\{1, 2, 4, p^k, 2p^k\}$ and generator candidate trial.
- [ ] **3.3 Batch Modular Reduction (`multi_mod`)**
  - `multi_mod(&self, moduli: &[Self]) -> Vec<Self>`: Evaluates $X \pmod{m_i}$ simultaneously across multiple moduli using the remainder tree in sub-quadratic time.
- [ ] **3.4 Sub-Quadratic Batch GCD (`batch_shared_factor_detection`)**
  - Bernstein's FactHacks algorithm: Detects shared pairwise factors in an array of $k$ integers in $O(M(k N) \log k)$ time using product tree, remainder tree over squares $x_i^2$, and exact quotient extraction.

---

## Phase 4: Advanced Modular Equation Solving

*Goal: Non-trivial modular root and logarithm solvers.*

- [ ] **4.1 Modular Square Root (`sqrt_mod`)**
  - **Prime $p$**:
    - $p \equiv 3 \pmod 4$: Direct Euler exponentiation $a^{(p+1)/4} \pmod p$.
    - $p \equiv 5 \pmod 8$: Atkin algorithm.
    - General $p$: Tonelli-Shanks algorithm with quadratic non-residue search.
  - **Prime Power $p^k$**: Hensel lifting from base root modulo $p$.
  - **Composite $m$**: Factor $m \to \prod p_i^{e_i}$, solve components, and combine via Garner CRT.
- [ ] **4.2 Discrete Logarithm (`discrete_log` - Feature-gated `experimental`)**
  - Pohlig-Hellman reduction over prime factors of group order.
  - Baby-Step Giant-Step (BSGS) / Pollard's $\rho$ for small prime-power subgroups.
  - CRT reconstruction into final discrete logarithm.

---

## Phase 5: Combinatorics & Integer Sequences (`combinatorics` feature)

*Goal: Asymptotically optimal sequence generation closed over `MpUint`.*

- [ ] **5.1 Prime Sieve & Primorial**
  - Segmented Sieve of Eratosthenes generating prime streams up to bound $N$ in $O(N)$ time with $O(\sqrt{N})$ scratch memory.
  - `primorial(n: usize) -> Self`: Product of primes $\le n$ evaluated via sieve and balanced product tree.
- [ ] **5.2 Fibonacci & Lucas Sequences via Fast Doubling**
  - `fibonacci(n: usize) -> Self`, `lucas(n: usize) -> Self`.
  - Infallible fast doubling identities:
    $$F_{2k} = F_k(2F_{k+1} - F_k), \quad F_{2k+1} = F_{k+1}^2 + F_k^2$$
    evaluating in $O(M(n) \log n)$ time with scratch buffer reuse.
- [ ] **5.3 Binomial & Catalan Coefficients**
  - `binomial(n: usize, k: usize) -> Self`: Kummer's theorem prime-valuation counting:
    $$v_p\left(\binom{n}{k}\right) = \frac{S_p(k) + S_p(n-k) - S_p(n)}{p - 1}$$
    Evaluates prime powers $p^{v_p}$ and multiplies them via balanced product tree.
  - `catalan(n: usize) -> Self`: Evaluated via Kummer binomial $\binom{2n}{n}$ with valuation adjustment for $n+1$.
  - Rising/falling factorials, double factorial ($n!!$), subfactorial ($!n$).
- [ ] **5.4 Integer Partition Function (`partition`)**
  - `partition(n: usize) -> Self`: Euler's pentagonal number recurrence:
    $$p(n) = \sum_{k \ne 0} (-1)^{k-1} p(n - g_k), \quad g_k = \frac{3k^2 - k}{2}$$
    optimized via rolling cyclic buffer for $n \le 100,000$.

---

## Phase 6: Ecosystem Integrations

*Goal: Seamless interop with standard Rust crate ecosystem.*

- [ ] **6.1 Serde Serialization (`serde` feature)**
  - Human-readable format: decimal string serialization and deserialization with precision policy preservation.
  - Binary format: compact limb slice encoding with endian portability.
- [ ] **6.2 Random Generation (`rand` feature)**
  - `random_bits(bits: usize, rng: &mut R) -> Self`.
  - `random_below(bound: &Self, rng: &mut R) -> Self`: Rejection sampling avoiding modulo bias.
  - `random_range(range: Range<Self>, rng: &mut R) -> Self`.
  - `random_prime(bits: usize, rng: &mut R) -> Self`.
  - `is_probably_prime_with_rng(&self, k: u32, rng: &mut R) -> bool`.
- [ ] **6.3 Interoperability & Security Traits**
  - `zeroize`: Implement `Zeroize` for sensitive memory clearing.
  - `num-bigint`: `From`/`TryFrom` bridge between `MpUint` and `num_bigint::BigUint`.
  - `arbitrary`: Fuzz testing generator implementations for `proptest` and `cargo-fuzz`.

---

## Pillar 9: Foundational Infrastructure & High-Performance Engines

*These pillars provide the extreme-scale hardware and mathematical optimizations for the entire crate.*

### 9.1 Batch Sieve via Sub-Quadratic GCD (Bernstein's Algorithm)
- **Target**: Gigabyte-scale primality candidate testing ($N \ge 65,536$ bits up to 2GB integers).
- **Pipeline**:
  1. Build balanced binary product tree of primes up to sieve bound $B$: $P = \prod_{3 \le p \le B} p$ via SSA.
  2. Perform single Newton-Raphson remainder reduction: $R = N \pmod P$ in $O(M(n))$ time.
  3. Compute sub-quadratic Half-GCD $G = \gcd(R, P)$. $G > 1 \implies$ composite; $G = 1 \implies$ prime to bound $B$.

### 9.2 Dedicated Lean Parallelism Engine (Static Executor)
- **Target**: Replace dynamic Rayon work-stealing overhead with deterministic static scheduling for structured BigInt workloads.
- **Pipeline**:
  1. Policy-driven global runtime (`ParallelConfig`, OS thread affinity pinning to physical cores).
  2. Contiguous scratch buffer partitioning with NUMA / L2 cache locality retention.
  3. Phased barrier synchronization without per-task allocations for FFT butterflies and CRT stages.

### 9.3 Constant-Time Fixed-Precision Cryptographic Engine (`CtUint<const LIMBS: usize>`)
- **Target**: Side-channel-resistant (timing and cache-timing immune) stack-allocated integer arithmetic for secret keys.
- **Types**: `CtUint<const LIMBS: usize>`, with aliases `U128`, `U256`, `U384`, `U512`, `U2048`, `U4096`.
- **Guarantees**: No secret-dependent branches, no secret-dependent memory indexing, constant loop bounds, `no_std` native with zero heap allocation.
- **6 Atomic Primitives**:
  1. `conditional_select(choice, a, b)` / `conditional_swap`: Bitwise mask $b \oplus (\text{mask} \ \& \ (a \oplus b))$ or hardware `cmov`.
  2. `ct_eq(a, b)` / `ct_lt(a, b)`: Cumulative bitwise OR difference and borrow tracking.
  3. `ct_add(a, b)`: Unconditional $N$-limb addition with carry (`adcq`).
  4. `ct_sub(a, b)`: Unconditional $N$-limb subtraction with borrow (`sbbq`).
  5. `ct_mul_wide(a, b)`: Full-width $N \times N \to 2N$ multiplication without zero-skipping (`mulx`).
  6. `ct_shl(a, bits)` / `ct_shr(a, bits)`: Fixed-stage barrel rotator.
- **Modular Pipeline**: Branchless `add_mod`, `sub_mod`, Montgomery CIOS multiplication, and constant-time inversion via Bernstein-Yang.

---

## 10. Definition of Done & Quality Gate

Every task committed under this roadmap must satisfy:
1. **Audits**: `python3 tools/structure_audit.py` (0 findings, production files $\le 500$ lines) and `python3 tools/import_audit.py` (0 findings).
2. **Correctness**: 100% test coverage including property-based tests (`proptest`) across edge widths, zero-values, and crossover boundaries.
3. **Safety**: Every `unsafe` block backed by a mathematically rigorous `// SAFETY:` proof citing verified invariants.
4. **Portability**: Verified zero-warning compilation on 38 cross-compilation targets (including 16-bit AVR/MSP430).
5. **Documentation**: Updates to [`docs/int/api-inventory.md`](docs/int/api-inventory.md) and [`docs/int/spec.md`](docs/int/spec.md).
