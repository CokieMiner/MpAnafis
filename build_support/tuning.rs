//! Shared tuning-profile schema for the build script and host-side autotuner.
//!
//! This file deliberately has no dependency on the library crate. `build.rs`
//! uses it while compiling the crate, and `tools/tune/` uses the same type
//! when it writes a machine-specific override.
//!
//! # What an architecture profile is for
//!
//! Three sources can supply a profile, in falling priority: an
//! `MP_TUNING_PROFILE` path, a committed `src/int/tuned_thresholds.rs` from
//! the autotuner, and finally the architecture profiles below. **The
//! architecture profiles are defaults for hosts that have never been tuned**,
//! so they are chosen to be safe across a whole architecture rather than
//! optimal on any one member of it. A value that is excellent on one
//! microarchitecture and poor on another belongs in a tuned profile, not here.
//!
//! That distinction matters most for values that depend on cache geometry.
//! A crossover between two algorithms of different asymptotic cost moves slowly
//! with hardware and generalises well; a value that names a working-set size,
//! or that pins a transform geometry, does not. Fields of the second kind carry
//! that caveat individually.
//!
//! # Measured against assumed
//!
//! Only `apply_x86_64` is measured. Every other architecture profile is
//! reasoned from the properties of the target — register file, multiply
//! latency, cache scale, vector width — and is explicitly marked where it is a
//! guess. Those numbers are starting points for a tuning run, not results.

/// Transform log above every reachable geometry, used as the
/// `ssa_four_step_min_log` value that selects the recursive transform instead
/// of the explicit four-step layout.
///
/// A transform length of `2^64` coefficients is not representable, so this
/// disables the four-step path on every target without a separate flag.
pub const FOUR_STEP_DISABLED: usize = 64;

/// Constant definitions required in every complete tuned profile.
///
/// Prefixes include the type delimiter so similarly named constants cannot
/// satisfy one another accidentally.
pub const REQUIRED_DEFINITIONS: [&str; 36] = [
    "const RADIX_DECIMAL_RECURSIVE_THRESHOLD:",
    "const RADIX_SMALL_RECURSIVE_THRESHOLD:",
    "const RADIX_LARGE_RECURSIVE_THRESHOLD:",
    "const KARATSUBA_THRESHOLD:",
    "const TOOM_COOK_THRESHOLD:",
    "const TOOM_COOK_4_THRESHOLD:",
    "const TOOM_COOK_6_THRESHOLD:",
    "const TOOM_COOK_85_THRESHOLD:",
    "const TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS:",
    "const TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS:",
    "const SQR_KARATSUBA_THRESHOLD:",
    "const SQR_TOOM_COOK_THRESHOLD:",
    "const SQR_TOOM_COOK_4_THRESHOLD:",
    "const SQR_TOOM_COOK_6_THRESHOLD:",
    "const SQR_TOOM_COOK_85_THRESHOLD:",
    "const BURNIKEL_ZIEGLER_THRESHOLD:",
    "const NEWTON_RAPHSON_THRESHOLD:",
    "const BURNIKEL_ZIEGLER_BLOCK_LIMBS:",
    "const NEWTON_RAPHSON_BASECASE_LIMBS:",
    "const NTT_THRESHOLD:",
    "const SSA_THRESHOLD:",
    "const SQR_SSA_THRESHOLD:",
    "const TRANSFORM_MIN_SMALLER_LIMBS:",
    "const TRANSFORM_MAX_OPERAND_RATIO:",
    "const SSA_GEOMETRY_EXPONENTS:",
    "const SSA_BASE_MODULUS_BITS:",
    "const SSA_BNM1_BASECASE_LIMBS:",
    "const SSA_NEGACYCLIC_FACTOR3_THRESHOLD:",
    "const SSA_NEGACYCLIC_FACTOR5_THRESHOLD:",
    "const SSA_COEFFICIENT_VISIT_OVERHEAD:",
    "const SSA_BASECASE_COST_WEIGHT_16THS:",
    "const SSA_NESTED_COST_PENALTY_16THS:",
    "const SSA_SQRT2_TWIST_PASSES:",
    "const SSA_FOUR_STEP_MIN_LOG:",
    "const SSA_TRANSPOSE_TILE_LIMBS:",
    "const SSA_DIRECT_SHIFT_MAX_LIMBS:",
];

/// Complete integer performance profile consumed by generated builds.
///
/// This includes directly measurable hardware crossovers, cache-sensitive
/// geometry, and planner-model coefficients. The host tuner does not vary a
/// field merely because it appears here: a parameter is autotuned only when a
/// benchmark domain can isolate its effect. Model coefficients remain part of
/// the complete generated profile so builds are reproducible, but retain their
/// architecture defaults until an independently fitted calibration exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TuningProfile {
    pub radix_decimal_recursive: usize,
    pub radix_small_recursive: usize,
    pub radix_large_recursive: usize,
    pub karatsuba: usize,
    pub toom_cook_3: usize,
    pub toom_cook_4: usize,
    /// Entry width for Toom-6.
    ///
    /// A value of `usize::MAX - 1` shadows balanced Toom-6 completely. Equal
    /// Toom-6 and Toom-8.5 thresholds have the same effect because dispatch
    /// offers the higher tier first. Toom-6 remains available to recursive
    /// children and shape-specific paths in either case.
    pub toom_cook_6: usize,
    pub toom_cook_85: usize,
    pub toom85_paired_reconstruction_min_limbs: usize,
    /// Toom-8 evaluation split width where multiplying both guard limbs becomes
    /// cheaper than expanding their cross terms with linear passes.
    pub toom8_full_guard_product_min_split_limbs: usize,
    pub sqr_karatsuba: usize,
    pub sqr_toom_cook_3: usize,
    pub sqr_toom_cook_4: usize,
    pub sqr_toom_cook_6: usize,
    pub sqr_toom_cook_85: usize,
    pub burnikel_ziegler: usize,
    pub newton_raphson: usize,
    /// Base block size for the Burnikel-Ziegler recursion.
    ///
    /// The dispatch threshold [`Self::burnikel_ziegler`] decides *whether* the
    /// tier runs; this field decides the *geometry* of the blocks the recursion
    /// splits into. The two are independent tuning questions: the block size
    /// must be tunable in isolation, and must not be derived from the
    /// threshold, because a disabled tier (threshold at `usize::MAX - 1`) still
    /// has to be forceable by the tuner with a sane block size. Deriving the
    /// block from the threshold is what made a disabled Burnikel-Ziegler
    /// attempt to allocate an astronomically wide block.
    pub burnikel_ziegler_block: usize,
    /// Reciprocal basecase cutoff for Newton-Raphson iteration.
    ///
    /// Below this divisor width the Newton reciprocal is computed with one
    /// Algorithm D division of `B^2n - 1` by `D` instead of the doubling
    /// recursion. It is deliberately independent of [`Self::burnikel_ziegler`]:
    /// reusing that threshold collapses the iteration to its basecase whenever
    /// Burnikel-Ziegler is disabled, which makes Newton-Raphson measure as a
    /// slow wrapper around Algorithm D.
    pub newton_reciprocal_basecase: usize,
    /// Conventional multiplication tower to NTT crossover; zero disables NTT.
    ///
    /// NTT is unfinished, so every current profile keeps this at zero and the
    /// tuner does not vary it.
    pub ntt: usize,
    /// Conventional multiplication tower to SSA crossover; zero disables SSA.
    pub ssa: usize,
    /// Conventional squaring tower to SSA crossover.
    ///
    /// Defaults to the `ssa` value on every unmeasured target rather than
    /// inheriting the x86-64 result, because the squaring tower has a different
    /// cost curve from the multiplication one and the gap between the two is
    /// not portable. Where it *has* been measured the two differ: on x86-64 the
    /// squaring crossover sits 25% below the multiplication one.
    pub sqr_ssa: usize,
    /// Shorter-operand floor below which a transform loses to blocking.
    ///
    /// Shape economics rather than cache: a transform's ring is sized by the
    /// longer operand while only `smaller` limbs of content exist. Generalises
    /// across microarchitectures better than the working-set fields below.
    pub transform_min_smaller_limbs: usize,
    /// Largest longer-to-shorter operand ratio admitted to one transform.
    pub transform_max_operand_ratio: usize,
    /// Exact `(ring_bits, transform_exponent)` overrides.
    ///
    /// **Host-specific.** Each entry overrides the cost model unconditionally
    /// for one ring width, so a wrong entry has no fallback, and the right entry
    /// depends on cache geometry. Entries belong in a tuned profile; an
    /// architecture profile should pin one only where the model is known to be
    /// wrong across the whole architecture. An exponent of zero delegates that
    /// ring to the model; `(0, 0)` is an unused slot.
    pub ssa_geometry_exponents: [(usize, u8); 64],
    /// Widest inner ring left to the multiplication tower.
    ///
    /// **Cache-sensitive.** It decides where the pointwise stage stops being a
    /// tower product and becomes a nested transform, which sets the working set
    /// of the inner level. Expect it to track cache capacity rather than to be
    /// constant across an architecture.
    pub ssa_base_modulus_bits: usize,
    pub ssa_bnm1_basecase_limbs: usize,
    pub ssa_negacyclic_factor3: usize,
    pub ssa_negacyclic_factor5: usize,
    /// Planner-model coefficient retained from the architecture profile.
    ///
    /// End-to-end coordinate timing cannot identify it independently from the
    /// other planner weights and geometry pins, so the default tuner does not
    /// vary it.
    pub ssa_coefficient_visit_overhead: usize,
    /// Planner-model coefficient retained from the architecture profile.
    ///
    /// This needs isolated fitting against lower-tower measurements rather
    /// than a search over whole SSA products.
    pub ssa_basecase_cost_weight_16ths: usize,
    /// Planner surcharge, in sixteenths, on a pointwise stage that enters
    /// another transform instead of returning to the multiplication tower.
    ///
    /// The recursive cost model prices a nested ring by planning it, so a nested
    /// geometry is charged its own modelled transform cost and nothing else.
    /// What that omits is everything nesting costs *outside* the arithmetic: a
    /// second live coefficient matrix, its scratch, and the cache pressure of
    /// carrying both. Measured, nesting is dearer than the model believes, and
    /// the error is proportional to the nested cost rather than additive, which
    /// is why this is a factor.
    ///
    /// Sixteen is neutral. It exists because the nested exponent search and
    /// [`TuningProfile`]'s own centre both had to move onto
    /// `log2/2`, and correcting the search window without correcting this
    /// pricing makes the under-charge bite more often rather than less.
    ///
    /// This remains explicit for reproducible profiles, but is not varied by
    /// the default tuner: without an isolated model-fitting objective it is
    /// confounded with base modulus and per-ring geometry pins.
    pub ssa_nested_cost_penalty_16ths: usize,
    /// Planner operation-count estimate retained from the architecture profile.
    ///
    /// It describes model calibration rather than a directly selectable
    /// kernel boundary, so the default tuner does not vary it.
    pub ssa_sqrt2_twist_passes: usize,
    /// Transform log at which the explicit four-step layout takes over from the
    /// recursive one; [`FOUR_STEP_DISABLED`] keeps the recursive path always.
    ///
    /// The recursive DIF/DIT is cache-*oblivious*: it splits into contiguous
    /// halves, so every sub-transform that fits a cache level completes all its
    /// remaining stages while resident, at every level of the hierarchy and
    /// without being told any capacity. The four-step layout reaches the same
    /// residency only at one scale, needs this threshold tuned to the machine,
    /// and pays two full out-of-place transposes plus a duplicate-matrix
    /// buffer to get there.
    ///
    /// Disabling it is therefore the *more* portable default, not a local
    /// optimisation: it is the option that needs no knowledge of the cache. It
    /// stays reachable for a tuning run on targets whose transpose is cheap
    /// enough to overturn that.
    pub ssa_four_step_min_log: usize,
    pub ssa_transpose_tile_limbs: usize,
    pub ssa_direct_shift_max_limbs: usize,
}

impl TuningProfile {
    /// Conservative architecture-independent profile.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            radix_decimal_recursive: 170,
            radix_small_recursive: 10,
            radix_large_recursive: 12,
            karatsuba: 18,
            toom_cook_3: 240,
            toom_cook_4: 640,
            toom_cook_6: 1_600,
            toom_cook_85: 1_900,
            toom85_paired_reconstruction_min_limbs: 256,
            toom8_full_guard_product_min_split_limbs: 384,
            sqr_karatsuba: 28,
            sqr_toom_cook_3: 64,
            sqr_toom_cook_4: 176,
            sqr_toom_cook_6: 376,
            sqr_toom_cook_85: 376,
            burnikel_ziegler: 160,
            newton_raphson: 2_880,
            burnikel_ziegler_block: 48,
            newton_reciprocal_basecase: 32,
            ntt: 0,
            ssa: 0,
            sqr_ssa: 0,
            transform_min_smaller_limbs: 1_024,
            transform_max_operand_ratio: 32,
            ssa_geometry_exponents: [(0, 0); 64],
            ssa_base_modulus_bits: 16_384,
            ssa_bnm1_basecase_limbs: 32,
            ssa_negacyclic_factor3: 48,
            ssa_negacyclic_factor5: 32,
            ssa_coefficient_visit_overhead: 16,
            ssa_basecase_cost_weight_16ths: 16,
            ssa_nested_cost_penalty_16ths: 16,
            ssa_sqrt2_twist_passes: 4,
            ssa_four_step_min_log: 10,
            ssa_transpose_tile_limbs: 512,
            ssa_direct_shift_max_limbs: 9,
        }
    }

    /// Measured on a 5.09 GHz Zen 5 core with a 16 MiB L3, and the only measured
    /// architecture profile.
    ///
    /// The conventional crossovers below were measured with the ADX/BMI2
    /// basecase selected. That kernel is chosen by *runtime* detection, so an
    /// x86-64 without ADX runs the portable backend against thresholds tuned
    /// for the fast one, and will switch to Karatsuba slightly later than it
    /// should. The error is small — the two backends differ in constant factor,
    /// not in asymptotics — and pre-Haswell x86-64 is rare enough that a second
    /// profile is not worth the branch. A tuning run fixes it exactly.
    const fn apply_x86_64(&mut self) {
        // One CPU-pinned run on the documented Zen 5 host measured 0.28%
        // timing CV and used a 2% acceptance margin. The multiplication tower
        // selected 20, 775, 876, 2049, and 2419 limbs. The square tower selected
        // 56, 202, 320, shadowed Toom-6, then entered Toom-8.5 at 398. Production
        // division selected 192 and 3072 limbs, with a 64-limb Burnikel block;
        // the Newton reciprocal basecase remained 40.
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 20;
        self.toom_cook_3 = 775;
        self.toom_cook_4 = 876;
        self.toom_cook_6 = 2_049;
        self.toom_cook_85 = 2_419;
        self.sqr_karatsuba = 56;
        self.sqr_toom_cook_3 = 202;
        self.sqr_toom_cook_4 = 320;
        // Equal thresholds shadow Toom-6 because Toom-8.5 is offered first.
        self.sqr_toom_cook_6 = 398;
        self.sqr_toom_cook_85 = 398;
        self.burnikel_ziegler = 192;
        self.newton_raphson = 3_072;
        self.burnikel_ziegler_block = 64;
        self.newton_reciprocal_basecase = 40;
        // Forced transform against forced Toom-8.5 on identical balanced
        // operands in one process, repeated across four runs: the transform wins
        // 2816 by 1.6% to 2.7%, 3072 by 2.1% to 3.5%, and then pulls away to 8%,
        // 13% and 18% at 3328, 4096 and 6144. Toom-8.5 wins 2560 by 4.5% to 6.1%.
        //
        // Between those the band is soft rather than sharp: 2624 goes to
        // Toom-8.5 by 1.5%, 2688 to the transform by 2.1%, and 2752 ties exactly,
        // so the winner is not monotone inside it and no single width in there is
        // the crossover. This sits at the top of that band, which is the widest
        // point where the transform wins on every run rather than on average.
        //
        // The earlier value of 3072 came from a coarser ladder that read the tie
        // at 3000, and left 2816 to 3071 on the slower path.
        self.ssa = 2_816;
        // Squaring gets its own crossover because it is a different question,
        // and the answer is not the multiplication one. Forced transform against
        // the dedicated squaring tower: the tower wins at 1800 to 2200 limbs (by
        // 14%, 6% and 5%) and the transform wins from 2400 up (by 2%, 9%, 12%
        // and 8%), so the boundary sits near 2300 rather than at 3072. Reusing
        // the multiplication threshold left every width from 2400 to 3072 on the
        // slower path.
        self.sqr_ssa = 2_304;
        // Forced transform against forced blocking over 58 shapes spanning six
        // widths from 4091 to 65537 limbs and ratios from six to forty to one.
        // Sorted by the *shorter* operand the two winners separate with no
        // crossings and the longer operand drops out: blocking won every shape
        // at 1024 limbs and below, the transform won every shape from 1170 limbs
        // up, and nothing was measured between. This sits near the middle of
        // that gap.
        //
        // Sampling inside the gap afterwards found it soft rather than sharp:
        // from 900 to 1280 limbs the two paths stay within about 10% of each
        // other and the winner is not monotone in the width, and two runs of
        // effectively the same shape (16385 by 1024 and by 1022) disagreed by
        // 0.86x against 1.01x. So the band is worth at most a few percent on a
        // handful of shapes and cannot be resolved further by this benchmark --
        // do not re-tune it on a single run.
        self.transform_min_smaller_limbs = 1_100;
        // Nesting is dearer than the recursive model prices it. Measured by
        // forcing the whole exponent window at each RAM-sized ring: at a 2^26
        // ring the model preferred a nested exponent 10 at 533.3 ms over
        // exponent 11 at 517.9 ms, and the same surcharge that corrects that
        // leaves every other pinned selection from 2^18 to 2^29 unchanged.
        // Twenty is the low end of the interval that does so; twenty-four
        // over-charges and tips 2^26 onto exponent 13.
        self.ssa_nested_cost_penalty_16ths = 20;
        // Past this ratio a single transform loses to blocking the product into
        // transform-sized pieces, because its ring is sized by the longer
        // operand while only `smaller` limbs of content exist. Measured against
        // production blocking over 36 shapes from 16385 to 524289 limbs: the
        // transform wins every ratio up to 16 to 1 by 1.06x to 1.51x, the two
        // are within 10% at 32 to 1, and blocking wins every ratio from 64 to 1
        // by 1.14x to 1.72x.
        self.transform_max_operand_ratio = 32;
        // Where the pointwise stage leaves the tower for a nested transform.
        // Forced balanced transforms from 8192 to 1048577 limbs, geometric mean
        // over the sweep, lower is better:
        //
        //   8192  16384  32768  65536  131072   (candidate)
        //   1.139  1.005  1.000  0.955   0.964
        //
        // 65536 wins 15% at 524289 limbs and 12% at 1048577 and costs 2% at
        // 8192. 131072 is close behind but gives up 10% at 8192. This value is
        // coupled to `ssa_geometry_exponents` below: it reprices every geometry
        // whose inner ring falls near it, so the two must be re-swept together.
        self.ssa_base_modulus_bits = 65_536;
        // Sparse exact corrections for RAM-sized rings where measurements beat
        // the recursive cost model. They do not interpolate and do not affect a
        // parent or child ring; every other width remains model-driven.
        //
        // These three overrides were each measured at their exact ring:
        //
        //   2^27, 2097152-limb operands: e11 467.6ms vs e10 487.1ms
        //   2^28, 4194304-limb operands: e12 1011ms  vs e11 1057ms
        //   2^29, 8388608-limb operands: e12 2.127s  vs e11 2.194s
        //
        // The 2^26 ring agrees with the model and therefore has no override.
        self.ssa_geometry_exponents[0] = (1 << 27, 11);
        self.ssa_geometry_exponents[1] = (1 << 28, 12);
        self.ssa_geometry_exponents[2] = (1 << 29, 12);
        // Worth 3-6% at two through eight million limbs here, and portable for
        // the reason given on the field: the recursive path needs no cache
        // knowledge, the four-step does.
        self.ssa_four_step_min_log = FOUR_STEP_DISABLED;
    }

    /// **Unmeasured.** Reasoned from the ISA; a tuning run supersedes all of it.
    ///
    /// `AArch64` has no ADX equivalent: a 128-bit product needs `MUL` plus
    /// `UMULH`, and carry propagation runs one chain through `ADCS` rather than
    /// x86-64's two independent chains. The basecase is therefore weaker
    /// relative to the higher tiers than on x86-64, so every conventional
    /// crossover sits *below* its x86-64 counterpart — which the values here
    /// already reflect.
    ///
    /// The two transform fields are set explicitly rather than inherited: the
    /// portable defaults assume a small cache and a tuned four-step layout,
    /// and neither assumption suits the hardware this target runs on.
    /// Conventional and division thresholds preserve this ISA's prior
    /// relative scaling against the measured x86-64 profile, rounded to useful
    /// prediction granularity.
    const fn apply_aarch64(&mut self) {
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 16;
        self.toom_cook_3 = 200;
        self.toom_cook_4 = 550;
        self.toom_cook_6 = 1_360;
        self.toom_cook_85 = 1_600;
        self.sqr_karatsuba = 28;
        self.sqr_toom_cook_3 = 50;
        self.sqr_toom_cook_4 = 150;
        self.sqr_toom_cook_6 = 320;
        self.sqr_toom_cook_85 = 320;
        self.burnikel_ziegler = 120;
        self.newton_raphson = 2_400;
        self.burnikel_ziegler_block = 40;
        self.newton_reciprocal_basecase = 24;
        self.ssa = 4_096;
        self.sqr_ssa = 4_096;
        // The portable default of 16384 bits assumes a small cache. Every
        // 64-bit AArch64 host this crate is likely to meet -- Apple M-series,
        // Graviton, Ampere, recent Snapdragon -- has last-level cache at least
        // comparable to the Zen host where 65536 measured best, and the field
        // tracks cache capacity. Half the x86-64 value is the conservative
        // reading of that: it moves toward the measured answer without assuming
        // the largest cache in the family.
        self.ssa_base_modulus_bits = 32_768;
        // Cache-oblivious beats cache-tuned when the cache is unknown, and this
        // family spans 4 MiB to 32 MiB of last-level cache, so no single
        // four-step threshold could serve it. See the field documentation.
        self.ssa_four_step_min_log = FOUR_STEP_DISABLED;
    }

    /// **Unmeasured.** POWER and s390x, reasoned from the ISA.
    ///
    /// Both are big-core server designs with wide multipliers — POWER has
    /// `mulld`/`mulhdu` and s390x `MLGR` producing a 128-bit product in a
    /// register pair — so the basecase is comparatively strong and the
    /// Karatsuba crossover belongs near the x86-64 value rather than the
    /// `AArch64` one. Both also carry large last-level caches, so the inner-ring
    /// field follows the same reasoning as `AArch64`.
    ///
    /// These two targets share a profile because the tower has never been run
    /// on either; that grouping is an admission of ignorance, not a claim that
    /// they behave alike. They should split as soon as one is measured.
    const fn apply_power_s390x(&mut self) {
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 20;
        self.sqr_karatsuba = 36;
        self.sqr_toom_cook_3 = 60;
        self.burnikel_ziegler = 160;
        self.newton_raphson = 2_800;
        self.burnikel_ziegler_block = 48;
        self.newton_reciprocal_basecase = 32;
        self.ssa = 4_096;
        self.sqr_ssa = 4_096;
        self.ssa_base_modulus_bits = 32_768;
        self.ssa_four_step_min_log = FOUR_STEP_DISABLED;
    }

    /// **Unmeasured.** Any other 64-bit target: `RISC-V`, `LoongArch`, `MIPS64`,
    /// `SPARC64`, `wasm64`.
    ///
    /// The floor of the 64-bit family. A 128-bit product may need two
    /// instructions or a library call, and the crossovers are set below the
    /// `AArch64` ones on the assumption that the basecase is the weakest part.
    /// The four-step layout is disabled here for a stronger reason than
    /// elsewhere: these targets span microcontroller-class caches to server
    /// parts, so no threshold could describe them, and the cache-oblivious path
    /// is the only one that adapts on its own.
    const fn apply_generic_64(&mut self) {
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 16;
        self.toom_cook_3 = 192;
        self.toom_cook_4 = 550;
        self.toom_cook_6 = 1_360;
        self.toom_cook_85 = 1_600;
        self.sqr_karatsuba = 28;
        self.sqr_toom_cook_3 = 48;
        self.sqr_toom_cook_4 = 150;
        self.sqr_toom_cook_6 = 320;
        self.sqr_toom_cook_85 = 320;
        self.burnikel_ziegler = 112;
        self.newton_raphson = 2_300;
        self.burnikel_ziegler_block = 40;
        self.newton_reciprocal_basecase = 24;
        self.ssa = 4_096;
        self.sqr_ssa = 4_096;
        self.ssa_four_step_min_log = FOUR_STEP_DISABLED;
    }

    /// **Unmeasured.** 32-bit hosts with an operating system.
    ///
    /// A limb is half as wide, so a product of a given bit width needs twice
    /// the limbs and four times the basecase work, which pushes every crossover
    /// up in limb terms even though the bit-width crossover barely moves. The
    /// values here are roughly the 64-bit ones scaled by that reasoning.
    const fn apply_std32(&mut self) {
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 28;
        self.toom_cook_3 = 320;
        self.toom_cook_4 = 900;
        self.toom_cook_6 = 2_400;
        self.toom_cook_85 = 2_800;
        self.sqr_karatsuba = 42;
        self.sqr_toom_cook_3 = 80;
        self.sqr_toom_cook_4 = 250;
        self.sqr_toom_cook_6 = 560;
        self.sqr_toom_cook_85 = 560;
        self.burnikel_ziegler = 230;
        self.newton_raphson = 3_600;
        self.burnikel_ziegler_block = 80;
        self.newton_reciprocal_basecase = 48;
        self.ssa = 8_192;
        self.sqr_ssa = 8_192;
        self.ssa_four_step_min_log = FOUR_STEP_DISABLED;
    }

    /// **Unmeasured.** 32-bit embedded targets and `wasm32`.
    ///
    /// `ssa` and `sqr_ssa` stay at the portable zero, which disables the
    /// transform outright. That is deliberate rather than unfinished: the
    /// transform's scratch is proportional to the product and these targets are
    /// memory-constrained, so a tier that allocates a multiple of the operand
    /// is the wrong trade even where it would be faster. Operands large enough
    /// to want a transform are not the workload here.
    const fn apply_embedded_wasm(&mut self) {
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 14;
        self.toom_cook_3 = 160;
        self.toom_cook_4 = 450;
        self.toom_cook_6 = 1_200;
        self.toom_cook_85 = 1_400;
        self.sqr_karatsuba = 22;
        self.sqr_toom_cook_3 = 40;
        self.sqr_toom_cook_4 = 125;
        self.sqr_toom_cook_6 = 275;
        self.sqr_toom_cook_85 = 275;
        self.burnikel_ziegler = 96;
        self.newton_raphson = 1_900;
        self.burnikel_ziegler_block = 32;
        self.newton_reciprocal_basecase = 20;
    }

    /// **Unmeasured.** AVR and `MSP430`, and any other 16-bit target.
    ///
    /// A 16-bit limb makes the basecase sixteen times cheaper per limb pair
    /// than a 64-bit one but needs four times as many limbs for the same value,
    /// so the crossovers move up sharply in limb terms. On parts where a
    /// multiply is multi-cycle or synthesised, the basecase stays competitive
    /// far longer than any 64-bit intuition suggests, which is why these are the
    /// highest crossovers in the file. The transform is disabled as for
    /// `apply_embedded_wasm`.
    const fn apply_16bit(&mut self) {
        self.radix_decimal_recursive = 170;
        self.radix_small_recursive = 10;
        self.radix_large_recursive = 12;
        self.karatsuba = 32;
        self.toom_cook_3 = 384;
        self.toom_cook_4 = 1_100;
        self.toom_cook_6 = 2_700;
        self.toom_cook_85 = 3_200;
        self.sqr_karatsuba = 42;
        self.sqr_toom_cook_3 = 96;
        self.sqr_toom_cook_4 = 300;
        self.sqr_toom_cook_6 = 625;
        self.sqr_toom_cook_85 = 625;
        self.burnikel_ziegler = 270;
        self.newton_raphson = 4_300;
        self.burnikel_ziegler_block = 90;
        self.newton_reciprocal_basecase = 56;
    }

    /// Render a complete Rust source profile with `header` above the constants.
    #[must_use]
    pub fn render(self, header: &str) -> String {
        format!(
            "{header}\n\
             /// Formatting crossover: Schoolbook to recursive for radix 10.\n\
             pub const RADIX_DECIMAL_RECURSIVE_THRESHOLD: usize = {};\n\
             /// Formatting crossover: Schoolbook to recursive for radices 3..=9.\n\
             pub const RADIX_SMALL_RECURSIVE_THRESHOLD: usize = {};\n\
             /// Formatting crossover: Schoolbook to recursive for radices 11..=36.\n\
             pub const RADIX_LARGE_RECURSIVE_THRESHOLD: usize = {};\n\
             /// Multiplication crossover: Schoolbook to Karatsuba.\n\
             pub const KARATSUBA_THRESHOLD: usize = {};\n\
             /// Multiplication crossover: Karatsuba to Toom-Cook 3.\n\
             pub const TOOM_COOK_THRESHOLD: usize = {};\n\
             /// Multiplication crossover: Toom-Cook 3 to Toom-Cook 4.\n\
             pub const TOOM_COOK_4_THRESHOLD: usize = {};\n\
             /// Multiplication crossover: Toom-Cook 4 to Toom-Cook 6.\n\
             pub const TOOM_COOK_6_THRESHOLD: usize = {};\n\
             /// Multiplication crossover: Toom-Cook 6 to Toom-Cook 8.5.\n\
             pub const TOOM_COOK_85_THRESHOLD: usize = {};\n\
             /// Packed coefficient width where Toom-8.5 uses paired reconstruction adds.\n\
             pub const TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS: usize = {};\n\
             /// Toom-8 split width where point products retain both guard limbs.\n\
             pub const TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS: usize = {};\n\
             /// Squaring crossover: Schoolbook to Karatsuba.\n\
             pub const SQR_KARATSUBA_THRESHOLD: usize = {};\n\
             /// Squaring crossover: Karatsuba to Toom-Cook 3.\n\
             pub const SQR_TOOM_COOK_THRESHOLD: usize = {};\n\
             /// Squaring crossover: Toom-Cook 3 to Toom-Cook 4.\n\
             pub const SQR_TOOM_COOK_4_THRESHOLD: usize = {};\n\
             /// Squaring crossover: Toom-Cook 4 to Toom-Cook 6.\n\
             pub const SQR_TOOM_COOK_6_THRESHOLD: usize = {};\n\
             /// Squaring crossover: Toom-Cook 6 to Toom-Cook 8.5.\n\
             pub const SQR_TOOM_COOK_85_THRESHOLD: usize = {};\n\
             /// Division crossover: Algorithm D to Burnikel-Ziegler.\n\
             pub const BURNIKEL_ZIEGLER_THRESHOLD: usize = {};\n\
             /// Division crossover: Burnikel-Ziegler to Newton-Raphson.\n\
             pub const NEWTON_RAPHSON_THRESHOLD: usize = {};\n\
             /// Burnikel-Ziegler recursion base block size, in limbs.\n\
             pub const BURNIKEL_ZIEGLER_BLOCK_LIMBS: usize = {};\n\
             /// Newton-Raphson reciprocal basecase cutoff, in limbs.\n\
             pub const NEWTON_RAPHSON_BASECASE_LIMBS: usize = {};\n\
             /// Multiplication crossover: Toom-Cook 8.5 to multi-prime NTT; zero disables it.\n\
             pub const NTT_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Multiplication crossover: conventional tower to SSA.\n\
             pub const SSA_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Squaring crossover: dedicated squaring tower to SSA; zero disables it.\n\
             pub const SQR_SSA_THRESHOLD: usize = {};\n\
             /// Shortest operand, in limbs, worth padding to a transform ring.\n\
             pub const TRANSFORM_MIN_SMALLER_LIMBS: usize = {};\n\
             /// Widest operand ratio a single transform still beats blocking at.\n\
             pub const TRANSFORM_MAX_OPERAND_RATIO: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Exact (ring bits, transform exponent) overrides; zero delegates to the model.\n\
             #[allow(clippy::unreadable_literal, reason = \"generated code\")]\n\
             pub const SSA_GEOMETRY_EXPONENTS: [(usize, u8); 64] = {:?};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Widest SSA inner ring, in bits, handled by the multiplication tower.\n\
             pub const SSA_BASE_MODULUS_BITS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Widest direct B^n-1 multiplication-and-fold, in limbs.\n\
             pub const SSA_BNM1_BASECASE_LIMBS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Factor-3 Fermat-product crossover, in coefficient limbs.\n\
             pub const SSA_NEGACYCLIC_FACTOR3_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Factor-5 Fermat-product crossover, in coefficient limbs.\n\
             pub const SSA_NEGACYCLIC_FACTOR5_THRESHOLD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Planner cost of one coefficient visit relative to one limb multiply.\n\
             pub const SSA_COEFFICIENT_VISIT_OVERHEAD: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Planner interpolation weight from an n^1.5 to n^1.75 basecase model.\n\
             pub const SSA_BASECASE_COST_WEIGHT_16THS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Planner surcharge, in sixteenths, on a nested pointwise stage.\n\
             pub const SSA_NESTED_COST_PENALTY_16THS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Planner penalty for an odd half-step pre-twist.\n\
             pub const SSA_SQRT2_TWIST_PASSES: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Transform log at which SSA switches to the cache-blocked four-step FFT.\n\
             pub const SSA_FOUR_STEP_MIN_LOG: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Limb working-set budget for one cache-blocked transpose tile.\n\
             pub const SSA_TRANSPOSE_TILE_LIMBS: usize = {};\n\
             #[cfg(not(target_pointer_width = \"16\"))]\n\
             /// Largest coefficient width using the direct Fermat shift loop.\n\
             pub const SSA_DIRECT_SHIFT_MAX_LIMBS: usize = {};\n",
            self.radix_decimal_recursive,
            self.radix_small_recursive,
            self.radix_large_recursive,
            self.karatsuba,
            self.toom_cook_3,
            self.toom_cook_4,
            self.toom_cook_6,
            self.toom_cook_85,
            self.toom85_paired_reconstruction_min_limbs,
            self.toom8_full_guard_product_min_split_limbs,
            self.sqr_karatsuba,
            self.sqr_toom_cook_3,
            self.sqr_toom_cook_4,
            self.sqr_toom_cook_6,
            self.sqr_toom_cook_85,
            self.burnikel_ziegler,
            self.newton_raphson,
            self.burnikel_ziegler_block,
            self.newton_reciprocal_basecase,
            self.ntt,
            self.ssa,
            self.sqr_ssa,
            self.transform_min_smaller_limbs,
            self.transform_max_operand_ratio,
            self.ssa_geometry_exponents,
            self.ssa_base_modulus_bits,
            self.ssa_bnm1_basecase_limbs,
            self.ssa_negacyclic_factor3,
            self.ssa_negacyclic_factor5,
            self.ssa_coefficient_visit_overhead,
            self.ssa_basecase_cost_weight_16ths,
            self.ssa_nested_cost_penalty_16ths,
            self.ssa_sqrt2_twist_passes,
            self.ssa_four_step_min_log,
            self.ssa_transpose_tile_limbs,
            self.ssa_direct_shift_max_limbs,
        )
    }
}

impl Default for TuningProfile {
    fn default() -> Self {
        Self::portable()
    }
}

/// Select the conservative built-in profile for a target architecture.
#[must_use]
pub fn profile_for_target(target_arch: &str, pointer_width: &str) -> TuningProfile {
    let mut profile = TuningProfile::portable();
    match pointer_width {
        "64" => match target_arch {
            "x86_64" => profile.apply_x86_64(),
            "aarch64" | "arm64ec" => profile.apply_aarch64(),
            "powerpc64" | "powerpc64le" | "s390x" => profile.apply_power_s390x(),
            _ => profile.apply_generic_64(),
        },
        "32" => match target_arch {
            "wasm32" | "riscv32" | "loongarch32" | "xtensa" => {
                profile.apply_embedded_wasm();
            }
            _ => profile.apply_std32(),
        },
        "16" => profile.apply_16bit(),
        _ => profile.apply_embedded_wasm(),
    }
    profile
}

/// Return the first missing definition in a tuned profile source.
#[must_use]
pub fn missing_definition(source: &str) -> Option<&'static str> {
    REQUIRED_DEFINITIONS
        .iter()
        .find(|definition| !source.contains(**definition))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::profile_for_target;

    #[test]
    fn pointer_width_precedes_isa_specific_profile() {
        let x32 = profile_for_target("x86_64", "32");
        let aarch64_ilp32 = profile_for_target("aarch64", "32");
        let hypothetical_x86_16 = profile_for_target("x86_64", "16");

        assert_eq!(x32.karatsuba, 28);
        assert_eq!(x32.ssa, 8_192);
        assert_eq!(aarch64_ilp32.karatsuba, 28);
        assert_eq!(aarch64_ilp32.ssa, 8_192);
        assert_eq!(hypothetical_x86_16.karatsuba, 32);
        assert_eq!(hypothetical_x86_16.ssa, 0);
    }

    #[test]
    fn architecture_profiles_preserve_dispatch_order() {
        for (architecture, width) in [
            ("x86_64", "64"),
            ("aarch64", "64"),
            ("powerpc64le", "64"),
            ("riscv64", "64"),
            ("x86", "32"),
            ("wasm32", "32"),
            ("avr", "16"),
        ] {
            let profile = profile_for_target(architecture, width);
            assert!(profile.karatsuba < profile.toom_cook_3);
            assert!(profile.toom_cook_3 < profile.toom_cook_4);
            assert!(profile.toom_cook_4 < profile.toom_cook_6);
            assert!(profile.toom_cook_6 < profile.toom_cook_85);
            assert!(profile.sqr_karatsuba < profile.sqr_toom_cook_3);
            assert!(profile.sqr_toom_cook_3 < profile.sqr_toom_cook_4);
            assert!(profile.sqr_toom_cook_4 < profile.sqr_toom_cook_85);
            assert!(
                profile.sqr_toom_cook_6 == usize::MAX - 1
                    || profile.sqr_toom_cook_6 == profile.sqr_toom_cook_85
            );
            assert!(profile.burnikel_ziegler < profile.newton_raphson);
            if profile.ssa != 0 {
                assert!(profile.toom_cook_85 < profile.ssa);
                assert!(profile.sqr_toom_cook_85 < profile.sqr_ssa);
            }
        }
    }
}
