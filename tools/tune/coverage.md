# Tuning parameter coverage

`TuningProfile` contains 70 fields: 67 serial search coordinates and three
parallel coordinates. This catalog gives each constant's unit and operation.
[The tuner guide](README.md) describes execution and acceptance. Exact candidate
grids and operand shapes are declared in [`compiled/`](compiled/),
[`tiers/`](tiers/), and [`worker/`](worker/); session plans retain the executed
indices and dimensions.

## Multiplication and squaring

Conventional entries compare forced algorithm roots on identical balanced
operands. Reconstruction policies use their coefficient or split dimensions.

| Constant | Unit | Operation or policy |
| --- | --- | --- |
| `KARATSUBA_THRESHOLD` | Operand limbs | Schoolbook to Karatsuba multiplication. |
| `LOW_PRODUCT_RECURSIVE_THRESHOLD` | Low-product limbs | Triangular schoolbook to Mulders short product. |
| `LOW_PRODUCT_FULL_THRESHOLD` | Low-product limbs | Mulders to truncated full multiplication. |
| `MUL_MOD_BNM1_THRESHOLD` | Requested minimum modulus limbs | Shared cyclic-product admission; direct balanced and 2:1 ring products. Zero disables it. |
| `TOOM_COOK_THRESHOLD` | Operand limbs | Toom-3 multiplication entry. |
| `TOOM_COOK_4_THRESHOLD` | Operand limbs | Toom-4 multiplication entry. |
| `TOOM_COOK_6_THRESHOLD` | Operand limbs | Toom-6 multiplication entry. |
| `TOOM_COOK_85_THRESHOLD` | Operand limbs | Toom-8.5 multiplication entry. |
| `TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS` | Coefficient limbs | Paired coefficient reconstruction in products and squares. |
| `TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS` | Split limbs | Guard expansion or full evaluation products. |
| `SQR_KARATSUBA_THRESHOLD` | Operand limbs | Schoolbook to Karatsuba squaring. |
| `SQR_TOOM_COOK_THRESHOLD` | Operand limbs | Toom-3 squaring entry. |
| `SQR_TOOM_COOK_4_THRESHOLD` | Operand limbs | Toom-4 squaring entry. |
| `SQR_TOOM_COOK_6_THRESHOLD` | Operand limbs | Toom-6 squaring entry. |
| `SQR_TOOM_COOK_85_THRESHOLD` | Operand limbs | Toom-8.5 squaring entry. |

## Transforms

Ring and kernel policies measure complete forced SSA products and squares.
Admission policies measure the production dispatcher on balanced and unbalanced
operands. Planner coefficients select execution policies on those catalogs.

| Constant | Unit | Operation or policy |
| --- | --- | --- |
| `SSA_THRESHOLD` | Operand limbs | Conventional multiplication to SSA. |
| `SQR_SSA_THRESHOLD` | Operand limbs | Conventional squaring to SSA. |
| `BALANCED_TOOM8_THRESHOLD` | Operand limbs | Balanced production Toom-8 entry. |
| `TRANSFORM_MIN_SMALLER_LIMBS` | Shorter-operand limbs | Transform admission floor. |
| `TRANSFORM_MAX_OPERAND_RATIO` | Longer/shorter ratio | Transform admission cap. |
| `LOPSIDED_TRANSFORM_BLOCK_RATIO` | Block/shorter-operand ratio | Unbalanced transform block geometry. |
| `SSA_BASE_MODULUS_BITS` | Ring bits | Conventional or nested pointwise products. |
| `SSA_BNM1_BASECASE_LIMBS` | Ring limbs | Direct or recursive Mersenne products. |
| `SSA_NEGACYCLIC_FACTOR3_THRESHOLD` | Ring limbs | Factor-3 negacyclic decomposition. |
| `SSA_NEGACYCLIC_FACTOR5_THRESHOLD` | Ring limbs | Factor-5 negacyclic decomposition. |
| `SSA_COEFFICIENT_VISIT_OVERHEAD` | Relative cost | Planner coefficient visits. |
| `SSA_BASECASE_COST_WEIGHT_16THS` | Relative cost/16 | Planner basecase products. |
| `SSA_NESTED_COST_PENALTY_16THS` | Relative cost/16 | Planner nested products. |
| `SSA_DIRECT_SHIFT_MAX_LIMBS` | Coefficient limbs | Direct Fermat shifts. |
| `SSA_SHIFT_SCALAR_THRESHOLD` | Shift-span limbs | Scalar or staged negated shifts. |
| `SSA_SHIFT_BLOCK_WIDTH` | Block limbs | Staged shift geometry. |
| `CACHE_BLOCK_BYTES` | Bytes | Convolution working-set budget. |

## Division

Production cells distinguish quotient, remainder, combined output, and
divisibility. Fixtures construct `U = qD + r` with normalized and shifted
divisors. Objectives select the relevant output and geometry classes; complete
phase validation includes the remaining cells.

| Constant | Unit | Operation or policy |
| --- | --- | --- |
| `BURNIKEL_ZIEGLER_THRESHOLD` | Divisor limbs | Complete-remainder recursive entry. |
| `NEWTON_RAPHSON_THRESHOLD` | Divisor limbs | Reciprocal division entry. |
| `BURNIKEL_QUOTIENT_THRESHOLD` | Divisor/prefix limbs | Short or balanced recursive quotient entry. |
| `BURNIKEL_LONG_QUOTIENT_THRESHOLD` | Divisor limbs | Long recursive quotient entry. |
| `NEWTON_QUOTIENT_THRESHOLD` | Divisor/prefix limbs | Reciprocal quotient entry. |
| `BURNIKEL_ZIEGLER_BLOCK_LIMBS` | Recursive divisor limbs | Divide-and-conquer leaf cutoff. |
| `NEWTON_RAPHSON_BASECASE_LIMBS` | Reciprocal problem limbs | Reciprocal leaf cutoff. |
| `APPROXIMATE_DIVISION_BLOCK_LIMBS` | Retained divisor limbs | Guarded quotient leaf cutoff. |
| `NEWTON_SMALL_QUOTIENT_BLOCK_RATIO` | Divisor/quotient ratio | Reciprocal block width for short quotients. |
| `DIVISION_TRUNCATION_RATIO` | Divisor/retained-prefix ratio | Quotient truncation admission. |
| `DIVISION_DIVISIBLE_THRESHOLD` | Divisor and excess limbs | Recursive divisibility admission. |
| `DIVISION_STACK_LIMBS` | Normalized capacity limbs | Stack or pooled normalization storage. |
| `DIVISION_SINGLE_NORMALIZED_PREINVERSE` | Remaining numerator limbs | Normalized scalar preinversion. |
| `DIVISION_SINGLE_UNNORMALIZED_PREINVERSE` | Remaining numerator limbs | Shifted scalar preinversion. |
| `DIVISION_BASECASE_QUOTIENT_MAX_LIMBS` | Quotient-span limbs | Complete basecase division admission. |
| `DIVISION_SMALL_QUOTIENT_MAX` | Leading-limb estimate | Scalar quotient correction admission. |

## Modular arithmetic

Forced reduction algorithms receive identical bases, exponents, and odd moduli.
Complete production calls also participate in installation validation.
`MONTGOMERY_CIOS_MAX_LIMBS` is selected using repeated raw products with a
prepared domain. `MUL_MOD_BNM1_THRESHOLD` uses direct cyclic residues versus
complete multiplication followed by folding into the same modulus. Setup,
Montgomery powers, and forced Newton division are separate phase regression
guards, including candidate and selected cutoff neighbours. The shared product
coordinates precede division and the outer modular-power crossover.

| Constant | Unit | Operation or policy |
| --- | --- | --- |
| `MONTGOMERY_POW_MOD_THRESHOLD` | Modulus limbs | Montgomery to Barrett reduction crossover. |
| `MONTGOMERY_CIOS_MAX_LIMBS` | Modulus limbs, inclusive | Integrated scalar multiplication versus product-based reduction. |

## GCD

Production consumers include GCD, extended GCD, modular inversion, and Jacobi
symbols. The catalog contains random and structured operand pairs. Shared
reduction policies score all families; cofactor policies score extended GCD and
inversion, with complete-family validation afterward.

| Constant | Unit | Operation or policy |
| --- | --- | --- |
| `HGCD_CROSSOVER_THRESHOLD` | Problem limbs | Outer half-GCD entry. |
| `HGCD_BLOCK_THRESHOLD` | Recursive block limbs | Half-GCD recursion cutoff. |
| `LEHMER_FUSED_UPDATE_MAX_LIMBS` | Equal-width limbs | Inclusive fused-update maximum. |
| `WIDE_LEHMER_THRESHOLD` | Problem limbs | Narrow or wide quotient simulation. |
| `LEHMER_BRANCHLESS_THRESHOLD` | Initial problem limbs | Masked narrow quotient corrections. |
| `BINARY_EUCLID_DIVISION_SHIFT` | Operand bit gap | Scalar Euclidean or binary reduction. |
| `EXTENDED_HGCD_CROSSOVER_THRESHOLD` | Problem limbs | Retained-matrix half-GCD entry. |
| `EXTENDED_GCD_WIDE_THRESHOLD` | Initial second-operand limbs | Extended wide-simulation admission. |
| `EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS` | Cofactor limbs | Recursive cofactor completion floor. |
| `EXTENDED_GCD_COFACTOR_BATCH_RATIO` | Cofactor/remainder ratio | Recursive cofactor completion admission. |

## Conversion

Formatting compares schoolbook and recursive algorithms by input limb width.
Parsing measures complete calls by native radix chunks; one chunk contains the
largest digit group whose radix power fits a limb. Entry and reconstruction-leaf
policies use separate coordinates, with coupled refinement and radix guards.

| Constant | Unit | Operation or policy |
| --- | --- | --- |
| `RADIX_FORMAT_DECIMAL_RECURSIVE_THRESHOLD` | Input limbs | Decimal recursive formatting entry. |
| `RADIX_FORMAT_SMALL_RECURSIVE_THRESHOLD` | Input limbs | Recursive formatting entry for non-power-of-two radices 3–9. |
| `RADIX_FORMAT_LARGE_RECURSIVE_THRESHOLD` | Input limbs | Recursive formatting entry for non-power-of-two radices 11–36. |
| `RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD` | Native radix chunks | Decimal recursive parsing entry. |
| `RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD` | Native radix chunks | Recursive parsing entry for non-power-of-two radices 3–9. |
| `RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD` | Native radix chunks | Recursive parsing entry for non-power-of-two radices 11–36. |
| `RADIX_PARSE_LEAF_CHUNKS` | Native radix chunks | Shared schoolbook reconstruction leaf maximum. |

## Parallel execution

Explicit CPU-set runs measure forced SSA products and squares at every pool
width. A separate production gate checks the actual product dispatcher and
boundary shapes under the same worker budgets.

| Constant | Unit | Operation or policy |
| --- | --- | --- |
| `SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD` | CRT half-width limbs | Parallel direct Fermat admission. |
| `SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS` | Workers | Direct Fermat pool-width admission. |
| `SSA_PARALLEL_MIN_LIMB_WORK` | Estimated limb-work/worker | Transform fork admission. |
