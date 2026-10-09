#!/usr/bin/env bash
set -uo pipefail

# Each target covers a limb width, endianness, ABI, instruction set, or feature
# selector. Targets without prebuilt core/alloc libraries use rust-src.

readonly -a PREBUILT_STD_TARGETS=(
    "x86_64-unknown-linux-gnu"
    "x86_64-unknown-linux-gnux32"
    "i686-unknown-linux-gnu"
    "aarch64-unknown-linux-gnu"
    "arm64ec-pc-windows-msvc"
    "arm-unknown-linux-gnueabi"
    "armv7-unknown-linux-gnueabihf"
    "powerpc-unknown-linux-gnu"
    "powerpc64-unknown-linux-gnu"
    "powerpc64le-unknown-linux-gnu"
    "s390x-unknown-linux-gnu"
    "riscv64gc-unknown-linux-gnu"
    "loongarch64-unknown-linux-gnu"
    "sparc64-unknown-linux-gnu"
    "wasm32-unknown-unknown"
)

readonly -a PREBUILT_NO_STD_TARGETS=(
    "thumbv6m-none-eabi"
    "thumbv7em-none-eabi"
    "riscv32i-unknown-none-elf"
    "riscv32im-unknown-none-elf"
    "nvptx64-nvidia-cuda"
)

# Entries use TARGET|EXTRA_RUSTFLAGS. Source-built m68k compiler_builtins
# requires release code generation. avr-none requires an explicit CPU.
readonly -a SOURCE_NO_STD_TARGETS=(
    "aarch64-unknown-linux-gnu_ilp32|"
    "aarch64_be-unknown-none-softfloat|"
    "armebv7r-none-eabi|"
    "avr-none|-C target-cpu=atmega328p"
    "bpfeb-unknown-none|"
    "bpfel-unknown-none|"
    "csky-unknown-linux-gnuabiv2|"
    "hexagon-unknown-none-elf|"
    "loongarch32-unknown-none|"
    "m68k-unknown-none-elf|"
    "mips-unknown-linux-gnu|"
    "mipsel-unknown-linux-gnu|"
    "mips64-unknown-linux-gnuabi64|"
    "mips64el-unknown-linux-gnuabi64|"
    "msp430-none-elf|"
    "riscv32e-unknown-none-elf|"
    "sparc-unknown-none-elf|"
    "wasm64-unknown-unknown|"
)

# These targets are still attempted. A failure in mp-anafis is fatal, while
# a failure building rust-src itself is reported as an explicit toolchain block.
readonly -a TOOLCHAIN_PROBE_TARGETS=(
    "xtensa-esp32-none-elf|"
)

readonly -a X86_64_FEATURE_PROFILES=(
    "BMI2 only|-C target-feature=+bmi2,-adx"
    "ADX only|-C target-feature=+adx,-bmi2"
    "ADX and BMI2|-C target-feature=+adx,+bmi2"
    "AVX2 without ADX|-C target-feature=+avx2,-adx"
    "AVX-512 and AVX2|-C target-feature=+avx512f,+avx2"
)

# POWER9 kernels require compile-time ISA 3.0 selection. Both endiannesses
# exercise the shared kernels; POWER8 exercises the baseline selection.
readonly -a POWERPC64_FEATURE_PROFILES=(
    "POWER9 little-endian|powerpc64le-unknown-linux-gnu|-C target-cpu=pwr9"
    "POWER9 big-endian|powerpc64-unknown-linux-gnu|-C target-cpu=pwr9"
    "POWER8 baseline|powerpc64le-unknown-linux-gnu|-C target-cpu=pwr8"
)

# ARMv7 with NEON exercises the portable 32-bit NTT fallback.
readonly -a ARM_FEATURE_PROFILES=(
    "ARMv7 NEON fallback|thumbv7neon-unknown-linux-gnueabihf|"
)

if [[ -t 1 ]]; then
    readonly BLUE=$'\033[1;34m'
    readonly CYAN=$'\033[1;36m'
    readonly GREEN=$'\033[1;32m'
    readonly YELLOW=$'\033[1;33m'
    readonly RED=$'\033[1;31m'
    readonly RESET=$'\033[0m'
else
    readonly BLUE=""
    readonly CYAN=""
    readonly GREEN=""
    readonly YELLOW=""
    readonly RED=""
    readonly RESET=""
fi

declare -a FAILURES=()
declare -a TOOLCHAIN_BLOCKS=()
PASSED_CHECKS=0
PASSED_BUILDS=0
readonly REQUIRED_TARGETS=$((${#PREBUILT_STD_TARGETS[@]} + ${#PREBUILT_NO_STD_TARGETS[@]} + ${#SOURCE_NO_STD_TARGETS[@]}))

readonly BASE_RUSTFLAGS="${RUSTFLAGS-}"
if ! LOG_DIR="$(mktemp -d)"; then
    printf 'Unable to create the architecture-check log directory.\n' >&2
    exit 1
fi
readonly LOG_DIR
trap 'rm -rf "$LOG_DIR"' EXIT

run_clippy() {
    local label="$1"
    local extra_rustflags="$2"
    shift 2

    printf '  -> %s\n' "$label"
    if [[ -n "$extra_rustflags" ]]; then
        local combined_rustflags="$extra_rustflags"
        if [[ -n "$BASE_RUSTFLAGS" ]]; then
            combined_rustflags="$BASE_RUSTFLAGS $extra_rustflags"
        fi
        if env RUSTFLAGS="$combined_rustflags" cargo clippy "$@" -- -D warnings; then
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            FAILURES+=("$label")
        fi
    elif cargo clippy "$@" -- -D warnings; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        FAILURES+=("$label")
    fi
}

run_codegen() {
    local label="$1"
    local extra_rustflags="$2"
    shift 2
    local combined_rustflags="$BASE_RUSTFLAGS"
    if [[ -n "$extra_rustflags" ]]; then
        combined_rustflags="${BASE_RUSTFLAGS:+$BASE_RUSTFLAGS }$extra_rustflags"
    fi

    # Clippy checks Rust IR but does not assemble inline instructions. Actual
    # code generation catches ISA-extension and instruction-mode mismatches.
    printf '  -> %s (code generation)\n' "$label"
    if env RUSTFLAGS="$combined_rustflags" cargo build "$@"; then
        PASSED_BUILDS=$((PASSED_BUILDS + 1))
    else
        FAILURES+=("$label: code generation")
    fi
}

check_target() {
    local target="$1"
    local build_kind="$2"
    local supports_std="$3"
    local extra_rustflags="$4"
    local -a build_std_args=()
    local -a codegen_profile_args=()

    printf '\n%s=== Checking %s ===%s\n' "$CYAN" "$target" "$RESET"

    if [[ "$build_kind" == "source" ]]; then
        build_std_args=(-Z build-std=core,alloc)
        # Source-built compiler_builtins requires optimized code on m68k.
        codegen_profile_args=(--release)
    fi

    run_clippy \
        "$target: no features (no_std)" \
        "$extra_rustflags" \
        "${build_std_args[@]}" --release --locked --lib --no-default-features --target "$target"
    run_clippy \
        "$target: num-traits (no_std)" \
        "$extra_rustflags" \
        "${build_std_args[@]}" --release --locked --lib --no-default-features \
        --features num-traits --target "$target"

    if [[ "$supports_std" == "yes" ]]; then
        run_clippy \
            "$target: std" \
            "$extra_rustflags" \
            --release --locked --lib --no-default-features --features std --target "$target"
        run_clippy \
            "$target: std and num-traits" \
            "$extra_rustflags" \
            --release --locked --lib --no-default-features \
            --features std,num-traits --target "$target"
    fi
    run_codegen "$target: no_std" "$extra_rustflags" \
        "${build_std_args[@]}" "${codegen_profile_args[@]}" \
        --locked --lib --no-default-features --target "$target"
}

target_is_known() {
    local target="$1"
    grep -Fxq "$target" <<<"$RUST_TARGET_LIST"
}

printf '%sPreparing clippy, rust-src, and prebuilt target libraries...%s\n' "$BLUE" "$RESET"
if ! rustup component add clippy rust-src; then
    printf '%sUnable to install the clippy/rust-src components.%s\n' "$RED" "$RESET" >&2
    exit 1
fi

readonly RUST_TARGET_LIST="$(rustc --print target-list)"

for target in "${PREBUILT_STD_TARGETS[@]}" "${PREBUILT_NO_STD_TARGETS[@]}"; do
    if ! target_is_known "$target"; then
        FAILURES+=("$target: target is absent from this rustc")
    elif ! rustup target add "$target"; then
        FAILURES+=("$target: rustup target installation")
    fi
done

for entry in "${ARM_FEATURE_PROFILES[@]}"; do
    IFS='|' read -r _ target _ <<<"$entry"
    if ! target_is_known "$target"; then
        FAILURES+=("$target: ARM profile target is absent from this rustc")
    elif ! rustup target list --installed | grep -Fxq "$target"; then
        if ! rustup target add "$target"; then
            FAILURES+=("$target: ARM profile target installation")
        fi
    fi
done

printf '\n%sChecking prebuilt std targets...%s\n' "$BLUE" "$RESET"
for target in "${PREBUILT_STD_TARGETS[@]}"; do
    if rustup target list --installed | grep -Fxq "$target"; then
        check_target "$target" "prebuilt" "yes" ""
    fi
done

printf '\n%sChecking prebuilt no_std targets...%s\n' "$BLUE" "$RESET"
for target in "${PREBUILT_NO_STD_TARGETS[@]}"; do
    if rustup target list --installed | grep -Fxq "$target"; then
        check_target "$target" "prebuilt" "no" ""
    fi
done

printf '\n%sChecking source-built no_std targets...%s\n' "$BLUE" "$RESET"
for entry in "${SOURCE_NO_STD_TARGETS[@]}"; do
    IFS='|' read -r target extra_rustflags <<<"$entry"
    if target_is_known "$target"; then
        check_target "$target" "source" "no" "$extra_rustflags"
    else
        FAILURES+=("$target: target is absent from this rustc")
    fi
done

printf '\n%sChecking x86-64 compile-time feature selectors...%s\n' "$BLUE" "$RESET"
for entry in "${X86_64_FEATURE_PROFILES[@]}"; do
    IFS='|' read -r profile extra_rustflags <<<"$entry"
    run_clippy \
        "x86_64-unknown-linux-gnu: $profile (no_std)" \
        "$extra_rustflags" \
        --release --locked --lib --no-default-features --target x86_64-unknown-linux-gnu
    run_clippy \
        "x86_64-unknown-linux-gnu: $profile (std)" \
        "$extra_rustflags" \
        --release --locked --lib --no-default-features --features std --target x86_64-unknown-linux-gnu
    run_codegen "$profile: no_std" "$extra_rustflags" \
        --locked --lib --no-default-features --target x86_64-unknown-linux-gnu
    run_codegen "$profile: std" "$extra_rustflags" \
        --locked --lib --no-default-features --features std --target x86_64-unknown-linux-gnu
done

for entry in "${POWERPC64_FEATURE_PROFILES[@]}"; do
    IFS='|' read -r profile target extra_rustflags <<<"$entry"
    run_clippy \
        "$target: $profile (no_std)" \
        "$extra_rustflags" \
        --release --locked --lib --no-default-features --target "$target"
    run_codegen "$profile: no_std" "$extra_rustflags" \
        --locked --lib --no-default-features --target "$target"
done

printf '\n%sChecking ARM feature selectors...%s\n' "$BLUE" "$RESET"
for entry in "${ARM_FEATURE_PROFILES[@]}"; do
    IFS='|' read -r profile target extra_rustflags <<<"$entry"
    if rustup target list --installed | grep -Fxq "$target"; then
        run_clippy \
            "$target: $profile (no_std)" \
            "$extra_rustflags" \
            --release --locked --lib --no-default-features --target "$target"
        run_clippy \
            "$target: $profile (std)" \
            "$extra_rustflags" \
            --release --locked --lib --no-default-features --features std --target "$target"
        run_codegen "$target: $profile" "$extra_rustflags" \
            --locked --lib --no-default-features --target "$target"
    fi
done

printf '\n%sProbing targets blocked by known nightly/toolchain limitations...%s\n' "$BLUE" "$RESET"
for entry in "${TOOLCHAIN_PROBE_TARGETS[@]}"; do
    IFS='|' read -r target extra_rustflags <<<"$entry"
    log_path="$LOG_DIR/${target//\//_}.log"
    printf '\n%s=== Probing %s ===%s\n' "$CYAN" "$target" "$RESET"

    if ! target_is_known "$target"; then
        TOOLCHAIN_BLOCKS+=("$target: target is absent from this rustc")
        continue
    fi

    combined_rustflags="$extra_rustflags"
    if [[ -n "$BASE_RUSTFLAGS" && -n "$extra_rustflags" ]]; then
        combined_rustflags="$BASE_RUSTFLAGS $extra_rustflags"
    elif [[ -n "$BASE_RUSTFLAGS" ]]; then
        combined_rustflags="$BASE_RUSTFLAGS"
    fi

    printf '  -> %s\n' "$target: no features (no_std probe)"
    if env RUSTFLAGS="$combined_rustflags" cargo clippy -Z build-std=core,alloc \
        --release --locked --lib --no-default-features --target "$target" \
        -- -D warnings 2>&1 | tee "$log_path"; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        run_clippy \
            "$target: num-traits (no_std probe)" \
            "$extra_rustflags" \
            -Z build-std=core,alloc --release --locked --lib --no-default-features \
            --features num-traits --target "$target"
    elif grep -Eq 'could not compile `mp[_-]anafis`' "$log_path"; then
        FAILURES+=("$target: mp_anafis failed during toolchain probe")
    elif grep -Eq 'could not compile `(core|alloc|compiler_builtins|std|panic_abort|panic_unwind|unwind)`' "$log_path"; then
        TOOLCHAIN_BLOCKS+=("$target: rust-src library compilation failed")
    else
        FAILURES+=("$target: unclassified toolchain probe failure; inspect the diagnostic")
    fi
done

printf '\n%sCoverage summary%s\n' "$BLUE" "$RESET"
printf '  Required target configurations: %d\n' "$REQUIRED_TARGETS"
printf '  Passing clippy invocations: %d\n' "$PASSED_CHECKS"
printf '  Passing code-generation builds: %d\n' "$PASSED_BUILDS"

if ((${#TOOLCHAIN_BLOCKS[@]} > 0)); then
    printf '  %sToolchain-blocked probes: %d%s\n' "$YELLOW" "${#TOOLCHAIN_BLOCKS[@]}" "$RESET"
    for blocked in "${TOOLCHAIN_BLOCKS[@]}"; do
        printf '    - %s\n' "$blocked"
    done
fi

if ((${#FAILURES[@]} > 0)); then
    printf '  %sFailed required checks: %d%s\n' "$RED" "${#FAILURES[@]}" "$RESET"
    for failure in "${FAILURES[@]}"; do
        printf '    - %s\n' "$failure"
    done
    exit 1
fi

printf '\n%sAll required architecture checks passed.%s\n' "$GREEN" "$RESET"
