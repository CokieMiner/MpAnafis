#!/usr/bin/env bash
set -uo pipefail

# This is a bigint portability matrix, not a list of every OS/vendor spelling
# exposed by `rustc --print target-list`. Each entry covers a distinct limb
# width, endianness, ABI, instruction-set family, or target-feature selector
# used by the integer core. Tier-3 targets are compiled from rust-src because
# rustup does not distribute their core/alloc artifacts.

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

# Every entry is TARGET|EXTRA_RUSTFLAGS. Release mode avoids a rustc/LLVM
# debug-codegen crash while building compiler_builtins for m68k. The AVR CPU
# is explicit because the generic avr-none target intentionally has no default.
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

# These targets are still attempted. A failure in arbi-anafis is fatal, while
# a failure building rust-src itself is reported as an explicit toolchain block.
readonly -a TOOLCHAIN_PROBE_TARGETS=(
    "xtensa-esp32-none-elf|"
)

readonly -a X86_64_FEATURE_PROFILES=(
    "BMI2 only|-C target-feature=+bmi2,-adx"
    "ADX only|-C target-feature=+adx,-bmi2"
    "ADX and BMI2|-C target-feature=+adx,+bmi2"
)

# PowerPC64 selects its kernels at *compile* time, unlike x86-64 which detects
# at runtime, so a kernel behind a target feature is only ever built when that
# feature is requested. `power9-vector` is not a default of either PowerPC64
# target -- powerpc64le defaults to power8-vector and powerpc64 to plain
# altivec -- so without these entries the ISA 3.0 kernels are compiled by no
# check in this repository and are free to bit-rot unnoticed.
#
# Both endiannesses are listed because the P9 kernels are shared between them
# and a byte-order assumption in one would otherwise surface only downstream.
readonly -a POWERPC64_FEATURE_PROFILES=(
    "POWER9 little-endian|powerpc64le-unknown-linux-gnu|-C target-cpu=pwr9"
    "POWER9 big-endian|powerpc64-unknown-linux-gnu|-C target-cpu=pwr9"
    "POWER8 baseline|powerpc64le-unknown-linux-gnu|-C target-cpu=pwr8"
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
readonly REQUIRED_TARGETS=$((${#PREBUILT_STD_TARGETS[@]} + ${#PREBUILT_NO_STD_TARGETS[@]} + ${#SOURCE_NO_STD_TARGETS[@]}))

readonly BASE_RUSTFLAGS="${RUSTFLAGS-}"
readonly LOG_DIR="$(mktemp -d)"
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

check_target() {
    local target="$1"
    local build_kind="$2"
    local supports_std="$3"
    local extra_rustflags="$4"
    local -a build_std_args=()

    printf '\n%s=== Checking %s ===%s\n' "$CYAN" "$target" "$RESET"

    if [[ "$build_kind" == "source" ]]; then
        build_std_args=(-Z build-std=core,alloc)
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
done

for entry in "${POWERPC64_FEATURE_PROFILES[@]}"; do
    IFS='|' read -r profile target extra_rustflags <<<"$entry"
    run_clippy \
        "$target: $profile (no_std)" \
        "$extra_rustflags" \
        --release --locked --lib --no-default-features --target "$target"
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
    elif grep -Fq 'could not compile `arbi-anafis`' "$log_path"; then
        FAILURES+=("$target: arbi-anafis failed during toolchain probe")
    else
        TOOLCHAIN_BLOCKS+=("$target: rust-src/compiler backend failed before arbi-anafis")
    fi
done

printf '\n%sCoverage summary%s\n' "$BLUE" "$RESET"
printf '  Required target configurations: %d\n' "$REQUIRED_TARGETS"
printf '  Passing clippy invocations: %d\n' "$PASSED_CHECKS"

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
