"""Shared harness generation and execution for native differential tests."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

from ..asm_util import run, wsl_path
from ..regions import is_branch_instruction
from ..search.native_ast import native_named_registers, native_pointer_registers
from ..types import ArchitectureFamily

_CANDIDATE_RESULT = re.compile(r"CAND (\d+) (OK|FAIL)")
_OUTPUT_ARGUMENT = {
    ArchitectureFamily.AARCH64: "x1",
    ArchitectureFamily.ARM32: "r1",
    ArchitectureFamily.RISCV32: "x11",
    ArchitectureFamily.RISCV64: "x11",
    ArchitectureFamily.LOONGARCH32: "r5",
    ArchitectureFamily.LOONGARCH64: "r5",
    ArchitectureFamily.MIPS32: "r5",
    ArchitectureFamily.MIPS64: "r5",
    ArchitectureFamily.POWER32: "r4",
    ArchitectureFamily.POWER64: "r4",
    ArchitectureFamily.S390X: "r3",
}


def native_register_plan(
    bodies: Sequence[str],
    architecture: ArchitectureFamily,
    registers: Sequence[str],
    unavailable: set[str],
) -> Tuple[str, str, List[int], set[str]]:
    """Select harness registers and locate pointer-bearing kernel registers."""
    require_candidates(bodies)
    used = set().union(
        *(native_named_registers(body, architecture) for body in bodies),
    )
    spare = [
        register
        for register in reversed(registers)
        if register not in used and register not in unavailable
    ]
    if len(spare) < 2:
        raise ValueError(
            f"{architecture.value} differential ABI requires two unused GPRs",
        )
    # Wrappers copy input first, then output. The first destination must not
    # destroy the still-live output argument; the other spare is always distinct.
    if spare[0] == _OUTPUT_ARGUMENT.get(architecture):
        spare[0], spare[1] = spare[1], spare[0]
    pointers = set().union(
        *(native_pointer_registers(body, architecture) for body in bodies),
    )
    pointer_slots = sorted(
        registers.index(register)
        for register in pointers
        if register in registers
    )
    return spare[0], spare[1], pointer_slots, used


def prepare_body_labels(body: str, symbol: str) -> tuple[str, List[str]]:
    """Rename local labels and close branch exits at the wrapper epilogue."""
    defined = {
        match.group(1)
        for line in body.splitlines()
        if (match := re.match(r"^\s*([.$A-Za-z_][\w.$]*):\s*$", line)) is not None
    }
    targets = {
        target
        for line in body.splitlines()
        if is_branch_instruction(line)
        if (target := _branch_target(line)) is not None
    }
    external = targets - defined
    replacements = {
        label: f".L{symbol}_{'exit' if label in external else 'local'}_{label.lstrip('.')}"
        for label in defined | external
    }
    for label in sorted(replacements, key=len, reverse=True):
        body = re.sub(
            rf"(?<![\w.$]){re.escape(label)}(?![\w.$])",
            replacements[label],
            body,
        )
    return body, sorted(replacements[label] for label in external)


def driver_source(
    kernels: int,
    register_slots: int,
    output_slots: int,
    reserve_slots: Sequence[int],
    pointer_slots: Sequence[int],
    cases: int,
    word_bits: int,
) -> str:
    """Generate the deterministic Rust oracle driver for one native word size."""
    rust_type = f"u{word_bits}"
    word_bytes = word_bits // 8
    multiplier = 6364136223846793005 if word_bits == 64 else 1664525
    increment = 1442695040888963407 if word_bits == 64 else 1013904223
    seed = "0x9E3779B97F4A7C15" if word_bits == 64 else "0x9E3779B9"
    declarations = "\n".join(
        f"    fn k_{index}(p: *const {rust_type}, o: *mut {rust_type});"
        for index in range(kernels)
    )
    functions = ", ".join(f"k_{index}" for index in range(kernels))
    reserve = ", ".join(str(index) for index in reserve_slots)
    pointers = ", ".join(str(index) for index in pointer_slots)
    return f"""\
extern "C" {{
{declarations}
}}

static KERNELS: [
    unsafe extern "C" fn(*const {rust_type}, *mut {rust_type}); {kernels}
] = [{functions}];
const N_SLOTS: usize = {register_slots};
const OUT_SLOTS: usize = {output_slots};
const RESERVE: [usize; {len(reserve_slots)}] = [{reserve}];
const PTR_SLOTS: &[usize] = &[{pointers}];
const BUFLEN: usize = 2048;
const BASES: [usize; 3] = [512, 1024, 1536];

fn next(state: &mut {rust_type}) -> {rust_type} {{
    *state = state.wrapping_mul({multiplier}).wrapping_add({increment});
    *state
}}

fn run_one(
    function: unsafe extern "C" fn(*const {rust_type}, *mut {rust_type}),
    seed: {rust_type},
    case: {rust_type},
) -> ([{rust_type}; OUT_SLOTS], [{rust_type}; BUFLEN]) {{
    let mut buffer = [0; BUFLEN];
    let mut input = [0; N_SLOTS];
    let mut state = seed;
    for value in &mut buffer {{ *value = next(&mut state); }}
    let alias = case % 2 == 0;
    let base = buffer.as_mut_ptr() as usize;
    for (stream, &register) in PTR_SLOTS.iter().enumerate() {{
        let selected = if alias {{ 0 }} else {{ stream % 3 }};
        input[register] = (base + BASES[selected] * {word_bytes}) as {rust_type};
    }}
    for register in 0..N_SLOTS {{
        if RESERVE.contains(&register) || PTR_SLOTS.contains(&register) {{ continue; }}
        input[register] = next(&mut state) & 0xff;
    }}
    let mut output = [0; OUT_SLOTS];
    unsafe {{ function(input.as_ptr(), output.as_mut_ptr()); }}
    (output, buffer)
}}

fn main() {{
    let args: Vec<String> = std::env::args().collect();
    let cases: {rust_type} = args.get(1).and_then(|value| value.parse().ok()).unwrap_or({cases});
    let mut state: {rust_type} = {seed};
    for index in 1..KERNELS.len() {{
        let mut valid = true;
        for case in 0..cases {{
            let seed = next(&mut state);
            let (original_output, original_buffer) = run_one(KERNELS[0], seed, case);
            let (candidate_output, candidate_buffer) = run_one(KERNELS[index], seed, case);
            if original_output != candidate_output || original_buffer != candidate_buffer {{
                valid = false;
                break;
            }}
        }}
        println!("CAND {{}} {{}}", index, if valid {{ "OK" }} else {{ "FAIL" }});
    }}
}}
"""


def compile_and_run(
    wrappers: str,
    driver: str,
    candidate_count: int,
    cases: int,
    use_wsl: bool,
) -> List[bool]:
    """Assemble, link, execute, and decode one native differential harness."""
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        (work / "kernels.s").write_text(wrappers, encoding="utf-8")
        (work / "driver.rs").write_text(driver, encoding="utf-8")
        assembled = run(
            ["as", wsl_path(work / "kernels.s"), "-o", wsl_path(work / "kernels.o")],
            use_wsl,
        )
        if assembled.returncode != 0:
            raise RuntimeError(
                "as failed: " + (assembled.stderr or assembled.stdout)[:2000],
            )
        compiled = run(
            [
                "rustc", "--edition=2021", "-C", "opt-level=0",
                wsl_path(work / "driver.rs"),
                "-C", "link-arg=" + wsl_path(work / "kernels.o"),
                "-o", wsl_path(work / "diff_test"),
            ],
            use_wsl,
        )
        if compiled.returncode != 0:
            raise RuntimeError(
                "rustc failed: " + (compiled.stderr or compiled.stdout)[:2000],
            )
        executed = run(
            [wsl_path(work / "diff_test"), str(cases)],
            use_wsl,
        )
        if executed.returncode != 0:
            raise RuntimeError(
                f"diff_test run failed with exit code {executed.returncode}: "
                + (executed.stderr or executed.stdout)[:2000],
            )
    valid = [False] * candidate_count
    valid[0] = True
    seen: set[int] = set()
    for line in executed.stdout.splitlines():
        match = _CANDIDATE_RESULT.fullmatch(line.strip())
        if match:
            index = int(match.group(1))
            if not 0 < index < candidate_count or index in seen:
                raise RuntimeError("invalid or duplicate differential candidate result")
            seen.add(index)
            valid[index] = match.group(2) == "OK"
    if seen != set(range(1, candidate_count)):
        raise RuntimeError("incomplete differential candidate results")
    return valid


def require_candidates(bodies: Sequence[str]) -> None:
    """Reject a differential run without an original and a candidate."""
    if len(bodies) < 2:
        raise ValueError("need at least 2 bodies (original + candidate)")


def _branch_target(line: str) -> str | None:
    operands = line.split(None, 1)
    if len(operands) != 2:
        return None
    token = operands[1].rsplit(",", 1)[-1].strip().split()[-1].lstrip("*")
    return token if re.fullmatch(r"[.$A-Za-z_][\w.$]*", token) else None


__all__ = [
    "compile_and_run",
    "driver_source",
    "native_register_plan",
    "prepare_body_labels",
    "require_candidates",
]
