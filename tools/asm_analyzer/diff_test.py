#!/usr/bin/env python3
"""Differential tests for kernel-search candidates (compiled, not simulated)."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import List, Tuple

from .asm_util import GPR64, classify_regs, named_registers, run, wsl_path

_CAND_RE = re.compile(r"CAND (\d+) (OK|FAIL)")


def _pick_reserve(bodies: List[str]) -> Tuple[str, str]:
    used: set[str] = {"rax", "rdx"}
    for b in bodies:
        used |= named_registers(b)
    used.discard("rsp")
    spare = [r for r in GPR64 if r not in used]
    if len(spare) < 2:
        raise ValueError(
            f"kernel uses {len(used)} of {len(GPR64)} GPRs; the differential "
            "ABI needs at least 2 spare registers"
        )
    return spare[0], spare[1]


def _asm_wrapper(symbol: str, body: str, r_in: str, r_out: str) -> str:
    lines = [f".globl {symbol}", f"{symbol}:"]
    for r in ("rbp", "rbx", "r12", "r13", "r14", "r15"):
        lines.append(f"    push %{r}")
    lines.append(f"    mov %rdi, %{r_in}")
    lines.append(f"    mov %rsi, %{r_out}")
    for idx, reg in enumerate(GPR64):
        if reg in (r_in, r_out):
            continue
        lines.append(f"    mov {idx * 8}(%{r_in}), %{reg}")
    lines.append("    clc")
    for ln in body.splitlines():
        lines.append(("    " + ln) if ln.strip() else "")
    for idx, reg in enumerate(GPR64):
        lines.append(f"    mov %{reg}, {idx * 8}(%{r_out})")
    lines.append(f"    movq $0, 120(%{r_out})")
    for r in ("r15", "r14", "r13", "r12", "rbx", "rbp"):
        lines.append(f"    pop %{r}")
    lines.append("    ret")
    return "\n".join(lines) + "\n"


def _driver(n: int, r_in: str, r_out: str, ptr_slots: List[int], cases: int) -> str:
    decls = "\n".join(f"    fn k_{i}(p: *const u64, o: *mut u64);" for i in range(n))
    arr = ", ".join(f"k_{i}" for i in range(n))
    reserve_idx = [GPR64.index(r_in), GPR64.index(r_out)]
    ptr = ", ".join(str(i) for i in ptr_slots)
    res = ", ".join(str(i) for i in reserve_idx)
    return f"""\
#![allow(dead_code)]
extern "C" {{
{decls}
}}

static KERNELS: [unsafe extern "C" fn(*const u64, *mut u64); {n}] = [{arr}];

const N_SLOTS: usize = 15;
const RESERVE: [usize; 2] = [{res}];
const PTR_SLOTS: &[usize] = &[{ptr}];
const BUFLEN: usize = 2048;
const BASES: [usize; 3] = [0, 256, 512];

fn nxt(s: &mut u64) -> u64 {{
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}}

fn run_one(f: unsafe extern "C" fn(*const u64, *mut u64), seed: u64, case: u64)
    -> ([u64; 16], [u64; BUFLEN]) {{
    let mut buf = [0u64; BUFLEN];
    let mut inp = [0u64; N_SLOTS];
    let mut s = seed;
    for i in 0..BUFLEN {{
        buf[i] = nxt(&mut s);
    }}
    let alias = case % 2 == 0;
    let base = buf.as_mut_ptr() as usize;
    for (k, &r) in PTR_SLOTS.iter().enumerate() {{
        let b = if alias {{ 0 }} else {{ k % 3 }};
        inp[r] = (base + BASES[b] * 8) as u64;
    }}
    for r in 0..N_SLOTS {{
        if RESERVE.contains(&r) || PTR_SLOTS.contains(&r) {{ continue; }}
        inp[r] = nxt(&mut s) & 0xff;
    }}
    let mut out = [0u64; 16];
    unsafe {{ f(inp.as_ptr(), out.as_mut_ptr()); }}
    (out, buf)
}}

fn main() {{
    let args: Vec<String> = std::env::args().collect();
    let cases: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or({cases});
    let mut s: u64 = args.get(2).and_then(|s| s.parse().ok())
        .unwrap_or(0x9E3779B97F4A7C15);
    for i in 1..KERNELS.len() {{
        let mut ok = true;
        for c in 0..cases {{
            let se = nxt(&mut s);
            let (o0, b0) = run_one(KERNELS[0], se, c);
            let (o1, b1) = run_one(KERNELS[i], se, c);
            if o0 != o1 || b0 != b1 {{ ok = false; break; }}
        }}
        println!("CAND {{}} {{}}", i, if ok {{ "OK" }} else {{ "FAIL" }});
    }}
}}
"""


def diff_test_variants(bodies: List[str], cases: int = 150, use_wsl: bool = False) -> List[bool]:
    """Compile bodies[0] and every candidate, and differentially test them."""
    if len(bodies) < 2:
        raise ValueError("need at least 2 bodies (original + candidate)")
    r_in, r_out = _pick_reserve(bodies)
    pointers, _ = classify_regs("\n".join(bodies))
    ptr_slots = sorted(GPR64.index(r) for r in pointers if r in GPR64)
    n = len(bodies)

    with tempfile.TemporaryDirectory() as tmp_str:
        work = Path(tmp_str)
        (work / "kernels.s").write_text(
            "".join(_asm_wrapper(f"k_{i}", b, r_in, r_out) for i, b in enumerate(bodies)),
            encoding="utf-8",
        )
        (work / "driver.rs").write_text(
            _driver(n, r_in, r_out, ptr_slots, cases), encoding="utf-8"
        )

        as_cmd = ["as", wsl_path(work / "kernels.s"), "-o", wsl_path(work / "kernels.o")]
        ar = run(as_cmd, use_wsl)
        if ar.returncode != 0:
            raise RuntimeError("as failed: " + (ar.stderr or ar.stdout)[:400])

        rc_cmd = [
            "rustc", "--edition=2021", "-C", "opt-level=0",
            wsl_path(work / "driver.rs"),
            "-C", "link-arg=" + wsl_path(work / "kernels.o"),
            "-o", wsl_path(work / "diff_test"),
        ]
        rr = run(rc_cmd, use_wsl)
        if rr.returncode != 0:
            raise RuntimeError("rustc failed: " + (rr.stderr or rr.stdout)[:400])

        run_cmd = [wsl_path(work / "diff_test"), str(cases), "0x9E3779B97F4A7C15"]
        r = run(run_cmd, use_wsl)
        if r.returncode != 0:
            raise RuntimeError("diff_test run failed: " + (r.stderr or r.stdout)[:400])

        ok = [True] * n
        for line in r.stdout.splitlines():
            m = _CAND_RE.match(line.strip())
            if m:
                ok[int(m.group(1))] = (m.group(2) == "OK")
        return ok
