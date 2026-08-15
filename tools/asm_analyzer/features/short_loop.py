"""Short-operand finite loop iteration predictor.

Models total latency for small BigInt operations (1, 2, 3, 4, 8 limbs) by
combining:
1. Prologue instruction latency (setup, argument loads, count initialization).
2. N x Loop body latency (arithmetic, unrolling factors, multiplier stalls).
3. Loop exit branch misprediction penalty (14–19 cycles on modern out-of-order cores).
4. Epilogue instruction latency (carry normalization, store, return).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from ..asm_util import extract_mnemonic, instr_lines

# Average branch misprediction penalty on modern OoO cores (Zen 3/4, Golden Cove, Firestorm)
BRANCH_MISPREDICT_PENALTY_CYCLES = 16.0

_LABEL_RE = re.compile(r"^\s*([0-9a-zA-Z_.]+):\s*$")
_LOOP_BRANCH_RE = re.compile(r"^(?:jmp|jz|jnz|jne|je|js|jns|b|b\.[a-z]+|cbnz|cbz)\b", re.IGNORECASE)


def analyze_short_loops(
    asm: str,
    loop_body_cycles_per_iter: float = 1.5,
    unroll_factor: int = 1,
) -> Dict[str, object]:
    """Predict execution cost for short operand lengths (1, 2, 3, 4, 8 limbs)."""
    raw_lines = asm.splitlines()
    cleaned = instr_lines(asm)

    # Segment assembly into prologue, loop body, and epilogue
    prologue_instrs = 0
    loop_instrs = 0
    epilogue_instrs = 0

    in_prologue = True
    in_loop = False
    in_epilogue = False

    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            continue

        m_label = _LABEL_RE.match(s)
        if m_label:
            label_name = m_label.group(1)
            if label_name in ("1", "1b", ".Lloop", "loop"):
                in_prologue = False
                in_loop = True
                continue
            elif label_name in ("2", "2f", ".Ldone", "done", "exit"):
                in_loop = False
                in_epilogue = True
                continue

        # Count instructions
        mnem = extract_mnemonic(s)
        if not mnem or s.startswith("."):
            continue

        if in_prologue:
            prologue_instrs += 1
        elif in_loop:
            loop_instrs += 1
        elif in_epilogue:
            epilogue_instrs += 1

    # If no explicit loop label was found, estimate proportionally
    if loop_instrs == 0:
        loop_instrs = len(cleaned)
        prologue_instrs = 1
        epilogue_instrs = 1

    # Estimate prologue and epilogue cycles (~0.5 cycles per fused ALU op)
    prologue_cycles = max(prologue_instrs * 0.4, 0.5)
    epilogue_cycles = max(epilogue_instrs * 0.4, 0.5)
    effective_body_cycles = max(loop_body_cycles_per_iter, loop_instrs * 0.25)

    # Cost model for N limbs:
    # N=1: Prologue + 1 body iteration (often early-skipped) + epilogue (low mispredict)
    # N=2..8: Prologue + ceil(N / unroll) * body + epilogue + branch_mispredict
    predictions: Dict[int, float] = {}
    for n in (1, 2, 3, 4, 8, 16):
        iters = (n + unroll_factor - 1) // unroll_factor
        if n == 1:
            # Often handles N=1 through early skip or single fall-through with 50% mispredict
            cost = prologue_cycles + effective_body_cycles + epilogue_cycles + (0.3 * BRANCH_MISPREDICT_PENALTY_CYCLES)
        else:
            # Loop runs iters times, then takes terminal branch misprediction
            cost = prologue_cycles + (iters * effective_body_cycles) + epilogue_cycles + BRANCH_MISPREDICT_PENALTY_CYCLES
        predictions[n] = round(cost, 2)

    return {
        "prologue_instructions": prologue_instrs,
        "loop_body_instructions": loop_instrs,
        "epilogue_instructions": epilogue_instrs,
        "estimated_cycles_by_limbs": predictions,
        "branch_mispredict_overhead_cycles": BRANCH_MISPREDICT_PENALTY_CYCLES,
    }
