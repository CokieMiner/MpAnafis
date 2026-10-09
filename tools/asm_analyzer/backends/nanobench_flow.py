"""Conservative control-flow contracts for executable nanoBench snippets.

Only local forward branches and bounded, single-entry countdown loops are
accepted. This validates termination patterns; it does not prove memory bounds.
"""

from __future__ import annotations

import re
from typing import Optional, Set

from ..asm_util import GPR_ALIAS_MAP, split_asm_operands
from ..search.ast import get_instruction_spec, parse_line

LOOP_MNEMONICS = ("loop", "loope", "loopne", "loopnz", "loopz")
MAX_LOOP_COUNT = 4096


def validate_measurable(asm_code: str, pointers: Set[str], scalars: Set[str]) -> Optional[str]:
    """Return a refusal reason unless every branch has a bounded local target."""
    for line in asm_code.splitlines():
        code = line.split("#", 1)[0].strip()
        if code.startswith(".") and not code.endswith(":"):
            if code.split(None, 1)[0] not in (".text", ".p2align", ".align"):
                return "encoding and assembler-state directives require manual measurement"
    parsed, resolve_target = asm_flow(asm_code)
    for index, (mnemonic, operands) in enumerate(parsed):
        if mnemonic.startswith(("call", "ret", "sys", "int", "iret")):
            return f"external control transfer `{mnemonic}` is not measurable as a snippet"
        if not (mnemonic.startswith("j") or mnemonic in LOOP_MNEMONICS):
            continue
        if not operands:
            return "branch has no local target"
        target = resolve_target(operands[-1], index)
        if target is None:
            return f"branch target `{operands[-1]}` is external or indirect"
        if target > index:
            continue
        if mnemonic in LOOP_MNEMONICS:
            counter = "rcx"
            decrement = index
        else:
            if mnemonic not in ("jnz", "jne", "jns"):
                return f"backward branch `{mnemonic}` has no supported countdown proof"
            preceding = [i for i in range(target, index) if parsed[i][0]]
            if not preceding:
                return "backward branch has no countdown instruction"
            decrement = preceding[-1]
            operation, arguments = parsed[decrement]
            if operation not in ("dec", "decq", "decl") or len(arguments) != 1:
                return "backward branch requires an immediately preceding register decrement"
            counter = canonical_reg(arguments[0])
            if counter is None:
                return "backward branch requires a register counter"
        if counter in pointers or counter not in scalars:
            return f"counter %{counter} is outside the seeded scalar set"
        for inner_index in range(target, index):
            operation, arguments = parsed[inner_index]
            if not operation or inner_index == decrement:
                continue
            if operation.startswith("j") or operation in LOOP_MNEMONICS:
                return "nested or conditional loop bodies require manual measurement"
            instruction = parse_line(operation + " " + ", ".join(arguments))
            spec = get_instruction_spec(instruction)
            if spec.unknown or counter in spec.defs:
                return f"counter %{counter} is reset or modified inside its loop"
        refusal = _validate_initial_count(parsed, target, counter, scalars)
        if refusal:
            return refusal
    return None


def _validate_initial_count(parsed, target: int, counter: str, scalars: Set[str]) -> Optional[str]:
    """Require a positive bounded live-in or an unambiguous preheader assignment."""
    if any(mnemonic.startswith("j") or mnemonic in LOOP_MNEMONICS for mnemonic, _ in parsed[:target]):
        return "counter initialization across branches requires manual measurement"
    for index in range(target - 1, -1, -1):
        mnemonic, operands = parsed[index]
        if not mnemonic:
            continue
        if mnemonic.startswith("j") or mnemonic in LOOP_MNEMONICS:
            return "counter initialization across branches requires manual measurement"
        instruction = parse_line(mnemonic + " " + ", ".join(operands))
        spec = get_instruction_spec(instruction)
        if spec.unknown:
            return "unknown instruction before loop counter initialization"
        if counter not in spec.defs:
            continue
        if mnemonic not in ("mov", "movq", "movl") or len(operands) != 2:
            return f"counter %{counter} has no bounded preheader assignment"
        source = operands[0]
        if re.fullmatch(r"\$[-+]?(?:0x[\da-fA-F]+|\d+)", source):
            value = int(source[1:], 16 if "x" in source.lower() else 10)
            if 0 < value <= MAX_LOOP_COUNT:
                return None
            return f"counter %{counter} must start between 1 and {MAX_LOOP_COUNT}"
        origin = canonical_reg(source)
        if origin in scalars:
            for previous, arguments in parsed[:index]:
                if not previous:
                    continue
                prior = get_instruction_spec(parse_line(previous + " " + ", ".join(arguments)))
                if prior.unknown or origin in prior.defs:
                    return f"counter %{counter} derives from modified scalar %{origin}"
            return None
        return f"counter %{counter} derives from `{source}`, outside the seeded scalar set"
    return None  # The launcher seeds scalar live-ins with a small positive value.


def asm_flow(asm_code: str):
    """Parse instructions and resolve repeated numeric labels by branch position."""
    lines = [line.split("#", 1)[0].strip() for line in asm_code.splitlines()]
    parsed = [split_asm_line(line) for line in lines]
    labels: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        if re.fullmatch(r"(?:[.$A-Za-z_][\w.$]*|\d+):", line):
            labels.setdefault(line[:-1], []).append(index)

    def resolve_target(target: str, position: int) -> Optional[int]:
        if re.fullmatch(r"\d+[bf]", target):
            positions = labels.get(target[:-1], [])
            if target.endswith("b"):
                return max((value for value in positions if value <= position), default=None)
            return min((value for value in positions if value > position), default=None)
        positions = labels.get(target, [])
        return positions[0] if len(positions) == 1 else None

    return parsed, resolve_target


def split_asm_line(line: str) -> tuple[str, list[str]]:
    """Decode AT&T operands while preserving complete address tuples."""
    code = line.split("#", 1)[0].split("//", 1)[0].strip()
    if not code or code.startswith(".") or code.endswith(":"):
        return "", []
    parts = code.split(None, 1)
    return parts[0].lower(), split_asm_operands(parts[1]) if len(parts) > 1 else []


def canonical_reg(operand: str) -> Optional[str]:
    """Normalize one explicit x86 GPR operand."""
    match = re.fullmatch(r"%([a-z][a-z0-9]*)", operand.strip())
    return GPR_ALIAS_MAP.get(match.group(1)) if match else None
