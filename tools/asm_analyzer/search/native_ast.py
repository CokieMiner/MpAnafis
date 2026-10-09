"""Strict dependency parsing for non-x86 integer kernel instructions."""

from __future__ import annotations

import re
from typing import Dict, List, Set

from ..types import ArchitectureFamily
from .ast import FLAG_FULL, Instr, Op, Spec

_RISCV_ALIASES = {
    "zero": "x0", "ra": "x1", "sp": "x2", "gp": "x3", "tp": "x4",
    "t0": "x5", "t1": "x6", "t2": "x7", "s0": "x8", "fp": "x8",
    "s1": "x9", "a0": "x10", "a1": "x11", "a2": "x12", "a3": "x13",
    "a4": "x14", "a5": "x15", "a6": "x16", "a7": "x17",
    **{f"s{index}": f"x{index + 16}" for index in range(2, 12)},
    "t3": "x28", "t4": "x29", "t5": "x30", "t6": "x31",
    **{f"x{index}": f"x{index}" for index in range(32)},
}
_LOONGARCH_ALIASES = {
    "zero": "r0", "ra": "r1", "tp": "r2", "sp": "r3",
    **{f"a{index}": f"r{index + 4}" for index in range(8)},
    **{f"t{index}": f"r{index + 12}" for index in range(9)},
    "fp": "r22",
    **{f"s{index}": f"r{index + 23}" for index in range(9)},
    **{f"r{index}": f"r{index}" for index in range(32)},
}
_LOAD_PREFIXES = (
    "lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "ldr", "ldp",
    "ldur", "lg", "llg", "lgrl",
)
_STORE_PREFIXES = (
    "sb", "sh", "sw", "sd", "st",
)
_AARCH64_COMPARE = {"cmp", "cmn", "tst"}
_AARCH64_FLAG_READ = {
    "adc", "adcs", "sbc", "sbcs", "csel", "csinc", "csinv", "csneg",
    "cset", "csetm",
}
_AARCH64_FLAG_WRITE = {"adds", "adcs", "subs", "sbcs", *_AARCH64_COMPARE}
_POWER_CARRY_READ = {"adde", "addze", "subfe", "subfze"}
_POWER_CARRY_WRITE = {"addc", "adde", "addze", "subfc", "subfe", "subfze"}
_RECOGNIZED_ARITHMETIC = (
    "add", "adds", "adc", "adcs", "sub", "subs", "sbc", "sbcs", "mul",
    "madd", "msub", "umulh", "smulh", "umaal",
    "and", "orr", "or", "xor", "eor", "bic", "not", "neg", "mov",
    "lsl", "lsr", "asr", "sll", "srl", "sra", "slt", "sltu", "clz",
    "ctz", "rev", "rbit", "extr", "ubfx", "bfi", "sel", "addi", "addiu",
    "daddu", "daddiu", "addu", "subu", "mulld", "mulhdu", "lgr", "algr",
    "alcgr", "slgr", "slbgr", "aghi", "la", "li", "mr", "mflo", "mfhi",
    "dmultu", "multu",
)


def parse_native_line(
    line: str,
    architecture: ArchitectureFamily,
) -> Instr | None:
    """Parse one non-x86 instruction while preserving unknowns as barriers."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", ".")) or stripped.endswith(":"):
        return None
    parts = stripped.split(None, 1)
    mnemonic = parts[0].lower()
    tokens = _split_operands(parts[1] if len(parts) == 2 else "")
    operands = [
        _parse_operand(token, architecture, mnemonic, index, len(tokens))
        for index, token in enumerate(tokens)
    ]
    if architecture in (ArchitectureFamily.AARCH64, ArchitectureFamily.ARM32):
        register_width = 32 if architecture == ArchitectureFamily.ARM32 else 64
        if tokens and tokens[0].startswith(("q", "v")):
            register_width = 128
        if tokens and tokens[0].startswith("w"):
            register_width = 32
        if mnemonic.endswith("b"):
            register_width = 8
        elif mnemonic.endswith("h"):
            register_width = 16
        elif mnemonic.endswith("sw"):
            register_width = 32
        for operand in operands:
            if operand.kind == "mem":
                operand.width = register_width * (2 if mnemonic in ("ldp", "ldpsw", "stp", "ldrd", "strd") else 1)
    return Instr(line=stripped, mnemonic=mnemonic, ops=operands)


def native_instruction_spec(
    instruction: Instr,
    architecture: ArchitectureFamily,
) -> Spec:
    """Return conservative uses, definitions, flags, and memory effects."""
    mnemonic = instruction.mnemonic
    operands = instruction.ops
    spec = Spec()
    memory = next((operand for operand in operands if operand.kind == "mem"), None)
    implicit_loongarch_address = architecture in (
        ArchitectureFamily.LOONGARCH32,
        ArchitectureFamily.LOONGARCH64,
    ) and len(operands) >= 2
    is_load = mnemonic.startswith(_LOAD_PREFIXES) and (
        memory is not None or implicit_loongarch_address
    )
    is_store = mnemonic.startswith(_STORE_PREFIXES) and (
        memory is not None or implicit_loongarch_address
    )

    if is_load:
        spec.mem = "load"
        destination_count = 2 if mnemonic.startswith("ldp") else 1
        for operand in operands[:destination_count]:
            _define_register(spec, operand)
        for operand in operands[destination_count:]:
            spec.uses.update(operand.regs)
    elif is_store:
        spec.mem = "store"
        value_count = 2 if mnemonic.startswith("stp") else 1
        for operand in operands[:value_count]:
            spec.uses.update(operand.regs)
        for operand in operands[value_count:]:
            spec.uses.update(operand.regs)
    elif mnemonic in ("mflo", "mfhi"):
        _define_register(spec, operands[0] if operands else None)
        spec.uses.add("lo" if mnemonic == "mflo" else "hi")
    elif mnemonic in ("dmultu", "multu"):
        for operand in operands:
            spec.uses.update(operand.regs)
        spec.defs.update(("hi", "lo"))
    elif mnemonic == "umaal" and len(operands) >= 4:
        for operand in operands[:2]:
            spec.uses.update(operand.regs)
            _define_register(spec, operand)
        for operand in operands[2:]:
            spec.uses.update(operand.regs)
    elif mnemonic in {"umull", "smull", "umlal", "smlal"} and len(operands) >= 4:
        for operand in operands[:2]:
            if mnemonic.endswith("lal"):
                spec.uses.update(operand.regs)
            _define_register(spec, operand)
        for operand in operands[2:]:
            spec.uses.update(operand.regs)
    elif mnemonic in _AARCH64_COMPARE:
        for operand in operands:
            spec.uses.update(operand.regs)
    elif _is_recognized_arithmetic(mnemonic):
        if operands:
            _define_register(spec, operands[0])
            for operand in operands[1:]:
                spec.uses.update(operand.regs)
            if _destination_is_read(mnemonic, architecture):
                spec.uses.update(operands[0].regs)
            if architecture == ArchitectureFamily.ARM32 and len(operands) == 2 and mnemonic != "mov":
                spec.uses.update(operands[0].regs)
    else:
        for operand in operands:
            spec.uses.update(operand.regs)
        spec.unknown = True

    if memory is not None:
        spec.uses.update(memory.regs)
        if _has_writeback(instruction):
            for register in memory.regs[:1]:
                spec.defs.add(register)

    if mnemonic in _AARCH64_FLAG_READ or mnemonic in _POWER_CARRY_READ:
        spec.flags_read.add(FLAG_FULL)
    if mnemonic in _AARCH64_FLAG_WRITE or mnemonic in _POWER_CARRY_WRITE:
        spec.flags_write.add(FLAG_FULL)
    if architecture == ArchitectureFamily.S390X and mnemonic in {
        "algr", "alcgr", "slgr", "slbgr", "aghi",
    }:
        spec.flags_write.add(FLAG_FULL)
        if mnemonic in {"alcgr", "slbgr"}:
            spec.flags_read.add(FLAG_FULL)
    if architecture in (ArchitectureFamily.POWER32, ArchitectureFamily.POWER64) and mnemonic.endswith("."):
        spec.flags_write.add(FLAG_FULL)
    if architecture == ArchitectureFamily.AARCH64 and (is_load or is_store):
        if mnemonic not in {
            "ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw", "ldp", "ldpsw",
            "ldur", "ldurb", "ldurh", "ldursb", "ldursh", "ldursw",
            "str", "strb", "strh", "stp", "stur", "sturb", "sturh",
        }:
            spec.unknown = True  # Exclusive/atomic forms have additional effects.
    if architecture in (ArchitectureFamily.MIPS32, ArchitectureFamily.MIPS64) and mnemonic in ("madd", "msub", "mul"):
        spec.unknown = True  # These encodings also affect the HI/LO accumulator.
    return spec


def native_named_registers(
    body: str,
    architecture: ArchitectureFamily,
) -> set[str]:
    """Return every explicit register named by a non-x86 assembly body."""
    registers: set[str] = set()
    for line in body.splitlines():
        instruction = parse_native_line(line, architecture)
        if instruction is not None:
            for operand in instruction.ops:
                registers.update(operand.regs)
    return registers


def native_pointer_registers(
    body: str,
    architecture: ArchitectureFamily,
) -> set[str]:
    """Return explicit memory-base registers from a non-x86 body."""
    registers: set[str] = set()
    for line in body.splitlines():
        instruction = parse_native_line(line, architecture)
        if instruction is None:
            continue
        for operand in instruction.ops:
            if operand.kind == "mem" and operand.base is not None:
                registers.add(operand.base)
        if (
            architecture in (
                ArchitectureFamily.LOONGARCH32,
                ArchitectureFamily.LOONGARCH64,
            )
            and instruction.mnemonic.startswith((*_LOAD_PREFIXES, *_STORE_PREFIXES))
            and len(instruction.ops) >= 2
        ):
            registers.update(instruction.ops[1].regs)
    return registers


def _split_operands(text: str) -> List[str]:
    operands: List[str] = []
    current: List[str] = []
    depth = 0
    for character in text:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        if character == "," and depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if current:
        operands.append("".join(current).strip())
    return [operand for operand in operands if operand]


def _parse_operand(
    text: str,
    architecture: ArchitectureFamily,
    mnemonic: str,
    index: int,
    count: int,
) -> Op:
    token = text.strip()
    if "[" in token or "(" in token:
        registers = _registers_in(token, architecture)
        displacement = _displacement(token)
        return Op(
            text=token,
            kind="mem",
            base=registers[0] if registers else None,
            regs=registers,
            addr=(token if len(registers) == 1 and not re.search(r"(?<!\w)0\d+", token) and re.fullmatch(
                r"(?:\[\w+(?:,\s*#?[-+]?(?:0x[\da-fA-F]+|\d+))?\]|[-+]?(?:0x[\da-fA-F]+|\d+)?\([%$]?\w+\))",
                token,
            ) else None),
            displacement=displacement,
            width=_memory_width(mnemonic),
        )
    register = _register_name(token, architecture, mnemonic, index, count)
    if register is not None:
        return Op(token, "reg", register, [register], None)
    if token.startswith(("#", "$")) or re.fullmatch(r"[-+]?(?:0x[\da-fA-F]+|\d+)", token):
        return Op(token, "imm", None, [], None)
    return Op(token, "other", None, _registers_in(token, architecture), None)


def _registers_in(text: str, architecture: ArchitectureFamily) -> List[str]:
    region = text
    if "[" in text and "]" in text:
        region = text.split("[", 1)[1].rsplit("]", 1)[0]
    elif "(" in text and ")" in text:
        region = text.split("(", 1)[1].rsplit(")", 1)[0]
    candidates = re.findall(r"[%$]?[A-Za-z][A-Za-z\d]*|[%$]?\d+", region)
    registers: List[str] = []
    for index, candidate in enumerate(candidates):
        register = _register_name(candidate, architecture, "", index, len(candidates))
        if register is not None and register not in registers:
            registers.append(register)
    return registers


def _register_name(
    token: str,
    architecture: ArchitectureFamily,
    mnemonic: str,
    index: int,
    count: int,
) -> str | None:
    name = token.strip().lower().lstrip("%$")
    if architecture == ArchitectureFamily.AARCH64:
        if match := re.fullmatch(r"[xw](\d+)", name):
            return f"x{match.group(1)}"
        if match := re.fullmatch(r"[vqdshb](\d+)(?:\..*)?", name):
            return f"v{match.group(1)}"
        if name in {"xzr", "wzr"}:
            return "zr"
        return name if name == "sp" else None
    if architecture == ArchitectureFamily.ARM32:
        if re.fullmatch(r"r(?:1[0-5]|\d)", name):
            return name
        return {"sp": "r13", "lr": "r14", "pc": "r15"}.get(name)
    if architecture in (ArchitectureFamily.RISCV32, ArchitectureFamily.RISCV64):
        return _RISCV_ALIASES.get(name)
    if architecture in (ArchitectureFamily.POWER32, ArchitectureFamily.POWER64):
        if name.isdigit() and not (_power_immediate(mnemonic, index, count)):
            return f"r{name}"
        return name if re.fullmatch(r"r\d+", name) else None
    if architecture == ArchitectureFamily.S390X:
        return name if re.fullmatch(r"r\d+", name) else None
    if architecture in (ArchitectureFamily.MIPS32, ArchitectureFamily.MIPS64):
        return f"r{name}" if name.isdigit() else None
    if architecture in (ArchitectureFamily.LOONGARCH32, ArchitectureFamily.LOONGARCH64):
        return _LOONGARCH_ALIASES.get(name)
    return None


def _power_immediate(mnemonic: str, index: int, count: int) -> bool:
    return index == count - 1 and mnemonic.startswith(("addi", "addis", "cmpi", "sldi", "srdi"))


def _displacement(text: str) -> int:
    match = re.search(r"(?:^|[\[,])\s*#?([-+]?(?:0x[\da-fA-F]+|\d+))", text)
    if match is None:
        match = re.match(r"\s*([-+]?(?:0x[\da-fA-F]+|\d+))\(", text)
    try:
        return int(match.group(1), 0) if match is not None else 0
    except ValueError:
        return 0


def _memory_width(mnemonic: str) -> int:
    if mnemonic.startswith(("ldp", "stp")):
        return 128
    if mnemonic.startswith(("ld", "sd", "lg", "stg", "std", "ldr")) or mnemonic.endswith(".d"):
        return 64
    if mnemonic.startswith(("lw", "sw", "stw")) or mnemonic.endswith(".w"):
        return 32
    if mnemonic.startswith(("lh", "sh")):
        return 16
    return 8


def _define_register(spec: Spec, operand: Op | None) -> None:
    if operand is not None and operand.kind == "reg" and operand.base is not None:
        spec.defs.add(operand.base)


def _is_recognized_arithmetic(mnemonic: str) -> bool:
    return mnemonic in _RECOGNIZED_ARITHMETIC or mnemonic in {
        "addc", "adde", "addze", "subfc", "subfe", "subfze",
        "add.d", "add.w", "sub.d", "sub.w", "mul.d", "mul.w",
    }


def _destination_is_read(
    mnemonic: str,
    architecture: ArchitectureFamily,
) -> bool:
    if architecture == ArchitectureFamily.S390X:
        return mnemonic.startswith(("al", "sl", "x", "o", "n")) or mnemonic == "aghi"
    if architecture == ArchitectureFamily.AARCH64:
        return mnemonic == "bfi"
    if architecture == ArchitectureFamily.ARM32:
        return mnemonic == "umaal"
    return False


def _has_writeback(instruction: Instr) -> bool:
    return any(operand.text.endswith("]!") for operand in instruction.ops) or (
        any(operand.kind == "mem" and "]" in operand.text for operand in instruction.ops)
        and instruction.ops[-1].kind == "imm"
    )


def clone_native_instruction_with_renaming(
    instruction: Instr,
    rename_map: Dict[str, str],
    architecture: ArchitectureFamily,
) -> Instr:
    """Clone an AArch64 GPR instruction with its operand metadata reparsed.

    Other ISAs require separate encoding and implicit-register contracts.
    """
    if not rename_map:
        return instruction
    if architecture != ArchitectureFamily.AARCH64 or any(
        re.fullmatch(r"x(?:[12]?\d|30)", register) is None
        for pair in rename_map.items() for register in pair
    ):
        raise ValueError("native register cloning requires AArch64 GPRs x0 through x30")

    def replace_register(match: re.Match[str]) -> str:
        prefix, number = match.groups()
        target = rename_map.get("x" + number)
        return prefix + target[1:] if target else match.group()

    operands = [re.sub(r"\b([xw])(\d+)\b", replace_register, op.text) for op in instruction.ops]
    line = instruction.mnemonic + (" " + ", ".join(operands) if operands else "")
    return parse_native_line(line, architecture)


__all__ = [
    "clone_native_instruction_with_renaming",
    "native_instruction_spec",
    "native_named_registers",
    "native_pointer_registers",
    "parse_native_line",
]
