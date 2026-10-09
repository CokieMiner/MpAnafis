"""AT&T instruction and operand AST definitions for kernel search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Token -> base 64-bit register name (canonical map from asm_util)
from ..asm_util import GPR_ALIAS_MAP as _REG, split_asm_operands
REGS64 = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]

FLAG_FULL = "F"
FLAG_CF = "CF"
FLAG_OF = "OF"
FLAG_PSEUDOS = (FLAG_FULL, FLAG_CF, FLAG_OF)
_CONDITION = r"(?:a|ae|b|be|c|e|g|ge|l|le|na|nae|nb|nbe|nc|ne|ng|nge|nl|nle|no|np|ns|nz|o|p|pe|po|s|z)"


@dataclass
class Op:
    text: str
    kind: str            # "reg" | "imm" | "mem" | "other"
    base: Optional[str]  # base reg for "reg"
    regs: List[str]      # base regs referenced
    addr: Optional[str]  # canonical address string for "mem"
    width: int = 64      # access width in bits
    index: Optional[str] = None
    scale: int = 1
    displacement: int = 0


@dataclass
class Instr:
    line: str
    mnemonic: str
    ops: List[Op]


@dataclass
class Spec:
    uses: Set[str] = field(default_factory=set)
    defs: Set[str] = field(default_factory=set)
    flags_read: Set[str] = field(default_factory=set)
    flags_write: Set[str] = field(default_factory=set)
    mem: Optional[str] = None
    unknown: bool = False


def _tok_width(name: str) -> int:
    if name.startswith("r") and (name.endswith("d") or name.endswith("w") or name.endswith("b")):
        if name.endswith("d"): return 32
        if name.endswith("w"): return 16
        if name.endswith("b"): return 8
    if name.startswith("e"): return 32
    if name in ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp"): return 16
    if name.endswith("l") or name.endswith("h"): return 8
    return 64


def parse_operand(tok: str) -> Op:
    """Parse a single assembly operand token into structured register/memory/immediate Op."""
    t = tok.strip()
    if t.startswith("$"):
        return Op(text=t, kind="imm", base=None, regs=[], addr=None)
    if re.fullmatch(r"%[A-Za-z][A-Za-z0-9]*", t):
        name = t[1:]
        base = _REG.get(name)
        return Op(text=t, kind="reg", base=base or name,
                  regs=[base or name], addr=None,
                  width=_tok_width(name))
    if "(" in t:
        m = re.match(r"^(?P<disp>[+-]?(?:0x[0-9a-fA-F]+|\d+))?(?P<rest>\(.*\))$", t)
        if m is None:
            regs = [_REG.get(name, name) for name in re.findall(r"%([\w]+)", t)]
            return Op(t, "mem", None, regs, None)
        body = m.group("rest")[1:-1]
        disp_s = m.group("disp") if m else None
        disp = int(disp_s, 16 if "x" in disp_s.lower() else 10) if disp_s else 0
        base = index = None
        scale = 1
        parts = [part.strip() for part in body.split(",")]
        if len(parts) > 3 or any(
            part and re.fullmatch(r"%[A-Za-z][A-Za-z0-9]*", part) is None
            for part in parts[:2]
        ) or (len(parts) == 3 and parts[2] not in ("1", "2", "4", "8")):
            regs = [_REG.get(name, name) for name in re.findall(r"%([\w]+)", t)]
            return Op(t, "mem", None, regs, None)
        if parts[0]:
            base = _REG.get(parts[0][1:], parts[0][1:])
        if len(parts) > 1 and parts[1]:
            index = _REG.get(parts[1][1:], parts[1][1:])
        if len(parts) == 3:
            scale = int(parts[2])
        regs = [r for r in (base, index) if r]
        addr = f"m({base or '-'},{index or '-'},{scale},{disp})"
        if disp_s and re.fullmatch(r"[+-]?0\d+", disp_s):
            addr = None  # Leading-zero radix conventions are assembler-specific.
        return Op(
            text=t,
            kind="mem",
            base=base,
            regs=regs,
            addr=addr,
            index=index,
            scale=scale,
            displacement=disp,
        )
    return Op(text=t, kind="other", base=None, regs=[], addr=None)


def parse_operands(text: str) -> List[Op]:
    """Parse a comma-delimited operand list respecting nested parenthesis."""
    return [parse_operand(token) for token in split_asm_operands(text)]


def parse_line(line: str) -> Optional[Instr]:
    """Parse a raw assembly source line into a structured Instr object."""
    s = line.split("#", 1)[0].split("//", 1)[0].strip()
    if not s or s.startswith("#") or s.startswith(".") or s.endswith(":"):
        return None
    parts = s.split(None, 1)
    mnemonic = parts[0].lower()
    ops_text = parts[1] if len(parts) > 1 else ""
    ops = parse_operands(ops_text)
    _, width = _base(mnemonic)
    for op in ops:
        if op.kind == "mem":
            op.width = width
    return Instr(line=s, mnemonic=mnemonic, ops=ops)


_BASE = {
    "add": "bin", "sub": "bin", "and": "bin", "or": "bin", "xor": "bin",
    "adc": "binf", "sbb": "binf",
    "adcx": "adcx", "adox": "adox",
    "mov": "mov", "lea": "lea", "movzx": "movzx", "movsx": "movsx",
    "neg": "uni", "not": "unilogic",
    "inc": "uninc", "dec": "uninc",
    "mul": "mul", "imul": "imul", "mulx": "mulx",
    "shl": "shift", "shr": "shift", "sar": "shift",
    "shlx": "shiftx", "shrx": "shiftx", "sarx": "shiftx",
    "shld": "dshift", "shrd": "dshift",
    "cmp": "cmp", "test": "cmp", "bt": "cmp",
    "xchg": "xchg", "bswap": "bswap",
    "push": "push", "pop": "pop",
    "clc": "flagonly", "stc": "flagonly", "cmc": "flagonly",
    "nop": "nop",
}


def _base(mnem: str) -> Tuple[str, int]:
    low = mnem.lower()
    if low in ("retq", "ret"): return ("ret", 0)
    if low.startswith("j"): return ("br", 0)
    if re.fullmatch("set" + _CONDITION, low): return ("setcc", 8)
    if re.fullmatch("cmov" + _CONDITION + "[wlq]?", low): return ("cmov", {"w": 16, "l": 32, "q": 64}.get(low[-1], 64))
    if low in ("mulxq", "mulx"): return ("mulx", 64)
    if low in ("movabsq", "movabs"): return ("mov", 64)
    if low in ("movzbl", "movzbw", "movzwq", "movzwl"):
        return ("movzx", 64 if low.endswith("q") else 32)
    if low in ("movslq", "movswq", "movsbl", "movsbw"):
        return ("movsx", 64 if low.endswith("q") else 32)
    if low in ("shlxq", "shlx", "shrxq", "shrx", "sarxq", "sarx"):
        return ("shiftx", 64)
    if low in ("shldq", "shld", "shrdq", "shrd"):
        return ("dshift", 64)
    W = {"q": 64, "l": 32, "w": 16, "b": 8}
    if len(low) > 1 and low[-1] in W and low[:-1] in _BASE:
        return (_BASE[low[:-1]], W[low[-1]])
    if low in _BASE:
        return (_BASE[low], 64)
    return (low, 64)


def get_instruction_spec(instr: Instr) -> Spec:
    """Analyze instruction effects and compute read/written registers, flags, and memory effects."""
    base, _w = _base(instr.mnemonic)
    ops = instr.ops
    sp = Spec()
    src = ops[0] if ops else None
    dst = ops[-1] if ops else None

    for idx, op in enumerate(ops):
        if op.kind == "mem" and base not in ("lea", "nop"):
            sp.uses |= set(op.regs)
            if idx == len(ops) - 1 and base not in ("cmp", "push"):
                sp.mem = "store"
            elif sp.mem is None:
                sp.mem = "load"

    if base == "nop":
        pass
    elif base == "mov":
        if src: sp.uses |= set(src.regs)
        if dst and dst.kind == "reg" and dst.base: sp.defs.add(dst.base)
    elif base in ("movzx", "movsx"):
        if src: sp.uses |= set(src.regs)
        if dst and dst.kind == "reg" and dst.base: sp.defs.add(dst.base)
    elif base == "lea":
        if dst and dst.base: sp.defs.add(dst.base)
        if src: sp.uses |= set(src.regs)
    elif base in ("bin", "binf"):
        if src: sp.uses |= set(src.regs)
        if dst:
            if dst.kind == "reg" and dst.base:
                sp.uses.add(dst.base)
                sp.defs.add(dst.base)
        if base == "binf":
            sp.flags_read.add(FLAG_CF)
        sp.flags_write.add(FLAG_FULL)
    elif base in ("adcx", "adox"):
        flag = FLAG_CF if base == "adcx" else FLAG_OF
        if src: sp.uses |= set(src.regs)
        if dst and dst.kind == "reg" and dst.base:
            sp.uses.add(dst.base)
            sp.defs.add(dst.base)
        sp.flags_read.add(flag)
        sp.flags_write.add(flag)
    elif base == "mulx":
        sp.uses.add("rdx")
        if src: sp.uses |= set(src.regs)
        if len(ops) >= 3:
            if ops[1].base: sp.defs.add(ops[1].base)
            if ops[2].base: sp.defs.add(ops[2].base)
    elif base == "mul":
        # Single-operand mul: RDX:RAX = RAX * src
        sp.uses.add("rax")
        if src: sp.uses |= set(src.regs)
        sp.defs.add("rax")
        sp.defs.add("rdx")
        sp.flags_write.add(FLAG_FULL)
    elif base == "imul":
        if len(ops) == 1:
            # Single-operand: RDX:RAX = RAX * src
            sp.uses.add("rax")
            if src: sp.uses |= set(src.regs)
            sp.defs.add("rax")
            sp.defs.add("rdx")
        elif len(ops) == 2:
            # Two-operand: dst *= src
            if src: sp.uses |= set(src.regs)
            if dst and dst.kind == "reg" and dst.base:
                sp.uses.add(dst.base)
                sp.defs.add(dst.base)
        else:
            # Three-operand: dst = src1 * imm
            for op in ops[:-1]: sp.uses |= set(op.regs)
            if dst and dst.kind == "reg" and dst.base:
                sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
    elif base == "uni":
        # neg: reads and writes the operand, writes all flags
        if dst:
            sp.uses |= set(dst.regs)
            if dst.kind == "reg" and dst.base:
                sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
    elif base == "unilogic":
        # not: reads and writes the operand, does NOT write flags
        if dst:
            sp.uses |= set(dst.regs)
            if dst.kind == "reg" and dst.base:
                sp.defs.add(dst.base)
    elif base == "uninc":
        # inc/dec: reads and writes the operand, writes flags except CF
        if dst:
            sp.uses |= set(dst.regs)
            if dst.kind == "reg" and dst.base:
                sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
    elif base == "shift":
        # shl/shr/sar: AT&T shift — src is count, dst is modified
        if src and src.kind == "reg": sp.uses |= set(src.regs)
        if dst:
            sp.uses |= set(dst.regs)
            if dst.kind == "reg" and dst.base:
                sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
        # A masked count of zero preserves the incoming flags.
        sp.flags_read.add(FLAG_FULL)
    elif base == "shiftx":
        # shlx/shrx/sarx (BMI2): three-operand, NO flag writes
        # AT&T: shlxq %count, %src, %dst → dst = src << count
        if len(ops) >= 3:
            if ops[0].kind == "reg": sp.uses |= set(ops[0].regs)
            sp.uses |= set(ops[1].regs)
            if ops[2].kind == "reg" and ops[2].base:
                sp.defs.add(ops[2].base)
        elif len(ops) == 2:
            if src: sp.uses |= set(src.regs)
            if dst and dst.kind == "reg" and dst.base:
                sp.uses.add(dst.base)
                sp.defs.add(dst.base)
        # BMI2 shifts do not write flags
    elif base == "dshift":
        # shld/shrd: double-precision shift — src feeds bits into dst
        # AT&T: shldq $imm, %src, %dst
        for op in ops[:-1]: sp.uses |= set(op.regs)
        if dst:
            sp.uses |= set(dst.regs)
            if dst.kind == "reg" and dst.base:
                sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
        sp.flags_read.add(FLAG_FULL)
    elif base == "setcc":
        # setcc: reads flags, defines a byte register
        sp.flags_read.add(FLAG_FULL)
        if dst and dst.kind == "reg" and dst.base:
            sp.defs.add(dst.base)
    elif base == "cmov":
        # cmovcc: conditional move — reads flags, src, and dst; writes dst
        sp.flags_read.add(FLAG_FULL)
        if src: sp.uses |= set(src.regs)
        if dst and dst.kind == "reg" and dst.base:
            sp.uses.add(dst.base)
            sp.defs.add(dst.base)
    elif base == "br":
        # Conditional/unconditional branch — reads flags for conditional
        sp.flags_read.add(FLAG_FULL)
        if src: sp.uses |= set(src.regs)
    elif base == "cmp":
        # cmp/test/bt: read both operands, write flags only
        for op in ops:
            sp.uses |= set(op.regs)
        sp.flags_write.add(FLAG_FULL)
    elif base == "xchg":
        # xchg: swaps two operands, both read and written
        for op in ops:
            sp.uses |= set(op.regs)
            if op.kind == "reg" and op.base:
                sp.defs.add(op.base)
        if any(op.kind == "mem" for op in ops):
            sp.mem = "store"
            sp.unknown = True  # Memory XCHG also imposes atomic ordering.
    elif base == "bswap":
        # bswap: single-operand, reads and writes it
        if dst and dst.kind == "reg" and dst.base:
            sp.uses.add(dst.base)
            sp.defs.add(dst.base)
    elif base == "push":
        if src: sp.uses |= set(src.regs)
        sp.mem = "store"
        sp.uses.add("rsp")
        sp.defs.add("rsp")
        sp.unknown = True  # Includes an implicit stack access.
    elif base == "pop":
        if dst and dst.kind == "reg" and dst.base:
            sp.defs.add(dst.base)
        sp.mem = "load"
        sp.uses.add("rsp")
        sp.defs.add("rsp")
        sp.unknown = True
    elif base == "flagonly":
        # clc/stc/cmc: only modify flags
        sp.flags_write.add(FLAG_CF)
        if instr.mnemonic == "cmc":
            sp.flags_read.add(FLAG_CF)
    else:
        # Unknown instruction: conservatively mark all operands as both
        # used and defined, and assume flags are written.
        for op in ops:
            sp.uses |= set(op.regs)
        if dst and dst.kind == "reg" and dst.base:
            sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
        sp.unknown = True

    # Byte and word destinations merge with the prior full register value.
    for op in ops:
        if op.kind == "reg" and op.base in sp.defs and op.width < 32:
            sp.uses.add(op.base)
    # Unsupported register banks and absolute/symbolic operands remain barriers.
    if any(op.kind == "other" or any(r not in REGS64 for r in op.regs) for op in ops):
        sp.unknown = True
    return sp


_X86_REG_WIDTH_MAP = {
    "rax": {64: "rax", 32: "eax", 16: "ax", 8: "al"},
    "rbx": {64: "rbx", 32: "ebx", 16: "bx", 8: "bl"},
    "rcx": {64: "rcx", 32: "ecx", 16: "cx", 8: "cl"},
    "rdx": {64: "rdx", 32: "edx", 16: "dx", 8: "dl"},
    "rsi": {64: "rsi", 32: "esi", 16: "si", 8: "sil"},
    "rdi": {64: "rdi", 32: "edi", 16: "di", 8: "dil"},
    "rbp": {64: "rbp", 32: "ebp", 16: "bp", 8: "bpl"},
    "rsp": {64: "rsp", 32: "esp", 16: "sp", 8: "spl"},
    **{
        f"r{i}": {64: f"r{i}", 32: f"r{i}d", 16: f"r{i}w", 8: f"r{i}b"}
        for i in range(8, 16)
    },
}


def format_x86_reg(base: str, width: int = 64) -> str:
    """Format x86 register with specified access width."""
    base_lower = base.lower().lstrip("%")
    variants = _X86_REG_WIDTH_MAP.get(base_lower)
    if variants:
        return variants.get(width, variants.get(64, base_lower))
    return base_lower


def clone_instruction_with_renaming(instr: Instr, rename_map: Dict[str, str]) -> Instr:
    """Clone an instruction replacing registers according to rename_map."""
    if not rename_map:
        return instr
    new_ops: List[Op] = []
    for op in instr.ops:
        if op.kind == "reg" and op.base in rename_map:
            new_base = rename_map[op.base]
            reg_name = format_x86_reg(new_base, op.width)
            if op.text in ("%ah", "%bh", "%ch", "%dh"):
                high_bytes = {"rax": "ah", "rbx": "bh", "rcx": "ch", "rdx": "dh"}
                if new_base not in high_bytes:
                    raise ValueError("destination register has no high-byte alias")
                reg_name = high_bytes[new_base]
            new_ops.append(
                Op(
                    text=f"%{reg_name}",
                    kind="reg",
                    base=new_base,
                    regs=[new_base],
                    addr=None,
                    width=op.width,
                )
            )
        elif op.kind == "mem" and any(r in rename_map for r in op.regs):
            def renamed_address_register(match: re.Match[str]) -> str:
                name = match.group(1)
                canonical = _REG.get(name, name)
                return (f"%{format_x86_reg(rename_map[canonical], _tok_width(name))}"
                        if canonical in rename_map else match.group())

            new_text = re.sub(r"%([A-Za-z][A-Za-z0-9]*)", renamed_address_register, op.text)
            renamed_operand = parse_operand(new_text)
            renamed_operand.width = op.width
            new_ops.append(renamed_operand)
        else:
            new_ops.append(op)

    ops_str = ", ".join(op.text for op in new_ops)
    line_parts = instr.line.split(None, 1)
    prefix_indent = instr.line[:len(instr.line) - len(instr.line.lstrip())]
    mnemonic = line_parts[0] if line_parts else instr.mnemonic
    new_line = f"{prefix_indent}{mnemonic} {ops_str}" if ops_str else f"{prefix_indent}{mnemonic}"
    return Instr(line=new_line, mnemonic=instr.mnemonic, ops=new_ops)


__all__ = [
    "FLAG_CF",
    "FLAG_FULL",
    "FLAG_OF",
    "FLAG_PSEUDOS",
    "Instr",
    "Op",
    "Spec",
    "clone_instruction_with_renaming",
    "format_x86_reg",
    "get_instruction_spec",
    "parse_line",
    "parse_operand",
    "parse_operands",
]
