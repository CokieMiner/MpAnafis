"""AT&T instruction and operand AST definitions for kernel search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

# Token -> base 64-bit register name (canonical map from asm_util)
from ..asm_util import GPR_ALIAS_MAP as _REG
REGS64 = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]

FLAG_FULL = "F"
FLAG_CF = "CF"
FLAG_OF = "OF"
FLAG_PSEUDOS = (FLAG_FULL, FLAG_CF, FLAG_OF)


@dataclass
class Op:
    text: str
    kind: str            # "reg" | "imm" | "mem" | "other"
    base: Optional[str]  # base reg for "reg"
    regs: List[str]      # base regs referenced
    addr: Optional[str]  # canonical address string for "mem"
    width: int = 64      # access width in bits


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
    if t.startswith("%"):
        name = t[1:]
        base = _REG.get(name)
        return Op(text=t, kind="reg", base=base or name,
                  regs=[base or name] if base else [], addr=None,
                  width=_tok_width(name))
    if "(" in t:
        m = re.match(r"^(?P<disp>[+-]?(?:0x[0-9a-fA-F]+|\d+))?(?P<rest>\(.*\))$", t)
        body = (m.group("rest")[1:-1] if m else t[1:-1])
        disp_s = m.group("disp") if m else None
        disp = int(disp_s, 0) if disp_s else 0
        base = index = None
        scale = 1
        for part in body.split(","):
            p = part.strip()
            if not p:
                continue
            if p.startswith("%"):
                b = _REG.get(p[1:], p[1:])
                if base is None:
                    base = b
                else:
                    index = b
            else:
                scale = int(p, 0)
        regs = [r for r in (base, index) if r]
        addr = f"m({base or '-'},{index or '-'},{scale},{disp})"
        return Op(text=t, kind="mem", base=base, regs=regs, addr=addr)
    return Op(text=t, kind="other", base=None, regs=[], addr=None)


def parse_operands(text: str) -> List[Op]:
    """Parse a comma-delimited operand list respecting nested parenthesis."""
    if not text:
        return []
    out: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [parse_operand(t) for t in out if t]


def parse_line(line: str) -> Optional[Instr]:
    """Parse a raw assembly source line into a structured Instr object."""
    s = line.strip()
    if not s or s.startswith("#") or s.startswith(".") or s.endswith(":"):
        return None
    parts = s.split(None, 1)
    mnemonic = parts[0].lower()
    ops_text = parts[1] if len(parts) > 1 else ""
    return Instr(line=s, mnemonic=mnemonic, ops=parse_operands(ops_text))


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
    if low.startswith("set"): return ("setcc", 0)
    if low.startswith("cmov"): return ("cmov", 0)
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
            if src: sp.uses |= set(src.regs)
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
        if src: sp.uses |= set(src.regs)
        if dst:
            sp.uses |= set(dst.regs)
            if dst.kind == "reg" and dst.base:
                sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
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
    elif base == "bswap":
        # bswap: single-operand, reads and writes it
        if dst and dst.kind == "reg" and dst.base:
            sp.uses.add(dst.base)
            sp.defs.add(dst.base)
    elif base == "push":
        if src: sp.uses |= set(src.regs)
        sp.mem = "store"
    elif base == "pop":
        if dst and dst.kind == "reg" and dst.base:
            sp.defs.add(dst.base)
        sp.mem = "load"
    elif base == "flagonly":
        # clc/stc/cmc: only modify flags
        sp.flags_write.add(FLAG_FULL)
    else:
        # Unknown instruction: conservatively mark all operands as both
        # used and defined, and assume flags are written.
        for op in ops:
            sp.uses |= set(op.regs)
        if dst and dst.kind == "reg" and dst.base:
            sp.defs.add(dst.base)
        sp.flags_write.add(FLAG_FULL)
        sp.unknown = True

    return sp
