"""Parser for Rust ``asm!`` blocks and operand clauses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_STRING_START = re.compile(r'"|r#*"')
_CHAR_LITERAL = re.compile(r"'(?:\\(?:u\{[\da-fA-F_]+\}|x[\da-fA-F]{2}|.)|[^'\\\r\n])'")

@dataclass(frozen=True)
class Operand:
    """One Rust ``asm!`` operand clause."""

    name: Optional[str]
    cls: Optional[str]
    kind: str
    explicit: bool = False
    discard: bool = False


@dataclass(frozen=True)
class AsmBlock:
    """One parsed Rust ``asm!`` invocation."""

    line: int
    instructions: List[str]
    operands: List[Operand]
    options: Optional[str]


def find_asm_blocks(text: str) -> List[Tuple[int, int]]:
    """Locate balanced ``asm!(...)`` invocations outside strings and comments."""
    blocks: List[Tuple[int, int]] = []
    cursor = 0
    pattern = re.compile(r"\basm\s*!\s*\(")
    while cursor < len(text):
        if text.startswith(("//", "/*"), cursor):
            cursor = _skip_comment(text, cursor)
            continue
        if character := _CHAR_LITERAL.match(text, cursor):
            cursor = character.end()
            continue
        if _STRING_START.match(text, cursor):
            cursor = _skip_string(text, cursor)
            continue
        match = pattern.match(text, cursor)
        if match is None:
            cursor += 1
            continue
        open_position = match.end() - 1
        depth = 0
        index = open_position
        while index < len(text):
            character = text[index]
            if literal := _CHAR_LITERAL.match(text, index):
                index = literal.end()
                continue
            if _STRING_START.match(text, index):
                end = _skip_string(text, index)
                index = end
                continue
            if character == "/" and index + 1 < len(text) and text[index + 1] in ("/", "*"):
                index = _skip_comment(text, index)
                continue
            if character in "([{":
                depth += 1
            elif character in ")]}":
                depth -= 1
                if depth == 0:
                    blocks.append((match.start(), index + 1))
                    break
            index += 1
        cursor = index + 1
    return blocks


def split_args(body: str) -> List[str]:
    """Split top-level macro arguments while preserving nested expressions."""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    index = 0
    while index < len(body):
        character = body[index]
        if literal := _CHAR_LITERAL.match(body, index):
            current.append(literal.group())
            index = literal.end()
            continue
        if _STRING_START.match(body, index):
            end = _skip_string(body, index)
            current.append(body[index:end])
            index = end
            continue
        if character == "/" and index + 1 < len(body) and body[index + 1] in ("/", "*"):
            index = _skip_comment(body, index)
            continue
        if character in "([{":
            depth += 1
            current.append(character)
        elif character in ")]}":
            depth -= 1
            current.append(character)
        elif character == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
        index += 1
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def is_string_literal(argument: str) -> bool:
    """Return whether an argument begins with a normal or raw string."""
    return argument.startswith('"') or argument.startswith('r"') or re.match(
        r"r#+\"", argument,
    ) is not None


def parse_operand(argument: str) -> Optional[Operand]:
    """Parse a supported Rust inline-assembly operand clause."""
    argument = argument.strip()
    name: Optional[str] = None
    body = argument
    equals = argument.find("=")
    if equals != -1 and argument[:equals].strip().isidentifier():
        name = argument[:equals].strip()
        body = argument[equals + 1:].strip()
    match = re.match(r"(in|out|inout|lateout|inlateout)\s*\((.*?)\)", body, re.S)
    if match is None:
        return None
    kind, inner = match.group(1), match.group(2).strip()
    value = body[match.end():].strip()
    register_class = None
    explicit = False
    if inner.isidentifier():
        register_class = inner
    elif inner.startswith('"') and inner.endswith('"'):
        register_class = inner[1:-1]
        explicit = True
    return Operand(
        name=name,
        cls=register_class,
        kind=kind,
        explicit=explicit,
        discard=value == "_" or value.rsplit("=>", 1)[-1].strip() == "_",
    )


def extract_asm_blocks(path: Path) -> List[AsmBlock]:
    """Parse every supported ``asm!`` block in a Rust source file."""
    text = path.read_text(encoding="utf-8")
    blocks: List[AsmBlock] = []
    for start, end in find_asm_blocks(text):
        line = text.count("\n", 0, start) + 1
        body = text[text.index("(", start) + 1:end - 1]
        arguments = split_args(body)
        instructions = [
            argument for argument in arguments
            if is_string_literal(argument) or argument.startswith("concat!")
        ]
        options = None
        operands = []
        for argument in arguments:
            if argument in instructions:
                continue
            if re.match(r"options\s*\(", argument):
                options = argument
                continue
            operand = parse_operand(argument)
            if operand is None:
                raise ValueError(f"unsupported asm! argument at line {line}: {argument}")
            operands.append(operand)
        blocks.append(AsmBlock(line, instructions, operands, options))
    return blocks


def _skip_string(text: str, index: int) -> int:
    if text[index] == "r":
        hashes = 0
        cursor = index + 1
        while cursor < len(text) and text[cursor] == "#":
            hashes += 1
            cursor += 1
        if cursor < len(text) and text[cursor] == '"':
            terminator = '"' + "#" * hashes
            end = text.find(terminator, cursor + 1)
            return end + len(terminator) if end != -1 else len(text)
    if text[index] == '"':
        cursor = index + 1
        while cursor < len(text):
            if text[cursor] == "\\":
                cursor += 2
                continue
            if text[cursor] == '"':
                return cursor + 1
            cursor += 1
    return len(text)


def _skip_comment(text: str, index: int) -> int:
    if text.startswith("//", index):
        end = text.find("\n", index)
        return end if end != -1 else len(text)
    if text.startswith("/*", index):
        depth = 1
        cursor = index + 2
        while cursor < len(text) and depth:
            if text.startswith("/*", cursor):
                depth += 1
                cursor += 2
            elif text.startswith("*/", cursor):
                depth -= 1
                cursor += 2
            else:
                cursor += 1
        return cursor
    return index


__all__ = [
    "AsmBlock",
    "Operand",
    "extract_asm_blocks",
    "find_asm_blocks",
    "is_string_literal",
    "parse_operand",
    "split_args",
]
