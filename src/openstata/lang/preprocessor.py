"""Stata .do/.ado script preprocessor: comments, continuations, #delimit, blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openstata.core.errors import StataError


@dataclass
class Stmt:
    line_no: int


@dataclass
class CommandStmt(Stmt):
    text: str


@dataclass
class BlockStmt(Stmt):
    header: str
    body: list[Stmt] = field(default_factory=list)
    else_body: list[Stmt] | None = None


@dataclass
class ProgramDefStmt(Stmt):
    name: str
    body: list[Stmt] = field(default_factory=list)


@dataclass
class InputStmt(Stmt):
    header: str
    rows: list[str] = field(default_factory=list)


def strip_inline_comments(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Strip /* ... */ and // or /// comments outside quotes, returning (clean, still_in_block)."""
    out: list[str] = []
    i = 0
    n = len(line)
    in_dq = False
    compound_depth = 0

    while i < n:
        if in_block_comment:
            end = line.find("*/", i)
            if end == -1:
                return "".join(out), True
            in_block_comment = False
            i = end + 2
            continue

        ch = line[i]
        if ch == "`" and i + 1 < n and line[i + 1] == '"':
            compound_depth += 1
            out.append('`"')
            i += 2
            continue
        if compound_depth > 0 and ch == '"' and i + 1 < n and line[i + 1] == "'":
            compound_depth -= 1
            out.append('"\'')
            i += 2
            continue
        if compound_depth == 0 and ch == '"':
            in_dq = not in_dq
            out.append(ch)
            i += 1
            continue

        if not in_dq and compound_depth == 0:
            if line[i : i + 2] == "/*":
                in_block_comment = True
                i += 2
                continue
            if line[i : i + 3] == "///":
                # Line continuation marker: keep marker at end and drop rest of line
                out.append("///")
                break
            if line[i : i + 2] == "//":
                break
        out.append(ch)
        i += 1
    return "".join(out), in_block_comment


def split_logical_lines(source: str) -> list[tuple[int, str]]:
    """Split script source into (start_line_no, logical_statement) pairs."""
    raw_lines = source.splitlines()
    delimit: Literal["cr", ";"] = "cr"
    in_block_comment = False
    logical: list[tuple[int, str]] = []
    buf = ""
    buf_start = 1

    for idx, raw in enumerate(raw_lines, start=1):
        stripped_lead = raw.lstrip()
        if not in_block_comment and not buf and stripped_lead.startswith("*"):
            continue

        clean, in_block_comment = strip_inline_comments(raw, in_block_comment)
        if not buf and not clean.strip():
            continue

        # Handle #delimit directive on its own line in cr mode
        if not buf and clean.strip().lower().startswith("#delimit"):
            arg = clean.strip()[8:].strip().lower()
            if arg in (";", "cr"):
                delimit = ";" if arg == ";" else "cr"
                continue
            raise StataError(f"invalid #delimit argument: {arg}", rc=198)

        has_cont = clean.rstrip().endswith("///")
        if has_cont:
            pos = clean.rfind("///")
            part = clean[:pos]
            if not buf:
                buf_start = idx
            buf += part + " "
            continue

        if delimit == "cr":
            if buf:
                stmt = (buf + clean).strip()
                if stmt:
                    logical.append((buf_start, stmt))
                buf = ""
            else:
                s = clean.strip()
                if s:
                    logical.append((idx, s))
        else:
            # Semicolon delimiter mode
            if not buf and clean.strip().lower().startswith("#delimit"):
                arg = clean.strip()[8:].strip().rstrip(";").strip().lower()
                if arg in (";", "cr"):
                    delimit = ";" if arg == ";" else "cr"
                    continue
                raise StataError(f"invalid #delimit argument: {arg}", rc=198)
            if not buf:
                buf_start = idx
            buf += clean + " "
            while ";" in buf:
                # Split on ; outside quotes
                split_pos = _find_unquoted_semicolon(buf)
                if split_pos == -1:
                    break
                piece = buf[:split_pos].strip()
                buf = buf[split_pos + 1 :]
                if piece.lower().startswith("#delimit"):
                    arg = piece[8:].strip().lower()
                    delimit = "cr" if arg == "cr" else ";"
                elif piece:
                    logical.append((buf_start, piece))
                buf_start = idx

    if buf.strip():
        rem = buf.strip()
        if not rem.lower().startswith("#delimit"):
            logical.append((buf_start, rem))
    return logical


def _find_unquoted_semicolon(text: str) -> int:
    in_dq = False
    c_depth = 0
    i = 0
    while i < len(text):
        if text[i : i + 2] == '`"':
            c_depth += 1
            i += 2
            continue
        if c_depth > 0 and text[i : i + 2] == '"\'':
            c_depth -= 1
            i += 2
            continue
        if c_depth == 0 and text[i] == '"':
            in_dq = not in_dq
        elif not in_dq and c_depth == 0 and text[i] == ";":
            return i
        i += 1
    return -1


def parse_script(source: str) -> list[Stmt]:
    """Parse Stata script text into a list of AST statements."""
    lines = split_logical_lines(source)
    stmts, idx = _parse_block(lines, 0, stop_tokens=set())
    if idx < len(lines):
        raise StataError(f"unexpected '{lines[idx][1]}' at line {lines[idx][0]}", rc=198)
    return stmts


def _ends_with_open_brace(stmt: str) -> bool:
    s = stmt.rstrip()
    return s.endswith("{")


def _parse_block(
    lines: list[tuple[int, str]], idx: int, stop_tokens: set[str]
) -> tuple[list[Stmt], int]:
    stmts: list[Stmt] = []
    n = len(lines)
    while idx < n:
        line_no, text = lines[idx]
        low = text.strip().lower()
        if low in stop_tokens:
            return stmts, idx
        if low == "}" and "}" in stop_tokens:
            return stmts, idx

        # Inline `program define name ...` until `end`
        if low.startswith("program ") or low == "program":
            parts = text.split()
            if len(parts) >= 2 and parts[1].lower() in ("define", "def"):
                if len(parts) < 3:
                    raise StataError("program define requires a program name", rc=198)
                pname = parts[2]
                body, next_idx = _parse_block(lines, idx + 1, {"end"})
                if next_idx >= n or lines[next_idx][1].strip().lower() != "end":
                    raise StataError(f"missing 'end' for program {pname}", rc=198)
                stmts.append(ProgramDefStmt(line_no=line_no, name=pname, body=body))
                idx = next_idx + 1
                continue
            if len(parts) >= 2 and parts[1].lower() not in ("drop", "dir", "list"):
                pname = parts[1]
                body, next_idx = _parse_block(lines, idx + 1, {"end"})
                if next_idx >= n or lines[next_idx][1].strip().lower() != "end":
                    raise StataError(f"missing 'end' for program {pname}", rc=198)
                stmts.append(ProgramDefStmt(line_no=line_no, name=pname, body=body))
                idx = next_idx + 1
                continue

        # `input ... end`
        if low == "input" or low.startswith("input "):
            rows: list[str] = []
            j = idx + 1
            while j < n and lines[j][1].strip().lower() != "end":
                rows.append(lines[j][1])
                j += 1
            if j >= n:
                raise StataError("missing 'end' for input block", rc=198)
            stmts.append(InputStmt(line_no=line_no, header=text, rows=rows))
            idx = j + 1
            continue

        # Brace block: foreach / forvalues / while / if / quietly { / capture { / {
        if _ends_with_open_brace(text):
            header = text.rstrip()[:-1].strip()
            body, next_idx = _parse_block(lines, idx + 1, {"}"})
            if next_idx >= n or lines[next_idx][1].strip() != "}":
                raise StataError(f"unmatched '{{' at line {line_no}", rc=198)
            idx = next_idx + 1
            else_body: list[Stmt] | None = None
            # Check for trailing `else {`
            if idx < n and lines[idx][1].strip().lower().startswith("else") and _ends_with_open_brace(lines[idx][1]):
                else_body, next_idx2 = _parse_block(lines, idx + 1, {"}"})
                if next_idx2 >= n or lines[next_idx2][1].strip() != "}":
                    raise StataError(f"unmatched '{{' in else block at line {lines[idx][0]}", rc=198)
                idx = next_idx2 + 1
            stmts.append(BlockStmt(line_no=line_no, header=header, body=body, else_body=else_body))
            continue

        stmts.append(CommandStmt(line_no=line_no, text=text))
        idx += 1

    return stmts, idx
