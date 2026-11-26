"""Stata macro expansion (local `...', global $..., inline `=expr', r()/e()/c())."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from openstata.core.dataset import Dataset
    from openstata.core.state import StateStore

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RES_RE = re.compile(r"^(r|e|c|s)\(([^)]+)\)$")


def _lookup_local(
    name: str,
    state: "StateStore",
    dataset: "Dataset | None",
    eval_fn: Callable[[str, "StateStore", "Dataset | None"], str] | None,
) -> str:
    s = name.strip()
    if s.startswith("=") and eval_fn is not None:
        return eval_fn(s[1:].strip(), state, dataset)
    if s.startswith("++"):
        key = s[2:].strip()
        cur = float(state.locals.get(key, "0") or 0) + 1
        val = str(int(cur)) if cur.is_integer() else str(cur)
        state.locals[key] = val
        return val
    if s.startswith("--"):
        key = s[2:].strip()
        cur = float(state.locals.get(key, "0") or 0) - 1
        val = str(int(cur)) if cur.is_integer() else str(cur)
        state.locals[key] = val
        return val
    if s.endswith("++"):
        key = s[:-2].strip()
        cur = float(state.locals.get(key, "0") or 0)
        old = str(int(cur)) if cur.is_integer() else str(cur)
        nxt = cur + 1
        state.locals[key] = str(int(nxt)) if nxt.is_integer() else str(nxt)
        return old
    if s.endswith("--"):
        key = s[:-2].strip()
        cur = float(state.locals.get(key, "0") or 0)
        old = str(int(cur)) if cur.is_integer() else str(cur)
        nxt = cur - 1
        state.locals[key] = str(int(nxt)) if nxt.is_integer() else str(nxt)
        return old

    m = _RES_RE.match(s)
    if m:
        kind, key = m.group(1), m.group(2).strip()
        if kind == "c":
            v = state.get_c_value(key, dataset)
            return "" if v is None else str(v)
        store = getattr(state, kind)
        v = store.get_value(key)
        return "" if v is None else str(v)

    return state.locals.get(s, "")


def _expand_globals(text: str, state: "StateStore") -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] == "$":
            out.append("$")
            i += 2
            continue
        if ch == "$":
            if i + 1 < n and text[i + 1] == "{":
                end = text.find("}", i + 2)
                if end != -1:
                    gname = text[i + 2 : end]
                    out.append(state.globals.get(gname, ""))
                    i = end + 1
                    continue
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if j > i + 1:
                gname = text[i + 1 : j]
                out.append(state.globals.get(gname, ""))
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def expand_macros(
    text: str,
    state: "StateStore",
    dataset: "Dataset | None" = None,
    eval_fn: Callable[[str, "StateStore", "Dataset | None"], str] | None = None,
) -> str:
    """Expand local (`...') and global ($...) macros iteratively inside-out."""
    cur = text
    for _ in range(32):
        changed = False
        stack: list[int] = []
        i = 0
        while i < len(cur):
            if cur[i] == "\\" and i + 1 < len(cur) and cur[i + 1] == "`":
                i += 2
                continue
            # Compound string delimiter `"..."' starts with `" and ends with "'
            if cur[i] == "`" and i + 1 < len(cur) and cur[i + 1] == '"':
                i += 2
                continue
            if cur[i] == "`":
                stack.append(i)
                i += 1
            elif cur[i] == "'" and stack:
                # Ignore closing "' of compound quotes if preceded by "
                if i > 0 and cur[i - 1] == '"':
                    i += 1
                    continue
                start = stack.pop()
                inner = cur[start + 1 : i]
                rep = _lookup_local(inner, state, dataset, eval_fn)
                cur = cur[:start] + rep + cur[i + 1 :]
                changed = True
                break
            else:
                i += 1
        after_g = _expand_globals(cur, state)
        if after_g != cur:
            cur = after_g
            changed = True
        if not changed:
            break
    return cur
