"""Universal Stata command-line parser, varlist expander, and factor-variable helper."""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import re
from typing import TYPE_CHECKING

import numpy as np

from openstata.core.errors import StataError

if TYPE_CHECKING:
    from openstata.core.dataset import Dataset


@dataclass
class ParsedCommand:
    """Structured representation of a single Stata command line."""

    raw: str
    command: str
    body: str  # portion before comma (after command name)
    options_str: str
    options: dict[str, str | bool] = field(default_factory=dict)
    capture: bool = False
    quietly: bool = False
    noisily: bool = False
    by_vars: list[str] = field(default_factory=list)
    by_sort_extra: list[str] = field(default_factory=list)
    by_sort: bool = False
    if_expr: str | None = None
    in_range: tuple[int, int] | None = None  # 1-based inclusive (start, end) resolved at runtime
    in_raw: str | None = None
    using_path: str | None = None
    weight: tuple[str, str] | None = None  # (wtype, expr)
    main_args: str = ""  # body stripped of using/if/in/weights


def split_top_level_tokens(text: str) -> list[str]:
    """Split text on whitespace outside quotes and parentheses/brackets."""
    tokens: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(text)
    in_dq = False
    c_depth = 0
    paren = 0
    brack = 0

    while i < n:
        ch = text[i]
        if text[i : i + 2] == '`"':
            c_depth += 1
            buf.append('`"')
            i += 2
            continue
        if c_depth > 0 and text[i : i + 2] == '"\'':
            c_depth -= 1
            buf.append('"\'')
            i += 2
            continue
        if c_depth == 0 and ch == '"':
            in_dq = not in_dq
            buf.append(ch)
            i += 1
            continue
        if not in_dq and c_depth == 0:
            if ch == "(":
                paren += 1
            elif ch == ")" and paren > 0:
                paren -= 1
            elif ch == "[":
                brack += 1
            elif ch == "]" and brack > 0:
                brack -= 1
            elif ch.isspace() and paren == 0 and brack == 0:
                if buf:
                    tokens.append("".join(buf))
                    buf.clear()
                i += 1
                continue
        buf.append(ch)
        i += 1
    if buf:
        tokens.append("".join(buf))
    return tokens


def strip_quotes(token: str) -> str:
    if token.startswith('`"') and token.endswith('"\''):
        return token[2:-2]
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token


def _split_comma_outside_quotes(text: str) -> tuple[str, str]:
    in_dq = False
    c_depth = 0
    paren = 0
    brack = 0
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
        ch = text[i]
        if c_depth == 0 and ch == '"':
            in_dq = not in_dq
        elif not in_dq and c_depth == 0:
            if ch == "(":
                paren += 1
            elif ch == ")" and paren > 0:
                paren -= 1
            elif ch == "[":
                brack += 1
            elif ch == "]" and brack > 0:
                brack -= 1
            elif ch == "," and paren == 0 and brack == 0:
                return text[:i].strip(), text[i + 1 :].strip()
        i += 1
    return text.strip(), ""


def parse_options(options_str: str) -> dict[str, str | bool]:
    """Parse Stata options string like 'robust cluster(id) level(95) detail'."""
    opts: dict[str, str | bool] = {}
    if not options_str.strip():
        return opts
    for tok in split_top_level_tokens(options_str):
        lp = tok.find("(")
        if lp != -1 and tok.endswith(")"):
            key = tok[:lp].strip().lower()
            val = tok[lp + 1 : -1].strip()
            opts[key] = val
        else:
            opts[tok.strip().lower()] = True
    return opts


def _parse_by_prefix(text: str) -> tuple[list[str], list[str], bool, str]:
    """Parse optional `by ...:` or `bysort ...:` prefix."""
    m = re.match(r"^(by|bys|byso|bysor|bysort)\s+([^:]+):(.*)$", text.strip(), flags=re.IGNORECASE)
    if not m:
        return [], [], False, text.strip()
    kw = m.group(1).lower()
    is_bysort = kw.startswith("bys")
    spec = m.group(2).strip()
    rest = m.group(3).strip()

    # Check for trailing `, sort` inside `by var, sort:`
    if "," in spec:
        left_s, right_s = spec.split(",", 1)
        if "sort" in right_s.lower():
            is_bysort = True
        spec = left_s.strip()

    # Handle `by g1 (s1 s2):`
    extra: list[str] = []
    lp = spec.find("(")
    if lp != -1 and spec.endswith(")"):
        extra = spec[lp + 1 : -1].split()
        spec = spec[:lp].strip()
    by_vars = spec.split()
    return by_vars, extra, is_bysort, rest


def parse_in_range(in_raw: str, nobs: int) -> tuple[int, int]:
    """Resolve Stata `in` spec like '1/10', '5', '-3/l', 'f/10' to 1-based (start, end)."""
    if nobs == 0:
        return 1, 0

    def _resolve_idx(s: str) -> int:
        sl = s.strip().lower()
        if sl in ("f", "first"):
            return 1
        if sl in ("l", "last"):
            return nobs
        v = int(sl)
        if v < 0:
            v = nobs + 1 + v
        return v

    if "/" in in_raw:
        a, b = in_raw.split("/", 1)
        start, end = _resolve_idx(a), _resolve_idx(b)
    else:
        start = end = _resolve_idx(in_raw)
    if start < 1 or end > nobs or start > end:
        raise StataError(f"obs. nos. out of range in 'in {in_raw}'", rc=198)
    return start, end


def parse_command_line(line: str, split_clauses: bool = True) -> ParsedCommand:
    """Parse a preprocessed, macro-expanded Stata command line."""
    cur = line.strip()
    capture = False
    quietly = False
    noisily = False
    by_vars: list[str] = []
    by_extra: list[str] = []
    by_sort = False

    # Prefix modifiers can be chained (`quietly capture bysort id: ...`)
    while True:
        toks = split_top_level_tokens(cur)
        if not toks:
            break
        head = toks[0].lower()
        if head in ("cap", "capt", "captu", "captur", "capture"):
            capture = True
            cur = cur[len(toks[0]) :].strip()
            continue
        if head in ("qui", "quie", "quiet", "quietl", "quietly"):
            quietly = True
            cur = cur[len(toks[0]) :].strip()
            continue
        if head in ("noi", "nois", "noisi", "noisil", "noisily"):
            noisily = True
            cur = cur[len(toks[0]) :].strip()
            continue
        b_vars, b_extra, b_sort, after_by = _parse_by_prefix(cur)
        if b_vars:
            by_vars, by_extra, by_sort = b_vars, b_extra, b_sort
            cur = after_by
            continue
        break

    # Handle `cmd: ...` like `xi:` or `version 17:`
    body_all, opts_str = _split_comma_outside_quotes(cur)
    options = parse_options(opts_str)

    toks = split_top_level_tokens(body_all)
    if not toks:
        return ParsedCommand(
            raw=line,
            command="",
            body="",
            options_str=opts_str,
            options=options,
            capture=capture,
            quietly=quietly,
            noisily=noisily,
        )

    cmd = toks[0]
    rest_body = body_all[len(cmd) :].strip()

    # Check if command contains `=` directly attached like `gen x=1`
    if "=" in cmd and not cmd.startswith("="):
        eq_pos = cmd.find("=")
        # No, `gen x=1` has space before x; only e.g. `di=1` wouldn't happen
        pass

    if_expr: str | None = None
    in_raw: str | None = None
    using_path: str | None = None
    weight: tuple[str, str] | None = None
    main_args = rest_body

    if split_clauses and cmd.lower() not in (
        "if",
        "else",
        "foreach",
        "forvalues",
        "forval",
        "while",
        "local",
        "loc",
        "global",
        "gl",
        "display",
        "di",
        "dis",
        "disp",
        "displ",
        "displa",
        "program",
        "prog",
        "syntax",
        "return",
        "ret",
        "ereturn",
        "eret",
    ):
        main_args, if_expr, in_raw, using_path, weight = _extract_clauses(rest_body)

    return ParsedCommand(
        raw=line,
        command=cmd,
        body=rest_body,
        options_str=opts_str,
        options=options,
        capture=capture,
        quietly=quietly,
        noisily=noisily,
        by_vars=by_vars,
        by_sort_extra=by_extra,
        by_sort=by_sort,
        if_expr=if_expr,
        in_raw=in_raw,
        using_path=using_path,
        weight=weight,
        main_args=main_args,
    )


def _extract_clauses(rest_body: str) -> tuple[str, str | None, str | None, str | None, tuple[str, str] | None]:
    """Extract [using ...], [if ...], [in ...], [weight=...] from command body."""
    toks = split_top_level_tokens(rest_body)
    main_toks: list[str] = []
    if_toks: list[str] = []
    in_raw: str | None = None
    using_path: str | None = None
    weight: tuple[str, str] | None = None

    mode = "main"
    i = 0
    while i < len(toks):
        t = toks[i]
        tl = t.lower()
        # Weight bracket [aw=wt] / [pw=wt] / [fw=wt] / [iw=wt]
        if t.startswith("[") and t.endswith("]") and "=" in t:
            inner = t[1:-1].strip()
            wtype, wexpr = inner.split("=", 1)
            if wtype.strip().lower() in ("aw", "aweight", "pw", "pweight", "fw", "fweight", "iw", "iweight"):
                weight = (wtype.strip().lower()[:2], wexpr.strip())
                i += 1
                continue
        if tl == "using" and mode == "main" and i + 1 < len(toks):
            using_path = strip_quotes(toks[i + 1])
            i += 2
            continue
        if tl == "if" and mode in ("main",):
            mode = "if"
            i += 1
            continue
        if tl == "in" and mode in ("main", "if") and i + 1 < len(toks):
            in_raw = toks[i + 1]
            mode = "after_in"
            i += 2
            continue
        if tl == "using" and mode in ("if", "after_in") and i + 1 < len(toks):
            using_path = strip_quotes(toks[i + 1])
            i += 2
            continue

        if mode == "main":
            main_toks.append(t)
        elif mode == "if":
            if_toks.append(t)
        else:
            main_toks.append(t)
        i += 1

    if_expr = " ".join(if_toks).strip() if if_toks else None
    return " ".join(main_toks).strip(), if_expr, in_raw, using_path, weight


def expand_varlist(
    patterns: list[str] | str,
    dataset: "Dataset",
    abbrev: bool = True,
) -> list[str]:
    """Expand Stata varlist patterns (*, ?, ~, var1-varN, abbreviations)."""
    if isinstance(patterns, str):
        patterns = split_top_level_tokens(patterns)
    cols = dataset.columns
    result: list[str] = []

    for pat in patterns:
        if pat == "_all" or pat == "*":
            for c in cols:
                if c not in result:
                    result.append(c)
            continue

        # Range var1-var2
        if "-" in pat and not pat.startswith("-") and pat.count("-") == 1:
            left, right = pat.split("-", 1)
            if left in cols and right in cols:
                i1, i2 = cols.index(left), cols.index(right)
                if i1 <= i2:
                    slice_cols = cols[i1 : i2 + 1]
                else:
                    slice_cols = cols[i2 : i1 + 1]
                for c in slice_cols:
                    if c not in result:
                        result.append(c)
                continue

        # Wildcard *, ?, ~
        if "*" in pat or "?" in pat or "~" in pat:
            fn_pat = pat.replace("~", "*")
            matched = [c for c in cols if fnmatch.fnmatchcase(c, fn_pat)]
            if not matched:
                raise StataError(f"variable {pat} not found", rc=111)
            for c in matched:
                if c not in result:
                    result.append(c)
            continue

        # Exact match
        if pat in cols:
            if pat not in result:
                result.append(pat)
            continue

        # Unambiguous abbreviation
        if abbrev:
            cands = [c for c in cols if c.startswith(pat)]
            if len(cands) == 1:
                if cands[0] not in result:
                    result.append(cands[0])
                continue
            if len(cands) > 1:
                raise StataError(f"{pat} ambiguous abbreviation", rc=111)

        raise StataError(f"variable {pat} not found", rc=111)

    return result
