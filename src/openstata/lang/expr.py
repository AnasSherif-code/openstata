"""Stata expression tokenizer, recursive-descent parser, and vectorized evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from openstata.core.errors import StataError
from openstata.core.missing import (
    MISSING_DOT,
    MISSING_NONE,
    parse_missing_literal,
    stata_compare,
)

if TYPE_CHECKING:
    from openstata.core.dataset import Dataset
    from openstata.core.state import StateStore


@dataclass
class EvalResult:
    """Result of evaluating a Stata expression across observations or as a scalar."""

    values: np.ndarray | list[str]
    missing_codes: np.ndarray | None
    is_string: bool
    is_scalar: bool = False

    def as_truth_mask(self, nobs: int) -> np.ndarray:
        """Stata boolean rule: nonzero numeric (including missing!) is True; 0 is False."""
        if self.is_string:
            raise StataError("type mismatch: string expression in boolean context", rc=109)
        vals = np.asarray(self.values, dtype=np.float64)
        mc = self.missing_codes if self.missing_codes is not None else np.zeros(len(vals), dtype=np.uint8)
        mask = (vals != 0.0) | (mc > 0)
        if self.is_scalar and nobs > 0:
            return np.full(nobs, bool(mask[0]), dtype=bool)
        return mask.astype(bool)

    def to_scalar(self) -> float | str:
        if self.is_string:
            return str(self.values[0]) if len(self.values) else ""
        if self.missing_codes is not None and self.missing_codes[0] > 0:
            return np.nan
        return float(self.values[0])


@dataclass
class _Tok:
    kind: str  # NUM, STR, MISS, IDENT, OP, LPAREN, RPAREN, LBRACK, RBRACK, COMMA, EOF
    val: str


def tokenize_expr(expr: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        # Compound string `"..."'
        if ch == "`" and i + 1 < n and expr[i + 1] == '"':
            end = expr.find('"\'', i + 2)
            if end == -1:
                raise StataError("unterminated compound string", rc=198)
            tokens.append(_Tok("STR", expr[i + 2 : end]))
            i = end + 2
            continue
        # Regular double-quoted string
        if ch == '"':
            end = expr.find('"', i + 1)
            if end == -1:
                raise StataError("unterminated string literal", rc=198)
            tokens.append(_Tok("STR", expr[i + 1 : end]))
            i = end + 1
            continue
        # Extended missing literal .a - .z or .
        if ch == "." and (i + 1 >= n or not expr[i + 1].isdigit()):
            if i + 1 < n and expr[i + 1].isalpha() and expr[i + 1].islower() and (
                i + 2 >= n or not (expr[i + 2].isalnum() or expr[i + 2] == "_")
            ):
                tokens.append(_Tok("MISS", expr[i : i + 2]))
                i += 2
                continue
            tokens.append(_Tok("MISS", "."))
            i += 1
            continue
        # Number literal (including scientific notation)
        if ch.isdigit() or (ch == "." and i + 1 < n and expr[i + 1].isdigit()):
            j = i
            has_dot = False
            while j < n and (expr[j].isdigit() or (expr[j] == "." and not has_dot)):
                if expr[j] == ".":
                    has_dot = True
                j += 1
            if j < n and expr[j] in ("e", "E"):
                k = j + 1
                if k < n and expr[k] in ("+", "-"):
                    k += 1
                if k < n and expr[k].isdigit():
                    while k < n and expr[k].isdigit():
                        k += 1
                    j = k
            tokens.append(_Tok("NUM", expr[i:j]))
            i = j
            continue
        # Two-char operators
        if expr[i : i + 2] in ("==", "!=", "~=", "<=", ">="):
            op = expr[i : i + 2]
            if op == "~=":
                op = "!="
            tokens.append(_Tok("OP", op))
            i += 2
            continue
        # Single-char operators & punctuation
        if ch in "+-*/^<>&|!~":
            op = "!" if ch == "~" else ch
            tokens.append(_Tok("OP", op))
            i += 1
            continue
        if ch == "(":
            tokens.append(_Tok("LPAREN", ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Tok("RPAREN", ch))
            i += 1
            continue
        if ch == "[":
            tokens.append(_Tok("LBRACK", ch))
            i += 1
            continue
        if ch == "]":
            tokens.append(_Tok("RBRACK", ch))
            i += 1
            continue
        if ch == ",":
            tokens.append(_Tok("COMMA", ch))
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append(_Tok("IDENT", expr[i:j]))
            i = j
            continue
        raise StataError(f"unexpected character '{ch}' in expression", rc=198)
    tokens.append(_Tok("EOF", ""))
    return tokens


class _ExprEvaluator:
    def __init__(
        self,
        tokens: list[_Tok],
        dataset: "Dataset | None",
        state: "StateStore",
        force_scalar: bool = False,
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.dataset = dataset
        self.state = state
        self.force_scalar = force_scalar
        self.nobs = 1 if force_scalar or dataset is None or dataset.nobs == 0 else dataset.nobs

    def peek(self) -> _Tok:
        return self.tokens[self.pos]

    def consume(self, kind: str | None = None) -> _Tok:
        tok = self.tokens[self.pos]
        if kind is not None and tok.kind != kind:
            raise StataError(f"syntax error in expression near '{tok.val}'", rc=198)
        self.pos += 1
        return tok

    def parse(self) -> EvalResult:
        res = self._parse_or()
        if self.peek().kind != "EOF":
            raise StataError(f"unexpected token '{self.peek().val}' in expression", rc=198)
        return res

    def _broadcast_pair(self, a: EvalResult, b: EvalResult) -> tuple[EvalResult, EvalResult]:
        if a.is_scalar and not b.is_scalar:
            a = self._expand_scalar(a, len(b.values))
        elif b.is_scalar and not a.is_scalar:
            b = self._expand_scalar(b, len(a.values))
        return a, b

    def _expand_scalar(self, r: EvalResult, n: int) -> EvalResult:
        if r.is_string:
            return EvalResult([str(r.values[0])] * n, None, True, is_scalar=False)
        val = float(r.values[0])
        mc = int(r.missing_codes[0]) if r.missing_codes is not None else 0
        return EvalResult(
            np.full(n, val, dtype=np.float64),
            np.full(n, mc, dtype=np.uint8),
            False,
            is_scalar=False,
        )

    def _parse_or(self) -> EvalResult:
        left = self._parse_and()
        while self.peek().kind == "OP" and self.peek().val == "|":
            self.consume()
            right = self._parse_and()
            left, right = self._broadcast_pair(left, right)
            lmask = left.as_truth_mask(len(left.values))
            rmask = right.as_truth_mask(len(right.values))
            out = (lmask | rmask).astype(np.float64)
            left = EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, left.is_scalar and right.is_scalar)
        return left

    def _parse_and(self) -> EvalResult:
        left = self._parse_cmp()
        while self.peek().kind == "OP" and self.peek().val == "&":
            self.consume()
            right = self._parse_cmp()
            left, right = self._broadcast_pair(left, right)
            lmask = left.as_truth_mask(len(left.values))
            rmask = right.as_truth_mask(len(right.values))
            out = (lmask & rmask).astype(np.float64)
            left = EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, left.is_scalar and right.is_scalar)
        return left

    def _parse_cmp(self) -> EvalResult:
        left = self._parse_add()
        while self.peek().kind == "OP" and self.peek().val in ("==", "!=", "<", "<=", ">", ">="):
            op = self.consume().val
            right = self._parse_add()
            left, right = self._broadcast_pair(left, right)
            if left.is_string or right.is_string:
                if not (left.is_string and right.is_string):
                    raise StataError("type mismatch in comparison", rc=109)
                lv = np.asarray(left.values, dtype=object)
                rv = np.asarray(right.values, dtype=object)
                if op == "==":
                    res = lv == rv
                elif op == "!=":
                    res = lv != rv
                elif op == "<":
                    res = lv < rv
                elif op == "<=":
                    res = lv <= rv
                elif op == ">":
                    res = lv > rv
                else:
                    res = lv >= rv
                out = res.astype(np.float64)
                left = EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, left.is_scalar and right.is_scalar)
            else:
                lmc = left.missing_codes if left.missing_codes is not None else np.zeros(len(left.values), dtype=np.uint8)
                rmc = right.missing_codes if right.missing_codes is not None else np.zeros(len(right.values), dtype=np.uint8)
                res = stata_compare(
                    np.asarray(left.values, dtype=np.float64),
                    lmc,
                    np.asarray(right.values, dtype=np.float64),
                    rmc,
                    op,
                )
                out = res.astype(np.float64)
                left = EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, left.is_scalar and right.is_scalar)
        return left

    def _parse_add(self) -> EvalResult:
        left = self._parse_mul()
        while self.peek().kind == "OP" and self.peek().val in ("+", "-"):
            op = self.consume().val
            right = self._parse_mul()
            left, right = self._broadcast_pair(left, right)
            if left.is_string or right.is_string:
                if op != "+" or not (left.is_string and right.is_string):
                    raise StataError("type mismatch in string expression", rc=109)
                vals = [str(a) + str(b) for a, b in zip(left.values, right.values)]
                left = EvalResult(vals, None, True, left.is_scalar and right.is_scalar)
            else:
                lv = np.asarray(left.values, dtype=np.float64)
                rv = np.asarray(right.values, dtype=np.float64)
                lmc = left.missing_codes if left.missing_codes is not None else np.zeros(len(lv), dtype=np.uint8)
                rmc = right.missing_codes if right.missing_codes is not None else np.zeros(len(rv), dtype=np.uint8)
                with np.errstate(all="ignore"):
                    vals = lv + rv if op == "+" else lv - rv
                mc = np.where((lmc > 0) | (rmc > 0) | ~np.isfinite(vals), MISSING_DOT, MISSING_NONE).astype(np.uint8)
                vals = np.where(mc > 0, np.nan, vals)
                left = EvalResult(vals, mc, False, left.is_scalar and right.is_scalar)
        return left

    def _parse_mul(self) -> EvalResult:
        left = self._parse_unary()
        while self.peek().kind == "OP" and self.peek().val in ("*", "/"):
            op = self.consume().val
            right = self._parse_unary()
            left, right = self._broadcast_pair(left, right)
            if left.is_string or right.is_string:
                if op == "*" and left.is_string and not right.is_string:
                    vals = [str(a) * max(0, int(b)) if np.isfinite(b) else "" for a, b in zip(left.values, right.values)]
                    left = EvalResult(vals, None, True, left.is_scalar and right.is_scalar)
                    continue
                raise StataError("type mismatch", rc=109)
            lv = np.asarray(left.values, dtype=np.float64)
            rv = np.asarray(right.values, dtype=np.float64)
            lmc = left.missing_codes if left.missing_codes is not None else np.zeros(len(lv), dtype=np.uint8)
            rmc = right.missing_codes if right.missing_codes is not None else np.zeros(len(rv), dtype=np.uint8)
            with np.errstate(all="ignore"):
                vals = lv * rv if op == "*" else np.where(rv == 0, np.nan, lv / rv)
            mc = np.where((lmc > 0) | (rmc > 0) | ~np.isfinite(vals), MISSING_DOT, MISSING_NONE).astype(np.uint8)
            vals = np.where(mc > 0, np.nan, vals)
            left = EvalResult(vals, mc, False, left.is_scalar and right.is_scalar)
        return left

    def _parse_unary(self) -> EvalResult:
        if self.peek().kind == "OP" and self.peek().val in ("+", "-", "!"):
            op = self.consume().val
            operand = self._parse_unary()
            if operand.is_string:
                raise StataError("type mismatch", rc=109)
            if op == "+":
                return operand
            if op == "-":
                vals = -np.asarray(operand.values, dtype=np.float64)
                mc = operand.missing_codes if operand.missing_codes is not None else np.zeros(len(vals), dtype=np.uint8)
                return EvalResult(vals, mc.copy(), False, operand.is_scalar)
            # Logical negation !
            mask = operand.as_truth_mask(len(operand.values))
            out = (~mask).astype(np.float64)
            return EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, operand.is_scalar)
        return self._parse_pow()

    def _parse_pow(self) -> EvalResult:
        base = self._parse_primary()
        if self.peek().kind == "OP" and self.peek().val == "^":
            self.consume()
            exp = self._parse_unary()
            base, exp = self._broadcast_pair(base, exp)
            if base.is_string or exp.is_string:
                raise StataError("type mismatch", rc=109)
            bv = np.asarray(base.values, dtype=np.float64)
            ev = np.asarray(exp.values, dtype=np.float64)
            bmc = base.missing_codes if base.missing_codes is not None else np.zeros(len(bv), dtype=np.uint8)
            emc = exp.missing_codes if exp.missing_codes is not None else np.zeros(len(ev), dtype=np.uint8)
            with np.errstate(all="ignore"):
                vals = np.power(bv, ev)
            mc = np.where((bmc > 0) | (emc > 0) | ~np.isfinite(vals), MISSING_DOT, MISSING_NONE).astype(np.uint8)
            vals = np.where(mc > 0, np.nan, vals)
            return EvalResult(vals, mc, False, base.is_scalar and exp.is_scalar)
        return base

    def _parse_primary(self) -> EvalResult:
        tok = self.peek()
        if tok.kind == "NUM":
            self.consume()
            return EvalResult(
                np.array([float(tok.val)], dtype=np.float64),
                np.array([MISSING_NONE], dtype=np.uint8),
                False,
                is_scalar=True,
            )
        if tok.kind == "MISS":
            self.consume()
            code = parse_missing_literal(tok.val) or MISSING_DOT
            return EvalResult(
                np.array([np.nan], dtype=np.float64),
                np.array([code], dtype=np.uint8),
                False,
                is_scalar=True,
            )
        if tok.kind == "STR":
            self.consume()
            return EvalResult([tok.val], None, True, is_scalar=True)
        if tok.kind == "LPAREN":
            self.consume()
            inner = self._parse_or()
            self.consume("RPAREN")
            return inner
        if tok.kind == "IDENT":
            name = self.consume().val
            # Function or r()/e()/c()/s() call
            if self.peek().kind == "LPAREN":
                return self._parse_call(name)
            # Subscript var[idx] or matrix[r, c]
            if self.peek().kind == "LBRACK":
                return self._parse_subscript(name)
            return self._resolve_ident(name)
        raise StataError(f"invalid expression syntax near '{tok.val}'", rc=198)

    def _resolve_ident(self, name: str) -> EvalResult:
        if name == "_n":
            if self.force_scalar or self.dataset is None or self.dataset.nobs == 0:
                return EvalResult(np.array([1.0]), np.array([0], dtype=np.uint8), False, is_scalar=True)
            arr = np.arange(1, self.dataset.nobs + 1, dtype=np.float64)
            return EvalResult(arr, np.zeros(len(arr), dtype=np.uint8), False, is_scalar=False)
        if name == "_N":
            n_val = float(self.dataset.nobs if self.dataset is not None else 0)
            return EvalResult(np.array([n_val]), np.array([0], dtype=np.uint8), False, is_scalar=True)
        if name == "_pi":
            return EvalResult(np.array([math.pi]), np.array([0], dtype=np.uint8), False, is_scalar=True)
        if name == "_rc":
            rc_val = float(self.state.r.scalars.get("_rc", 0.0))
            return EvalResult(np.array([rc_val]), np.array([0], dtype=np.uint8), False, is_scalar=True)

        if self.dataset is not None and name in self.dataset.df.columns and not self.force_scalar:
            if self.dataset.is_string(name):
                return EvalResult(self.dataset.df[name].astype(str).tolist(), None, True, is_scalar=False)
            vals, mc = self.dataset.get_numeric(name)
            return EvalResult(vals.copy(), mc.copy(), False, is_scalar=False)

        if name in self.state.scalars:
            v = float(self.state.scalars[name])
            mc = MISSING_DOT if np.isnan(v) else MISSING_NONE
            return EvalResult(np.array([v]), np.array([mc], dtype=np.uint8), False, is_scalar=True)

        raise StataError(f"{name} not found", rc=111)

    def _parse_subscript(self, name: str) -> EvalResult:
        self.consume("LBRACK")
        idx_res = self._parse_or()
        if self.peek().kind == "COMMA":
            self.consume()
            col_res = self._parse_or()
            self.consume("RBRACK")
            mat = self.state.matrices.get(name)
            if mat is None:
                raise StataError(f"matrix {name} not found", rc=111)
            r = int(idx_res.to_scalar()) - 1
            c = int(col_res.to_scalar()) - 1
            if not (0 <= r < mat.shape[0] and 0 <= c < mat.shape[1]):
                raise StataError("matrix subscript out of range", rc=503)
            v = float(mat[r, c])
            return EvalResult(np.array([v]), np.array([MISSING_DOT if np.isnan(v) else 0], dtype=np.uint8), False, is_scalar=True)

        self.consume("RBRACK")
        if self.dataset is None or name not in self.dataset.df.columns:
            raise StataError(f"{name} not found", rc=111)
        n = self.dataset.nobs
        if idx_res.is_scalar:
            idx_res = self._expand_scalar(idx_res, n)
        idx_arr = np.asarray(idx_res.values, dtype=np.float64)
        valid = np.isfinite(idx_arr) & (idx_arr >= 1) & (idx_arr <= n)
        int_idx = np.where(valid, idx_arr - 1, 0).astype(np.int64)

        if self.dataset.is_string(name):
            src = self.dataset.df[name].astype(str).tolist()
            out = [src[i] if ok else "" for i, ok in zip(int_idx, valid)]
            return EvalResult(out, None, True, is_scalar=False)

        vals, mc = self.dataset.get_numeric(name)
        out_v = np.where(valid, vals[int_idx], np.nan)
        out_mc = np.where(valid, mc[int_idx], MISSING_DOT).astype(np.uint8)
        return EvalResult(out_v, out_mc, False, is_scalar=False)

    def _parse_call(self, name: str) -> EvalResult:
        self.consume("LPAREN")
        # Special stores r(x), e(x), c(x), s(x)
        if name in ("r", "e", "c", "s"):
            if self.peek().kind != "IDENT":
                raise StataError(f"invalid argument to {name}()", rc=198)
            key = self.consume("IDENT").val
            self.consume("RPAREN")
            if name == "c":
                val = self.state.get_c_value(key, self.dataset)
            else:
                val = getattr(self.state, name).get_value(key)
            if val is None:
                return EvalResult(np.array([np.nan]), np.array([MISSING_DOT], dtype=np.uint8), False, is_scalar=True)
            if isinstance(val, str):
                return EvalResult([val], None, True, is_scalar=True)
            if isinstance(val, (int, float, np.floating, np.integer)):
                fv = float(val)
                mc = MISSING_DOT if np.isnan(fv) else MISSING_NONE
                return EvalResult(np.array([fv]), np.array([mc], dtype=np.uint8), False, is_scalar=True)
            raise StataError(f"{name}({key}) is a matrix", rc=198)

        args: list[EvalResult] = []
        if self.peek().kind != "RPAREN":
            args.append(self._parse_or())
            while self.peek().kind == "COMMA":
                self.consume()
                args.append(self._parse_or())
        self.consume("RPAREN")
        return _eval_builtin_func(name, args, self.nobs)


def _eval_builtin_func(name: str, args: list[EvalResult], nobs: int) -> EvalResult:
    fn = name.lower()

    def _all_scalar() -> bool:
        return all(a.is_scalar for a in args)

    def _vec(a: EvalResult) -> tuple[np.ndarray, np.ndarray]:
        if a.is_string:
            raise StataError(f"type mismatch in {fn}()", rc=109)
        if a.is_scalar:
            v = np.full(nobs, float(a.values[0]), dtype=np.float64)
            mc = np.full(nobs, int(a.missing_codes[0]) if a.missing_codes is not None else 0, dtype=np.uint8)
            return v, mc
        mc = a.missing_codes if a.missing_codes is not None else np.zeros(len(a.values), dtype=np.uint8)
        return np.asarray(a.values, dtype=np.float64), mc

    def _svec(a: EvalResult) -> list[str]:
        if not a.is_string:
            raise StataError(f"type mismatch in {fn}()", rc=109)
        if a.is_scalar:
            return [str(a.values[0])] * nobs
        return [str(x) for x in a.values]

    def _num_out(vals: np.ndarray, extra_miss: np.ndarray | None = None, scalar: bool | None = None) -> EvalResult:
        is_sc = _all_scalar() if scalar is None else scalar
        mc = np.where(~np.isfinite(vals), MISSING_DOT, MISSING_NONE).astype(np.uint8)
        if extra_miss is not None:
            mc = np.where(extra_miss > 0, MISSING_DOT, mc).astype(np.uint8)
        out_v = np.where(mc > 0, np.nan, vals)
        if is_sc:
            return EvalResult(out_v[:1], mc[:1], False, is_scalar=True)
        return EvalResult(out_v, mc, False, is_scalar=False)

    # Unary math
    unary_map = {
        "abs": np.abs,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "ln": np.log,
        "log": np.log,
        "log10": np.log10,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "floor": np.floor,
        "ceil": np.ceil,
        "int": np.trunc,
        "sign": np.sign,
    }
    if fn in unary_map:
        if len(args) != 1:
            raise StataError(f"{fn}() expects 1 argument", rc=198)
        v, mc = _vec(args[0])
        with np.errstate(all="ignore"):
            res = unary_map[fn](v)
        return _num_out(res, mc)

    if fn == "round":
        if len(args) not in (1, 2):
            raise StataError("round() expects 1 or 2 arguments", rc=198)
        v, mc = _vec(args[0])
        unit = 1.0 if len(args) == 1 else float(args[1].to_scalar())
        with np.errstate(all="ignore"):
            res = np.round(v / unit) * unit
        return _num_out(res, mc)

    if fn == "mod":
        if len(args) != 2:
            raise StataError("mod() expects 2 arguments", rc=198)
        a, amc = _vec(args[0])
        b, bmc = _vec(args[1])
        with np.errstate(all="ignore"):
            res = np.mod(a, b)
        return _num_out(res, (amc > 0) | (bmc > 0) | (b == 0))

    if fn in ("missing", "mi"):
        if not args:
            raise StataError(f"{fn}() expects at least 1 argument", rc=198)
        is_sc = _all_scalar()
        length = 1 if is_sc else nobs
        mask = np.zeros(length, dtype=bool)
        for a in args:
            if a.is_string:
                sv = [str(a.values[0])] * length if a.is_scalar else [str(x) for x in a.values]
                mask |= np.array([s == "" for s in sv], dtype=bool)
            else:
                mc = a.missing_codes if a.missing_codes is not None else np.zeros(len(a.values), dtype=np.uint8)
                if a.is_scalar:
                    mask |= bool(mc[0] > 0)
                else:
                    mask |= mc > 0
        out = mask.astype(np.float64)
        return EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, is_scalar=is_sc)

    if fn in ("max", "min"):
        if not args:
            raise StataError(f"{fn}() expects >= 1 argument", rc=198)
        stacked = []
        for a in args:
            v, mc = _vec(a)
            stacked.append(np.where(mc > 0, np.nan, v))
        mat = np.vstack(stacked)
        with np.errstate(all="ignore"):
            res = np.nanmax(mat, axis=0) if fn == "max" else np.nanmin(mat, axis=0)
        return _num_out(res)

    if fn == "sum":
        if len(args) != 1:
            raise StataError("sum() expects 1 argument (running sum)", rc=198)
        v, mc = _vec(args[0])
        clean = np.where(mc > 0, 0.0, v)
        res = np.cumsum(clean)
        return EvalResult(res, np.zeros(len(res), dtype=np.uint8), False, is_scalar=False)

    if fn == "inrange":
        if len(args) != 3:
            raise StataError("inrange(z, a, b) expects 3 arguments", rc=198)
        z, zmc = _vec(args[0])
        a, amc = _vec(args[1])
        b, bmc = _vec(args[2])
        ok = (zmc == 0) & (amc == 0) & (bmc == 0) & (z >= a) & (z <= b)
        out = ok.astype(np.float64)
        is_sc = _all_scalar()
        if is_sc:
            out = out[:1]
        return EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, is_scalar=is_sc)

    if fn == "inlist":
        if len(args) < 2:
            raise StataError("inlist() expects >= 2 arguments", rc=198)
        target = args[0]
        is_sc = _all_scalar()
        length = 1 if is_sc else nobs
        mask = np.zeros(length, dtype=bool)
        if target.is_string:
            tv = [str(target.values[0])] * length if target.is_scalar else [str(x) for x in target.values]
            for opt in args[1:]:
                ov = [str(opt.values[0])] * length if opt.is_scalar else _svec(opt)
                mask |= np.array([x == y for x, y in zip(tv, ov)], dtype=bool)
        else:
            tv, tmc = _vec(target)
            if is_sc:
                tv, tmc = tv[:1], tmc[:1]
            for opt in args[1:]:
                ov, omc = _vec(opt)
                if is_sc:
                    ov, omc = ov[:1], omc[:1]
                mask |= stata_compare(tv, tmc, ov, omc, "==")
        out = mask.astype(np.float64)
        return EvalResult(out, np.zeros(len(out), dtype=np.uint8), False, is_scalar=is_sc)

    if fn == "cond":
        if len(args) not in (3, 4):
            raise StataError("cond(c, a, b [, d]) expects 3 or 4 arguments", rc=198)
        cond_r = args[0]
        is_sc = _all_scalar()
        length = 1 if is_sc else nobs
        mask = cond_r.as_truth_mask(length)[:length]
        a, b = args[1], args[2]
        if a.is_string or b.is_string:
            if not (a.is_string and b.is_string):
                raise StataError("type mismatch in cond()", rc=109)
            av = [str(a.values[0])] * length if a.is_scalar else _svec(a)
            bv = [str(b.values[0])] * length if b.is_scalar else _svec(b)
            out_s = [x if m else y for m, x, y in zip(mask, av, bv)]
            return EvalResult(out_s, None, True, is_scalar=is_sc)
        av, amc = _vec(a)
        bv, bmc = _vec(b)
        if is_sc:
            av, amc = av[:1], amc[:1]
            bv, bmc = bv[:1], bmc[:1]
        out_v = np.where(mask, av, bv)
        out_mc = np.where(mask, amc, bmc).astype(np.uint8)
        if len(args) == 4:
            cmask = (cond_r.missing_codes[:length] > 0) if cond_r.missing_codes is not None else np.zeros(length, dtype=bool)
            dv, dmc = _vec(args[3])
            if is_sc:
                dv, dmc = dv[:1], dmc[:1]
            out_v = np.where(cmask, dv, out_v)
            out_mc = np.where(cmask, dmc, out_mc).astype(np.uint8)
        return EvalResult(out_v, out_mc, False, is_scalar=is_sc)

    if fn == "runiform":
        rng = np.random.default_rng()
        if len(args) == 0:
            vals = rng.random(nobs)
        elif len(args) == 2:
            lo = float(args[0].to_scalar())
            hi = float(args[1].to_scalar())
            vals = rng.uniform(lo, hi, size=nobs)
        else:
            raise StataError("runiform() expects 0 or 2 arguments", rc=198)
        return EvalResult(vals, np.zeros(nobs, dtype=np.uint8), False, is_scalar=False)

    if fn == "rnormal":
        rng = np.random.default_rng()
        mean = float(args[0].to_scalar()) if len(args) >= 1 else 0.0
        sd = float(args[1].to_scalar()) if len(args) >= 2 else 1.0
        vals = rng.normal(mean, sd, size=nobs)
        return EvalResult(vals, np.zeros(nobs, dtype=np.uint8), False, is_scalar=False)

    if fn == "runiformint":
        if len(args) != 2:
            raise StataError("runiformint(a, b) expects 2 arguments", rc=198)
        lo = int(args[0].to_scalar())
        hi = int(args[1].to_scalar())
        vals = np.random.default_rng().integers(lo, hi + 1, size=nobs).astype(np.float64)
        return EvalResult(vals, np.zeros(nobs, dtype=np.uint8), False, is_scalar=False)

    # String functions
    if fn in ("strlen", "ustrlen"):
        sv = _svec(args[0])
        res = np.array([len(s) for s in sv], dtype=np.float64)
        return _num_out(res, scalar=args[0].is_scalar)

    if fn in ("upper", "strupper"):
        sv = _svec(args[0])
        out = [s.upper() for s in sv]
        return EvalResult(out[:1] if args[0].is_scalar else out, None, True, is_scalar=args[0].is_scalar)

    if fn in ("lower", "strlower"):
        sv = _svec(args[0])
        out = [s.lower() for s in sv]
        return EvalResult(out[:1] if args[0].is_scalar else out, None, True, is_scalar=args[0].is_scalar)

    if fn in ("trim", "strtrim"):
        sv = _svec(args[0])
        out = [s.strip() for s in sv]
        return EvalResult(out[:1] if args[0].is_scalar else out, None, True, is_scalar=args[0].is_scalar)

    if fn == "ltrim":
        sv = _svec(args[0])
        out = [s.lstrip() for s in sv]
        return EvalResult(out[:1] if args[0].is_scalar else out, None, True, is_scalar=args[0].is_scalar)

    if fn == "rtrim":
        sv = _svec(args[0])
        out = [s.rstrip() for s in sv]
        return EvalResult(out[:1] if args[0].is_scalar else out, None, True, is_scalar=args[0].is_scalar)

    if fn in ("substr", "usubstr"):
        if len(args) != 3:
            raise StataError(f"{fn}(s, n1, n2) expects 3 arguments", rc=198)
        sv = _svec(args[0])
        n1 = int(args[1].to_scalar())
        n2 = int(args[2].to_scalar())
        out = []
        for s in sv:
            start = (n1 - 1) if n1 > 0 else max(0, len(s) + n1)
            out.append(s[start : start + max(0, n2)])
        is_sc = _all_scalar()
        return EvalResult(out[:1] if is_sc else out, None, True, is_scalar=is_sc)

    if fn == "strpos":
        if len(args) != 2:
            raise StataError("strpos(s1, s2) expects 2 arguments", rc=198)
        s1 = _svec(args[0])
        s2 = _svec(args[1])
        res = np.array([a.find(b) + 1 for a, b in zip(s1, s2)], dtype=np.float64)
        return _num_out(res, scalar=_all_scalar())

    if fn == "subinstr":
        if len(args) != 4:
            raise StataError("subinstr(s, old, new, n) expects 4 arguments", rc=198)
        sv = _svec(args[0])
        old = str(args[1].to_scalar())
        new = str(args[2].to_scalar())
        cnt = args[3].to_scalar()
        out = [s.replace(old, new) if np.isnan(cnt) else s.replace(old, new, int(cnt)) for s in sv]
        is_sc = _all_scalar()
        return EvalResult(out[:1] if is_sc else out, None, True, is_scalar=is_sc)

    if fn == "word":
        if len(args) != 2:
            raise StataError("word(s, n) expects 2 arguments", rc=198)
        sv = _svec(args[0])
        widx = int(args[1].to_scalar())
        out = []
        for s in sv:
            parts = s.split()
            if 1 <= widx <= len(parts):
                out.append(parts[widx - 1])
            elif widx < 0 and abs(widx) <= len(parts):
                out.append(parts[widx])
            else:
                out.append("")
        is_sc = _all_scalar()
        return EvalResult(out[:1] if is_sc else out, None, True, is_scalar=is_sc)

    if fn == "wordcount":
        sv = _svec(args[0])
        res = np.array([len(s.split()) for s in sv], dtype=np.float64)
        return _num_out(res, scalar=args[0].is_scalar)

    if fn == "real":
        sv = _svec(args[0])
        vals = []
        for s in sv:
            try:
                vals.append(float(s.strip()))
            except ValueError:
                vals.append(np.nan)
        return _num_out(np.array(vals, dtype=np.float64), scalar=args[0].is_scalar)

    if fn == "string":
        v, mc = _vec(args[0])
        out = [
            "." if m > 0 else (str(int(x)) if float(int(x)) == x else f"{x:g}")
            for x, m in zip(v, mc)
        ]
        return EvalResult(out[:1] if args[0].is_scalar else out, None, True, is_scalar=args[0].is_scalar)

    if fn == "regexm":
        if len(args) != 2:
            raise StataError("regexm(s, re) expects 2 arguments", rc=198)
        sv = _svec(args[0])
        pat = re.compile(str(args[1].to_scalar()))
        res = np.array([1.0 if pat.search(s) else 0.0 for s in sv], dtype=np.float64)
        return _num_out(res, scalar=_all_scalar())

    if fn == "regexr":
        if len(args) != 3:
            raise StataError("regexr(s, re, rep) expects 3 arguments", rc=198)
        sv = _svec(args[0])
        pat = re.compile(str(args[1].to_scalar()))
        rep = str(args[2].to_scalar())
        out = [pat.sub(rep, s, count=1) for s in sv]
        is_sc = _all_scalar()
        return EvalResult(out[:1] if is_sc else out, None, True, is_scalar=is_sc)

    if fn == "td":
        s = str(args[0].to_scalar())
        dt = datetime.strptime(s, "%d%b%Y")
        stata_epoch = datetime(1960, 1, 1)
        days = float((dt - stata_epoch).days)
        return EvalResult(np.array([days]), np.array([0], dtype=np.uint8), False, is_scalar=True)

    raise StataError(f"unknown function {name}()", rc=133)


def eval_expression(
    expr: str,
    dataset: "Dataset | None",
    state: "StateStore",
    force_scalar: bool = False,
) -> EvalResult:
    """Evaluate a Stata expression string."""
    tokens = tokenize_expr(expr)
    return _ExprEvaluator(tokens, dataset, state, force_scalar=force_scalar).parse()
