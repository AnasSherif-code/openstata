"""Clean-room Stata .dta (format 118 / 119) binary reader and writer.

Supports extended missing values (., .a - .z), variable/value labels,
display formats, fixed strings (str1..str2045), strL, and characteristics.
Falls back to pandas.read_stata for legacy formats (<118).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import struct
from typing import Any

import numpy as np
import pandas as pd

from openstata.core.dataset import Dataset, ValueLabelTable, VariableMeta
from openstata.core.errors import StataError
from openstata.core.missing import MISSING_DOT, MISSING_NONE

TYPE_STRL = 32768
TYPE_DOUBLE = 65526
TYPE_FLOAT = 65527
TYPE_LONG = 65528
TYPE_INT = 65529
TYPE_BYTE = 65530

CODE_TO_TYPE = {
    TYPE_STRL: "strL",
    TYPE_DOUBLE: "double",
    TYPE_FLOAT: "float",
    TYPE_LONG: "long",
    TYPE_INT: "int",
    TYPE_BYTE: "byte",
}
TYPE_TO_CODE = {v: k for k, v in CODE_TO_TYPE.items()}

# Base missing thresholds in IEEE 754 / integer representations
_F64_BASE = 0x7FE0000000000000
_F64_STEP = 0x0000010000000000
_F32_BASE = 0x7F000000
_F32_STEP = 0x00000800
_I32_BASE = 2147483621
_I16_BASE = 32741
_I8_BASE = 101


def _decode_cstr(raw: bytes) -> str:
    nul = raw.find(b"\x00")
    if nul != -1:
        raw = raw[:nul]
    return raw.decode("utf-8", errors="replace")


def _encode_cstr(text: str, width: int) -> bytes:
    b = text.encode("utf-8", errors="replace")[: width - 1]
    return b + b"\x00" * (width - len(b))


def _read_tag(buf: bytes, pos: int, tag: bytes) -> int:
    if buf[pos : pos + len(tag)] != tag:
        raise StataError(f"invalid .dta structure: expected {tag!r}", rc=610)
    return pos + len(tag)


def read_dta(path: str | Path) -> Dataset:
    """Read a Stata .dta file into an OpenStata Dataset."""
    p = Path(path)
    if not p.exists():
        raise StataError(f"file {p} not found", rc=601)

    raw = p.read_bytes()
    if not raw.startswith(b"<stata_dta>"):
        # Legacy .dta (< 118) fallback via pandas
        return _read_legacy_dta(p)

    pos = len(b"<stata_dta>")
    pos = _read_tag(raw, pos, b"<header>")
    pos = _read_tag(raw, pos, b"<release>")
    rel_end = raw.find(b"</release>", pos)
    release = int(raw[pos:rel_end].decode("ascii"))
    if release not in (118, 119):
        return _read_legacy_dta(p)
    pos = rel_end + len(b"</release>")

    pos = _read_tag(raw, pos, b"<byteorder>")
    bo = raw[pos : pos + 3]
    if bo == b"LSF":
        endian = "<"
    elif bo == b"MSF":
        endian = ">"
    else:
        raise StataError(f"unsupported byteorder {bo!r}", rc=610)
    pos += 3
    pos = _read_tag(raw, pos, b"</byteorder>")

    pos = _read_tag(raw, pos, b"<K>")
    if release == 118:
        (nvars,) = struct.unpack_from(f"{endian}H", raw, pos)
        pos += 2
    else:
        (nvars,) = struct.unpack_from(f"{endian}I", raw, pos)
        pos += 4
    pos = _read_tag(raw, pos, b"</K>")

    pos = _read_tag(raw, pos, b"<N>")
    (nobs,) = struct.unpack_from(f"{endian}Q", raw, pos)
    pos += 8
    pos = _read_tag(raw, pos, b"</N>")

    pos = _read_tag(raw, pos, b"<label>")
    (lbl_len,) = struct.unpack_from(f"{endian}H", raw, pos)
    pos += 2
    data_label = raw[pos : pos + lbl_len].decode("utf-8", errors="replace")
    pos += lbl_len
    pos = _read_tag(raw, pos, b"</label>")

    pos = _read_tag(raw, pos, b"<timestamp>")
    ts_len = raw[pos]
    pos += 1 + ts_len
    pos = _read_tag(raw, pos, b"</timestamp>")
    pos = _read_tag(raw, pos, b"</header>")

    # <map>
    pos = _read_tag(raw, pos, b"<map>")
    offsets = struct.unpack_from(f"{endian}14Q", raw, pos)

    # <variable_types>
    pos = offsets[2]
    pos = _read_tag(raw, pos, b"<variable_types>")
    type_codes = list(struct.unpack_from(f"{endian}{nvars}H", raw, pos))

    # <varnames>
    pos = offsets[3] + len(b"<varnames>")
    varnames = [_decode_cstr(raw[pos + i * 129 : pos + (i + 1) * 129]) for i in range(nvars)]

    # <sortlist>
    pos = offsets[4] + len(b"<sortlist>")
    if release == 118:
        slist_raw = struct.unpack_from(f"{endian}{nvars + 1}H", raw, pos)
    else:
        slist_raw = struct.unpack_from(f"{endian}{nvars + 1}I", raw, pos)
    sort_by = []
    for idx in slist_raw:
        if idx == 0:
            break
        if 1 <= idx <= nvars:
            sort_by.append(varnames[idx - 1])

    # <formats>
    pos = offsets[5] + len(b"<formats>")
    formats = [_decode_cstr(raw[pos + i * 57 : pos + (i + 1) * 57]) for i in range(nvars)]

    # <value_label_names>
    pos = offsets[6] + len(b"<value_label_names>")
    vlabel_names = [_decode_cstr(raw[pos + i * 129 : pos + (i + 1) * 129]) for i in range(nvars)]

    # <variable_labels>
    pos = offsets[7] + len(b"<variable_labels>")
    var_labels = [_decode_cstr(raw[pos + i * 321 : pos + (i + 1) * 321]) for i in range(nvars)]

    # <characteristics>
    characteristics: dict[str, dict[str, str]] = {}
    pos = offsets[8] + len(b"<characteristics>")
    while raw[pos : pos + 4] == b"<ch>":
        pos += 4
        (ch_len,) = struct.unpack_from(f"{endian}I", raw, pos)
        pos += 4
        ch_block = raw[pos : pos + ch_len]
        pos += ch_len
        pos = _read_tag(raw, pos, b"</ch>")
        if len(ch_block) >= 258:
            c_var = _decode_cstr(ch_block[:129])
            c_name = _decode_cstr(ch_block[129:258])
            c_val = _decode_cstr(ch_block[258:])
            characteristics.setdefault(c_var, {})[c_name] = c_val

    # <strls> (parse first so strL column data can resolve GSO references)
    strl_map: dict[tuple[int, int], str] = {}
    pos = offsets[10] + len(b"<strls>")
    while raw[pos : pos + 3] == b"GSO":
        pos += 3
        v, o, t, slen = struct.unpack_from(f"{endian}IQBI", raw, pos)
        pos += 17
        payload = raw[pos : pos + slen]
        pos += slen
        if t == 130:
            strl_map[(v, o)] = _decode_cstr(payload)
        else:
            strl_map[(v, o)] = payload.decode("utf-8", errors="replace")

    # <data>
    pos = offsets[9] + len(b"<data>")
    row_fmt_parts: list[tuple[int, str]] = []  # (byte_width, kind)
    for tc in type_codes:
        if 1 <= tc <= 2045:
            row_fmt_parts.append((tc, "str"))
        elif tc == TYPE_STRL:
            row_fmt_parts.append((8, "strl"))
        elif tc == TYPE_DOUBLE:
            row_fmt_parts.append((8, "f64"))
        elif tc == TYPE_FLOAT:
            row_fmt_parts.append((4, "f32"))
        elif tc == TYPE_LONG:
            row_fmt_parts.append((4, "i32"))
        elif tc == TYPE_INT:
            row_fmt_parts.append((2, "i16"))
        elif tc == TYPE_BYTE:
            row_fmt_parts.append((1, "i8"))
        else:
            raise StataError(f"unsupported .dta variable type code {tc}", rc=610)

    row_size = sum(w for w, _ in row_fmt_parts)
    data_bytes = raw[pos : pos + nobs * row_size]

    columns: dict[str, Any] = {}
    missing_codes: dict[str, np.ndarray] = {}
    meta: dict[str, VariableMeta] = {}

    col_offset = 0
    for col_idx, (name, tc, (width, kind)) in enumerate(zip(varnames, type_codes, row_fmt_parts)):
        stype = f"str{tc}" if kind == "str" else CODE_TO_TYPE[tc]
        meta[name] = VariableMeta(
            name=name,
            stata_type=stype,
            fmt=formats[col_idx],
            label=var_labels[col_idx],
            value_label=vlabel_names[col_idx],
        )

        if nobs == 0:
            if kind in ("str", "strl"):
                columns[name] = pd.Series([], dtype="object")
            else:
                columns[name] = np.empty(0, dtype=np.float64)
                missing_codes[name] = np.empty(0, dtype=np.uint8)
            col_offset += width
            continue

        if kind == "str":
            vals = [
                _decode_cstr(data_bytes[r * row_size + col_offset : r * row_size + col_offset + width])
                for r in range(nobs)
            ]
            columns[name] = vals
        elif kind == "strl":
            vals = []
            for r in range(nobs):
                cell = data_bytes[r * row_size + col_offset : r * row_size + col_offset + 8]
                if endian == "<":
                    v = int.from_bytes(cell[0:2], "little")
                    o = int.from_bytes(cell[2:8], "little")
                else:
                    v = int.from_bytes(cell[0:2], "big")
                    o = int.from_bytes(cell[2:8], "big")
                vals.append(strl_map.get((v, o), "") if (v or o) else "")
            columns[name] = vals
        else:
            # Extract strided column via numpy buffer view
            arr_u8 = np.frombuffer(data_bytes, dtype=np.uint8).reshape(nobs, row_size)
            col_bytes = np.ascontiguousarray(arr_u8[:, col_offset : col_offset + width]).tobytes()
            mc = np.zeros(nobs, dtype=np.uint8)
            if kind == "f64":
                u64 = np.frombuffer(col_bytes, dtype=f"{endian}u8")
                f64 = np.frombuffer(col_bytes, dtype=f"{endian}f8").astype(np.float64, copy=True)
                is_miss = (u64 >= _F64_BASE) & ((u64 >> 63) == 0)
                if np.any(is_miss):
                    codes = ((u64[is_miss] - _F64_BASE) // _F64_STEP) + 1
                    mc[is_miss] = np.clip(codes, 1, 27).astype(np.uint8)
                    f64[is_miss] = np.nan
                columns[name] = f64
            elif kind == "f32":
                u32 = np.frombuffer(col_bytes, dtype=f"{endian}u4")
                f32 = np.frombuffer(col_bytes, dtype=f"{endian}f4").astype(np.float64)
                is_miss = (u32 >= _F32_BASE) & ((u32 >> 31) == 0)
                if np.any(is_miss):
                    codes = ((u32[is_miss] - _F32_BASE) // _F32_STEP) + 1
                    mc[is_miss] = np.clip(codes, 1, 27).astype(np.uint8)
                    f32[is_miss] = np.nan
                columns[name] = f32
            elif kind == "i32":
                raw_i = np.frombuffer(col_bytes, dtype=f"{endian}i4").astype(np.float64)
                is_miss = raw_i >= _I32_BASE
                if np.any(is_miss):
                    mc[is_miss] = np.clip((raw_i[is_miss] - _I32_BASE + 1).astype(np.int64), 1, 27).astype(np.uint8)
                    raw_i[is_miss] = np.nan
                columns[name] = raw_i
            elif kind == "i16":
                raw_i = np.frombuffer(col_bytes, dtype=f"{endian}i2").astype(np.float64)
                is_miss = raw_i >= _I16_BASE
                if np.any(is_miss):
                    mc[is_miss] = np.clip((raw_i[is_miss] - _I16_BASE + 1).astype(np.int64), 1, 27).astype(np.uint8)
                    raw_i[is_miss] = np.nan
                columns[name] = raw_i
            elif kind == "i8":
                raw_i = np.frombuffer(col_bytes, dtype=f"{endian}i1").astype(np.float64)
                is_miss = raw_i >= _I8_BASE
                if np.any(is_miss):
                    mc[is_miss] = np.clip((raw_i[is_miss] - _I8_BASE + 1).astype(np.int64), 1, 27).astype(np.uint8)
                    raw_i[is_miss] = np.nan
                columns[name] = raw_i
            missing_codes[name] = mc

        col_offset += width

    # <value_labels>
    value_labels: dict[str, ValueLabelTable] = {}
    pos = offsets[11] + len(b"<value_labels>")
    while raw[pos : pos + 5] == b"<lbl>":
        pos += 5
        (lbl_block_len,) = struct.unpack_from(f"{endian}I", raw, pos)
        pos += 4
        block = raw[pos : pos + lbl_block_len]
        pos += lbl_block_len
        pos = _read_tag(raw, pos, b"</lbl>")
        labname = _decode_cstr(block[:129])
        # 3 padding bytes after labname (total 132)
        n_entries, txtlen = struct.unpack_from(f"{endian}II", block, 132)
        offsets_arr = struct.unpack_from(f"{endian}{n_entries}I", block, 140)
        vals_arr = struct.unpack_from(f"{endian}{n_entries}i", block, 140 + 4 * n_entries)
        txt_bytes = block[140 + 8 * n_entries : 140 + 8 * n_entries + txtlen]
        mapping: dict[int, str] = {}
        for v_int, t_off in zip(vals_arr, offsets_arr):
            mapping[int(v_int)] = _decode_cstr(txt_bytes[t_off:])
        value_labels[labname] = ValueLabelTable(name=labname, mapping=mapping)

    df = pd.DataFrame(columns)
    ds = Dataset(
        df=df,
        meta=meta,
        missing_codes=missing_codes,
        value_labels=value_labels,
        data_label=data_label,
        characteristics=characteristics,
        sort_by=sort_by,
        source_path=str(p),
    )
    ds.dirty = False
    return ds


def _read_legacy_dta(p: Path) -> Dataset:
    reader = pd.read_stata(p, convert_categoricals=False, iterator=True)
    try:
        df = reader.read()
        vlabels_raw = reader.value_labels()
        var_labels = reader.variable_labels()
        data_label = reader.data_label or ""
        fmt_list = getattr(reader, "fmtlist", [])
        lbl_list = getattr(reader, "lbllist", [])
    finally:
        reader.close()

    meta: dict[str, VariableMeta] = {}
    for i, col in enumerate(df.columns):
        fmt = fmt_list[i] if i < len(fmt_list) else "%9.0g"
        vlbl = lbl_list[i] if i < len(lbl_list) else ""
        stype = "str244" if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object else "double"
        meta[col] = VariableMeta(
            name=col,
            stata_type=stype,
            fmt=fmt,
            label=var_labels.get(col, ""),
            value_label=vlbl,
        )
    vlabels = {k: ValueLabelTable(name=k, mapping=dict(v)) for k, v in vlabels_raw.items()}
    ds = Dataset(df=df, meta=meta, value_labels=vlabels, data_label=data_label, source_path=str(p))
    ds.dirty = False
    return ds


def write_dta(dataset: Dataset, path: str | Path, replace: bool = False) -> None:
    """Write an OpenStata Dataset to a Stata .dta format 118 file."""
    p = Path(path)
    if p.exists() and not replace:
        raise StataError(f"file {p} already exists", rc=602)
    p.parent.mkdir(parents=True, exist_ok=True)

    nobs = dataset.nobs
    nvars = dataset.nvars
    cols = dataset.columns
    endian = "<"

    # Determine binary type codes and refresh string widths if needed
    type_codes: list[int] = []
    formats: list[str] = []
    for col in cols:
        m = dataset.meta[col]
        if m.is_string():
            max_bytes = max((len(str(v).encode("utf-8")) for v in dataset.df[col]), default=1)
            if m.stata_type == "strL" or max_bytes > 2045:
                type_codes.append(TYPE_STRL)
                formats.append(m.fmt or "%9s")
            else:
                width = max(1, min(2045, max_bytes))
                type_codes.append(width)
                formats.append(m.fmt or f"%{min(width, 244)}s")
        else:
            tc = TYPE_TO_CODE.get(m.stata_type, TYPE_DOUBLE)
            type_codes.append(tc)
            formats.append(m.fmt or "%9.0g")

    ts_bytes = datetime.now().strftime("%d %b %Y %H:%M").encode("ascii")[:17]
    lbl_bytes = dataset.data_label.encode("utf-8", errors="replace")[:320]

    header = bytearray()
    header += b"<stata_dta><header>"
    header += b"<release>118</release><byteorder>LSF</byteorder>"
    header += b"<K>" + struct.pack(f"{endian}H", nvars) + b"</K>"
    header += b"<N>" + struct.pack(f"{endian}Q", nobs) + b"</N>"
    header += b"<label>" + struct.pack(f"{endian}H", len(lbl_bytes)) + lbl_bytes + b"</label>"
    header += b"<timestamp>" + bytes([len(ts_bytes)]) + ts_bytes + b"</timestamp>"
    header += b"</header>"

    map_offset = len(header)
    map_placeholder = b"<map>" + b"\x00" * (14 * 8) + b"</map>"

    # Sections
    sec_vtypes = b"<variable_types>" + struct.pack(f"{endian}{nvars}H", *type_codes) + b"</variable_types>"
    sec_varnames = b"<varnames>" + b"".join(_encode_cstr(c, 129) for c in cols) + b"</varnames>"

    col_to_idx = {c: i + 1 for i, c in enumerate(cols)}
    slist = [col_to_idx[s] for s in dataset.sort_by if s in col_to_idx]
    slist_padded = (slist + [0] * (nvars + 1))[: nvars + 1]
    sec_sortlist = b"<sortlist>" + struct.pack(f"{endian}{nvars + 1}H", *slist_padded) + b"</sortlist>"

    sec_formats = b"<formats>" + b"".join(_encode_cstr(f, 57) for f in formats) + b"</formats>"
    sec_vlnames = (
        b"<value_label_names>"
        + b"".join(_encode_cstr(dataset.meta[c].value_label, 129) for c in cols)
        + b"</value_label_names>"
    )
    sec_varlabs = (
        b"<variable_labels>"
        + b"".join(_encode_cstr(dataset.meta[c].label, 321) for c in cols)
        + b"</variable_labels>"
    )

    ch_buf = bytearray(b"<characteristics>")
    for c_var, ch_dict in dataset.characteristics.items():
        for c_name, c_val in ch_dict.items():
            payload = (
                _encode_cstr(c_var, 129)
                + _encode_cstr(c_name, 129)
                + c_val.encode("utf-8", errors="replace")
                + b"\x00"
            )
            ch_buf += b"<ch>" + struct.pack(f"{endian}I", len(payload)) + payload + b"</ch>"
    ch_buf += b"</characteristics>"
    sec_chars = bytes(ch_buf)

    # Build <data> and <strls>
    strls_buf = bytearray(b"<strls>")
    col_chunks: list[np.ndarray] = []
    strl_counter: dict[int, int] = {}

    for col_idx, (col, tc) in enumerate(zip(cols, type_codes)):
        v_num = col_idx + 1
        if 1 <= tc <= 2045:
            col_arr = np.zeros((nobs, tc), dtype=np.uint8)
            for r, val in enumerate(dataset.df[col].astype(str)):
                b = val.encode("utf-8", errors="replace")[:tc]
                if b:
                    col_arr[r, : len(b)] = np.frombuffer(b, dtype=np.uint8)
            col_chunks.append(col_arr)
        elif tc == TYPE_STRL:
            col_arr = np.zeros((nobs, 8), dtype=np.uint8)
            for r, val in enumerate(dataset.df[col].astype(str)):
                if val == "":
                    continue
                o_num = strl_counter.get(v_num, 0) + 1
                strl_counter[v_num] = o_num
                ref = v_num.to_bytes(2, "little") + o_num.to_bytes(6, "little")
                col_arr[r, :] = np.frombuffer(ref, dtype=np.uint8)
                b = val.encode("utf-8", errors="replace") + b"\x00"
                strls_buf += b"GSO" + struct.pack(f"{endian}IQBI", v_num, o_num, 130, len(b)) + b
            col_chunks.append(col_arr)
        else:
            vals = dataset.df[col].to_numpy(dtype=np.float64, copy=True)
            mc = dataset.missing_codes[col]
            is_miss = mc > MISSING_NONE
            if tc == TYPE_DOUBLE:
                u64 = vals.astype("<f8").view("<u8").copy()
                if np.any(is_miss):
                    u64[is_miss] = _F64_BASE + (mc[is_miss].astype(np.uint64) - 1) * _F64_STEP
                col_chunks.append(u64.view(np.uint8).reshape(nobs, 8))
            elif tc == TYPE_FLOAT:
                f32 = vals.astype("<f4")
                u32 = f32.view("<u4").copy()
                if np.any(is_miss):
                    u32[is_miss] = _F32_BASE + (mc[is_miss].astype(np.uint32) - 1) * _F32_STEP
                col_chunks.append(u32.view(np.uint8).reshape(nobs, 4))
            elif tc == TYPE_LONG:
                i32 = np.where(is_miss, 0, np.nan_to_num(vals, nan=0.0)).astype("<i4")
                if np.any(is_miss):
                    i32[is_miss] = _I32_BASE + (mc[is_miss].astype(np.int32) - 1)
                col_chunks.append(i32.view(np.uint8).reshape(nobs, 4))
            elif tc == TYPE_INT:
                i16 = np.where(is_miss, 0, np.nan_to_num(vals, nan=0.0)).astype("<i2")
                if np.any(is_miss):
                    i16[is_miss] = _I16_BASE + (mc[is_miss].astype(np.int16) - 1)
                col_chunks.append(i16.view(np.uint8).reshape(nobs, 2))
            elif tc == TYPE_BYTE:
                i8 = np.where(is_miss, 0, np.nan_to_num(vals, nan=0.0)).astype("<i1")
                if np.any(is_miss):
                    i8[is_miss] = _I8_BASE + (mc[is_miss].astype(np.int8) - 1)
                col_chunks.append(i8.view(np.uint8).reshape(nobs, 1))

    strls_buf += b"</strls>"
    sec_strls = bytes(strls_buf)

    if nobs > 0 and col_chunks:
        packed_data = np.hstack(col_chunks).tobytes()
    else:
        packed_data = b""
    sec_data = b"<data>" + packed_data + b"</data>"

    # <value_labels>
    vl_buf = bytearray(b"<value_labels>")
    for labname, table in dataset.value_labels.items():
        if not table.mapping:
            continue
        sorted_items = sorted(table.mapping.items(), key=lambda kv: kv[0])
        txt_buf = bytearray()
        offs: list[int] = []
        vints: list[int] = []
        for k_val, v_txt in sorted_items:
            offs.append(len(txt_buf))
            vints.append(int(k_val))
            txt_buf += v_txt.encode("utf-8", errors="replace") + b"\x00"
        n_ent = len(sorted_items)
        inner = (
            _encode_cstr(labname, 129)
            + b"\x00\x00\x00"
            + struct.pack(f"{endian}II", n_ent, len(txt_buf))
            + struct.pack(f"{endian}{n_ent}I", *offs)
            + struct.pack(f"{endian}{n_ent}i", *vints)
            + bytes(txt_buf)
        )
        vl_buf += b"<lbl>" + struct.pack(f"{endian}I", len(inner)) + inner + b"</lbl>"
    vl_buf += b"</value_labels>"
    sec_vlabels = bytes(vl_buf)
    footer = b"</stata_dta>"

    # Compute exact section offsets
    offsets = [0] * 14
    offsets[0] = 0
    offsets[1] = map_offset
    cur = map_offset + len(map_placeholder)
    sections = [
        sec_vtypes,
        sec_varnames,
        sec_sortlist,
        sec_formats,
        sec_vlnames,
        sec_varlabs,
        sec_chars,
        sec_data,
        sec_strls,
        sec_vlabels,
    ]
    for i, sec in enumerate(sections, start=2):
        offsets[i] = cur
        cur += len(sec)
    offsets[12] = cur
    offsets[13] = cur + len(footer)

    map_bytes = b"<map>" + struct.pack(f"{endian}14Q", *offsets) + b"</map>"

    with p.open("wb") as fh:
        fh.write(header)
        fh.write(map_bytes)
        for sec in sections:
            fh.write(sec)
        fh.write(footer)

    dataset.source_path = str(p)
    dataset.dirty = False
