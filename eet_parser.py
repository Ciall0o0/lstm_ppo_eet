#!/usr/bin/env python3
"""
EET Pro file parser — decodes .eet binary project files from EET Pro V3.4.0+
(Elevator Engineering Training / 电梯仿真软件).

Usage:
    python eet_parser.py <file.eet>              # print summary
    python eet_parser.py <file.eet> -o out.json  # write JSON to file
    python eet_parser.py <dir>                   # batch parse all .eet files

The .eet format is a proprietary binary container (NOT encrypted).
Structure: 16-byte header → GBK project name → labeled sections → 64-byte
event records → footer (PLC IP address).
"""

import sys
import struct
import json
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_SIZE = 16
SECTION_SCAN_LIMIT = 1500     # bytes to scan for labeled sections before records
RECORD_SCAN_LIMIT = 4096      # bytes to scan for record region start
RECORD_SIZE = 64
FALLBACK_RECORD_OFFSET = 0x480
DATA_PARSE_LIMIT = 32         # max bytes to extract u32/f64 from
FLOAT64_MIN = 1e-12
FLOAT64_MAX = 1e10
FLOAT64_SANITY_BOUND = 1e8
FLOOR_SANITY_BOUND = 0x100

# ---------------------------------------------------------------------------
# GBK helpers
# ---------------------------------------------------------------------------

def _is_gbk_lead(b: int) -> bool:
    return 0x81 <= b <= 0xFE

def _is_gbk_trail(b: int) -> bool:
    return (0x40 <= b <= 0x7E) or (0x80 <= b <= 0xFE)

def is_valid_marker_boundary(data: bytes, pos: int) -> bool:
    """Check if byte at *pos* is preceded by a valid section boundary."""
    if pos <= 0:
        return True
    prev = data[pos - 1]
    if prev == 0x00 or prev < 0x20 or (0x20 <= prev <= 0x7E):
        return True
    if pos >= 2 and _is_gbk_lead(data[pos - 2]) and _is_gbk_trail(prev):
        return True
    return False

def read_gbk(data: bytes, offset: int):
    """Read a GBK string at *offset*. Returns (string, end_offset)."""
    start = offset
    while start < len(data) and data[start] == 0x00:
        start += 1
    if start >= len(data):
        return "", start

    end = start
    while end < len(data):
        b = data[end]
        if b < 0x20:
            break
        if _is_gbk_lead(b) and end + 1 < len(data) and _is_gbk_trail(data[end + 1]):
            end += 2
        else:
            end += 1

    raw = data[start:end]
    try:
        s = raw.decode("gbk")
    except (UnicodeDecodeError, LookupError):
        s = raw.decode("gbk", errors="replace")
    return s, end

def _count_chinese(s: str) -> int:
    return sum(1 for c in s if "一" <= c <= "鿿")

# ---------------------------------------------------------------------------
# Marker types
# ---------------------------------------------------------------------------

MARKER_NAMES = {
    0x03: "GROUP",
    0x06: "RECORD_VAR",
    0x08: "METRIC",
    0x0C: "SIGNAL",
    0x10: "FORMULA",
}

# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class EETFile:
    """Parsed representation of a .eet file."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.data = f.read()
        self.path = path

        self.header: dict = {}
        self.project_name: str = ""
        self.sections: list[dict] = []
        self.records: list[dict] = []
        self.footer: dict = {}

        self._parse()

    def _parse_header(self):
        d = self.data
        if len(d) < HEADER_SIZE:
            raise ValueError(f"File too small ({len(d)} bytes) for EET header")

        end = 13
        while end > 1 and d[end - 1] == 0:
            end -= 1
        magic = d[1:end].decode("ascii", errors="replace")

        self.header = {
            "magic": magic,
            "version": d[13],
            "flags_le16": struct.unpack_from("<H", d, 14)[0],
        }

    def _parse_project_name(self):
        name, _ = read_gbk(self.data, 16)
        self.project_name = name

    def _find_labeled_sections(self) -> list[dict]:
        d = self.data
        search_limit = min(SECTION_SCAN_LIMIT, len(d))
        sections = []

        i = 0
        while i < search_limit:
            b = d[i]
            if b not in MARKER_NAMES:
                i += 1
                continue

            if not is_valid_marker_boundary(d, i):
                i += 1
                continue

            label, label_end = read_gbk(d, i + 1)
            if not label:
                i += 1
                continue

            cn = _count_chinese(label)
            if cn < 2:
                i += 1
                continue
            if "@" in label:
                i += 1
                continue

            data_start = label_end
            while data_start < len(d) and d[data_start] == 0x00:
                data_start += 1

            next_marker = None
            for j in range(label_end, min(len(d), label_end + 2000)):
                if d[j] not in MARKER_NAMES:
                    continue
                if not is_valid_marker_boundary(d, j):
                    continue
                nl, _ = read_gbk(d, j + 1)
                if nl and _count_chinese(nl) >= 2 and "@" not in nl:
                    next_marker = j
                    break

            if next_marker:
                data_end = next_marker
                while data_end > data_start and d[data_end - 1] == 0x00:
                    data_end -= 1
            else:
                data_end = self._find_record_start(label_end)

            raw = d[data_start:data_end] if data_start < data_end else b""

            sections.append({
                "offset": i,
                "marker": b,
                "marker_name": MARKER_NAMES.get(b, f"0x{b:02X}"),
                "label": label,
                "data_offset": data_start,
                "data_len": len(raw),
                "data": self._parse_data_block(raw) if raw else None,
            })

            i = label_end

        return sections

    def _find_record_start(self, search_from: int) -> int:
        d = self.data
        for pos in range(search_from, min(search_from + RECORD_SCAN_LIMIT,
                                           len(d) - RECORD_SIZE * 3)):
            if (self._is_record(pos)
                    and self._is_record(pos + RECORD_SIZE)
                    and self._is_record(pos + RECORD_SIZE * 2)):
                return pos
        return len(d)

    def _is_record(self, pos: int) -> bool:
        d = self.data
        if pos + RECORD_SIZE > len(d):
            return False
        f = struct.unpack_from("<d", d, pos)[0]
        if math.isnan(f) or math.isinf(f) or abs(f) > FLOAT64_SANITY_BOUND:
            return False
        u = struct.unpack_from("<I", d, pos + 8)[0]
        return u < FLOOR_SANITY_BOUND

    def _parse_data_block(self, raw: bytes) -> dict | None:
        if not raw:
            return None

        result = {"len": len(raw)}

        u32s = []
        for off in range(0, min(len(raw) & ~3, DATA_PARSE_LIMIT), 4):
            v = struct.unpack_from("<I", raw, off)[0]
            if v != 0:
                u32s.append((off, v, f"0x{v:08X}"))
        if u32s:
            result["u32"] = u32s

        f64s = []
        for off in range(0, min(len(raw) & ~7, DATA_PARSE_LIMIT), 8):
            v = struct.unpack_from("<d", raw, off)[0]
            if not (math.isnan(v) or math.isinf(v)) and FLOAT64_MIN < abs(v) < FLOAT64_MAX:
                f64s.append((off, round(v, 6)))
        if f64s:
            result["f64"] = f64s

        result["hex_preview"] = raw[:DATA_PARSE_LIMIT].hex(" ")
        return result

    def _parse_records(self):
        d = self.data

        if self.sections:
            last = self.sections[-1]
            start = last["data_offset"] + last["data_len"]
        else:
            start = FALLBACK_RECORD_OFFSET
        start = (start + 7) & ~7

        while start < len(d) - RECORD_SIZE * 2:
            if self._is_record(start):
                break
            start += 8
        else:
            return

        records = []
        pos = start
        while pos + RECORD_SIZE <= len(d):
            raw = d[pos:pos + RECORD_SIZE]

            if raw[:16] == b"\x00" * 16:
                pos += RECORD_SIZE
                continue
            if b"192.168" in raw:
                break

            rec = {"index": len(records)}
            for off in range(0, RECORD_SIZE, 8):
                f = struct.unpack_from("<d", raw, off)[0]
                rec[f"f64_{off:02d}"] = (
                    round(f, 4) if not (math.isnan(f) or math.isinf(f)) else str(f)
                )
            for off in range(0, RECORD_SIZE, 4):
                u = struct.unpack_from("<I", raw, off)[0]
                rec[f"u32_{off:02d}"] = u

            records.append(rec)
            pos += RECORD_SIZE

        self.records = records

    def _parse_footer(self):
        d = self.data
        off = d.rfind(b"192.168")
        if off >= 0:
            end = off
            while end < len(d) and d[end] not in (0x00, 0x0A, 0x0D):
                end += 1
            self.footer["plc_ip"] = d[off:end].decode("ascii", errors="replace")
            self.footer["plc_ip_offset"] = off

    def _parse(self):
        self._parse_header()
        self._parse_project_name()
        self.sections = self._find_labeled_sections()
        self._parse_records()
        self._parse_footer()

    def to_dict(self, full_records: bool = False) -> dict:
        if full_records or len(self.records) <= 6:
            recs = self.records
        else:
            recs = (self.records[:2]
                    + [f"... ({len(self.records) - 4} records omitted) ..."]
                    + self.records[-2:])
        return {
            "file": os.path.basename(self.path),
            "file_size": len(self.data),
            "header": self.header,
            "project_name": self.project_name,
            "sections": self.sections,
            "record_count": len(self.records),
            "records": recs,
            "footer": self.footer,
        }

    def to_json(self, indent: int = 2, full_records: bool = False) -> str:
        return json.dumps(
            self.to_dict(full_records=full_records),
            indent=indent, ensure_ascii=False, default=str,
        )

    def summary(self) -> str:
        lines = [
            f"File:       {os.path.basename(self.path)}",
            f"Size:       {len(self.data)} bytes",
            f"Magic:      {self.header.get('magic', '?')}",
            f"Project:    {self.project_name}",
            f"Sections:   {len(self.sections)}",
        ]
        for s in self.sections:
            lines.append(
                f"  [{s['marker_name']:11s}] {s['label']:<24s} "
                f"data: {s['data_len']:>5d} bytes"
            )
        lines.append(f"Records:    {len(self.records)}")
        if "plc_ip" in self.footer:
            lines.append(f"PLC IP:     {self.footer['plc_ip']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_one(path: str, output_path: str | None = None, full: bool = False) -> EETFile:
    eet = EETFile(path)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(eet.to_json(full_records=full))
        print(f"Wrote: {output_path}")
    else:
        print(eet.summary())
    return eet


def parse_directory(dir_path: str, full: bool = False):
    out_dir = os.path.join(dir_path, "_decoded")
    os.makedirs(out_dir, exist_ok=True)

    eet_files = sorted(Path(dir_path).glob("*.eet"))
    if not eet_files:
        print(f"No .eet files found in {dir_path}")
        return

    for fp in eet_files:
        out_name = fp.stem + "_decoded.json"
        out_path = os.path.join(out_dir, out_name)
        parse_one(str(fp), out_path, full)

    print(f"\nDecoded {len(eet_files)} files → {out_dir}/")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    output_path = None
    full = False

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "-o" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        elif args[i] == "--full":
            full = True
            i += 1
        else:
            i += 1

    if os.path.isdir(target):
        parse_directory(target, full)
    else:
        parse_one(target, output_path, full)


if __name__ == "__main__":
    main()
