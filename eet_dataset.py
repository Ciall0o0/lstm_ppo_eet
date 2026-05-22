#!/usr/bin/env python3
"""
EET Dataset Extractor — extracts neural-network-ready training data from .eet files.

Usage:
    python eet_dataset.py <root_dir>                     # build dataset from all subdirs
    python eet_dataset.py <root_dir> -o ./datasets       # specify output directory
    python eet_dataset.py <root_dir> --csv               # also export flat CSV
    python eet_dataset.py <root_dir> --label-scheme flat # merge meeting start/scatter

Output (in output_dir/):
    global_features.npz       # float32 [n_files, n_global_features]
    event_sequences.npz       # object array of [n_events_i, 10] per file
    labels.npz                # int32 [n_files]
    file_ids.npz              # str [n_files]
    feature_names.txt         # column names for global features
    event_feature_names.txt   # column names for event features
    metadata.csv              # per-file summary
    events_flat.csv           # (--csv flag) all events with file_id
"""

import sys
import struct
import math
import os
import csv
import dataclasses
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eet_parser import EETFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECORD_SIZE = 64
BLOCK_SIZE = 32
MAX_FLOOR = 10
FALLBACK_OFFSET_A = 0x418
FALLBACK_OFFSET_BC = 0x420

# Pre-compiled struct formats for the decode hot path
_U32 = struct.Struct("<I")
_F32 = struct.Struct("<f")

# Feature vector index offsets (after the 7 FEATURE_SPEC entries)
IDX_FLAGS = 7
IDX_N_RECORDS = 8
IDX_N_EVENTS = 9

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScenarioMeta:
    scenario_id: int
    name: str
    group: str
    flags_le16: int
    label: int
    sub_type: str | None = None

    @property
    def label_name(self) -> str:
        if self.sub_type:
            return f"{self.name}_{self.sub_type}"
        return self.name


class ScenarioRegistry:
    """Classify .eet files by parsing directory name and header flags."""

    FLAG_TO_GROUP = {
        0x1900: "A",
        0x2100: "A",
        0x1F00: "B",
        0x1B00: "B",
        0x1E00: "C",
    }

    DIR_PATTERNS = [
        (r"1早高峰", 1, "morning_peak"),
        (r"2晚高峰", 2, "evening_peak"),
        (r"3午间双向交叉峰", 3, "noon_cross"),
        (r"4层间交互客流", 4, "interfloor"),
        (r"5极端闲时", 5, "extreme_idle"),
        (r"6会议活动突发流", 6, "meeting"),
    ]

    @classmethod
    def classify(cls, file_path: str, flags_le16: int) -> ScenarioMeta:
        path = os.path.abspath(file_path)
        group = cls.FLAG_TO_GROUP.get(flags_le16, "?")

        scenario_id = 0
        name = "unknown"
        for pattern, sid, sname in cls.DIR_PATTERNS:
            if pattern in path:
                scenario_id = sid
                name = sname
                break

        sub_type = None
        if scenario_id == 6:
            fname = os.path.basename(file_path)
            if "开始" in fname:
                sub_type = "start"
            elif "散场" in fname:
                sub_type = "scatter"

        return ScenarioMeta(
            scenario_id=scenario_id,
            name=name,
            group=group,
            flags_le16=flags_le16,
            label=scenario_id,
            sub_type=sub_type,
        )

    @classmethod
    def compute_label(cls, meta: ScenarioMeta, scheme: str = "split_meeting") -> int:
        if scheme == "flat":
            return meta.scenario_id
        if scheme == "split_meeting":
            if meta.scenario_id == 6 and meta.sub_type == "scatter":
                return 7
            return meta.scenario_id
        raise ValueError(f"Unknown label scheme: {scheme}")


# ---------------------------------------------------------------------------
# Section feature extractor
# ---------------------------------------------------------------------------

class SectionFeatureExtractor:
    """Extract global features from labeled sections of an EETFile."""

    FEATURE_SPEC: dict[str, tuple[str, str, int, float]] = {
        "auto_run_signal":  ("自动运行信号u",    "u32", 0),
        "ready_timestamp":  ("准备就绪信号(",    "u32", 4),
        "flow_stage":       ("客流环节",        "u32", 0),
        "flow_count":       ("客流环节",        "u32", 4),
        "passenger_count":  ("运输乘客数量y",    "u32", 0),
        "avg_ride_time":    ("乘客平均乘梯时间{", "f64", 0),
        "avg_wait_time":    ("乘客平均候梯时间z", "f64", 0),
    }

    _PLACEHOLDER_ALWAYS_VALID = frozenset(
        {"flow_stage", "auto_run_signal", "ready_timestamp"}
    )

    @staticmethod
    def _get_from_parsed(parsed_data: dict | None, field_type: str, byte_offset: int):
        if parsed_data is None:
            return np.nan
        if field_type == "u32" and "u32" in parsed_data:
            for off, val, _hex in parsed_data["u32"]:
                if off == byte_offset:
                    return float(val)
        if field_type == "f64" and "f64" in parsed_data:
            for off, val in parsed_data["f64"]:
                if off == byte_offset:
                    return float(val)
        return np.nan

    @classmethod
    def extract(cls, eet: EETFile) -> dict:
        seen = {}
        for sec in eet.sections:
            label = sec["label"]
            if label not in seen and sec.get("data") is not None:
                seen[label] = sec["data"]

        features = {}
        for feat_name, (sec_label, ftype, off) in cls.FEATURE_SPEC.items():
            val = cls._get_from_parsed(seen.get(sec_label), ftype, off)
            features[feat_name] = (
                val if not (isinstance(val, float) and math.isnan(val)) else np.nan
            )
        return features

    @classmethod
    def is_placeholder(cls, features: dict, meta: ScenarioMeta) -> list[bool]:
        is_template = meta.scenario_id < 6
        mask = []
        for feat_name in cls.FEATURE_SPEC:
            if feat_name in cls._PLACEHOLDER_ALWAYS_VALID:
                mask.append(False)
                continue
            v = features[feat_name]
            if feat_name == "passenger_count":
                mask.append(is_template and (v == 1 or v == 1.0 or np.isnan(v)))
            elif feat_name in ("avg_ride_time", "avg_wait_time"):
                mask.append(is_template and (v == 1.0 or np.isnan(v)))
            elif feat_name == "flow_count":
                mask.append(is_template and (v == 0 or np.isnan(v)))
            else:
                mask.append(False)
        return mask


# ---------------------------------------------------------------------------
# Trip event (unified schema)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TripEvent:
    """Single passenger trip event in unified schema."""
    source_floor: int
    dest_floor: int
    event_time: float
    arrival_time: float
    duration: float
    direction: int
    position: float
    time_delta: float
    floor_delta: int
    total_floors: int
    record_index: int
    block: int

    _EXPORT_FIELDS = frozenset({"record_index", "block"})


EVENT_FEATURE_NAMES = [
    f.name for f in dataclasses.fields(TripEvent)
    if f.name not in TripEvent._EXPORT_FIELDS
]


def _unpack_f32(raw: bytes, offset: int) -> float:
    if offset + 4 > len(raw):
        return 0.0
    f = _F32.unpack_from(raw, offset)[0]
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


# ---------------------------------------------------------------------------
# Record decoders
# ---------------------------------------------------------------------------

class RecordDecoder:
    """Base class for format-specific record decoders."""

    DECISECONDS_TO_SECONDS = 0.1
    _FALLBACK_OFFSET = FALLBACK_OFFSET_BC

    def decode(self, eet: EETFile) -> list[TripEvent]:
        d = eet.data
        rec_start = self._find_records_offset(eet)
        events = []
        prev_time = 0.0

        for idx in range(1, len(eet.records)):
            off = rec_start + idx * RECORD_SIZE
            if off + RECORD_SIZE > len(d):
                break
            raw = d[off:off + RECORD_SIZE]

            for blk in (0, 1):
                bo = blk * BLOCK_SIZE
                braw = raw[bo:bo + BLOCK_SIZE]
                if len(braw) < BLOCK_SIZE:
                    continue

                ev = self._decode_block(braw, blk, raw, idx)
                if ev is None:
                    continue

                time_delta = ev.event_time - prev_time if prev_time > 0 else 0.0
                ev = dataclasses.replace(ev, time_delta=time_delta)
                prev_time = ev.event_time
                events.append(ev)

        return events

    def _decode_block(self, braw: bytes, blk: int, full_raw: bytes,
                      rec_idx: int) -> TripEvent | None:
        raise NotImplementedError

    @staticmethod
    def _find_records_offset(eet: EETFile) -> int:
        if eet.sections:
            last = eet.sections[-1]
            return ((last["data_offset"] + last["data_len"]) + 7) & ~7
        return RecordDecoder._FALLBACK_OFFSET

    @staticmethod
    def _is_valid_event(src: int, dst: int, max_floor: int = MAX_FLOOR) -> bool:
        if src == 0 and dst == 0:
            return False
        if src > max_floor or dst > max_floor:
            return False
        if src == 0xFF or dst == 0xFF:
            return False
        return True


class GroupADecoder(RecordDecoder):
    """Scenarios 1, 2, 3 — morning/evening/noon peak.

    Per-block field layout (bytes 0-31):
        byte 0:  direction/source (high byte)
        byte 16: event timestamp (deciseconds, u32)
        byte 20: direction/marker (u32)
        byte 24: dest floor (high byte)
    """

    _FALLBACK_OFFSET = FALLBACK_OFFSET_A

    def _decode_block(self, braw: bytes, blk: int, full_raw: bytes,
                      rec_idx: int) -> TripEvent | None:
        src = (_U32.unpack_from(braw, 0)[0] >> 24) & 0xFF
        dst = (_U32.unpack_from(braw, 24)[0] >> 24) & 0xFF
        et = _U32.unpack_from(braw, 16)[0]
        direction = (_U32.unpack_from(braw, 20)[0] >> 24) & 0xFF

        if dst == 0 or not self._is_valid_event(src, dst):
            return None

        if blk == 0 and len(full_raw) >= BLOCK_SIZE * 2:
            at = _U32.unpack_from(full_raw, 48)[0]
        else:
            at = et

        event_time = et * self.DECISECONDS_TO_SECONDS
        arrival_time = at * self.DECISECONDS_TO_SECONDS

        return TripEvent(
            source_floor=src,
            dest_floor=dst,
            event_time=event_time,
            arrival_time=arrival_time,
            duration=max(0, arrival_time - event_time),
            direction=direction,
            position=0.0,
            time_delta=0.0,
            floor_delta=dst - src,
            total_floors=MAX_FLOOR,
            record_index=rec_idx,
            block=blk,
        )


class GroupBDecoder(RecordDecoder):
    """Scenarios 4, 5 — interfloor / extreme idle.

    Per-block field layout (bytes 0-31):
        byte 12: timestamp (raw ticks, u32)
        byte 20: source floor (byte 1: 0xN00 → floor N)
        byte 24: dest floor (byte 1: 0xN00 → floor N)
    """

    def _decode_block(self, braw: bytes, blk: int, full_raw: bytes,
                      rec_idx: int) -> TripEvent | None:
        ts = _U32.unpack_from(braw, 12)[0]
        src = (_U32.unpack_from(braw, 20)[0] >> 8) & 0xFF
        dst = (_U32.unpack_from(braw, 24)[0] >> 8) & 0xFF

        if not self._is_valid_event(src, dst):
            return None

        event_time = float(ts)

        return TripEvent(
            source_floor=src,
            dest_floor=dst,
            event_time=event_time,
            arrival_time=event_time,
            duration=0.0,
            direction=0,
            position=0.0,
            time_delta=0.0,
            floor_delta=dst - src,
            total_floors=MAX_FLOOR,
            record_index=rec_idx,
            block=blk,
        )


class GroupCDecoder(RecordDecoder):
    """Scenario 6 — meeting start / scatter.

    Per-block field layout (bytes 0-31):
        byte 0:  source floor (integer, u32)
        byte 8:  float32 car position
        byte 12: timestamp (deciseconds, u32)
        byte 20: dest floor (integer, u32)
    """

    def _decode_block(self, braw: bytes, blk: int, full_raw: bytes,
                      rec_idx: int) -> TripEvent | None:
        src = _U32.unpack_from(braw, 0)[0]
        dst = _U32.unpack_from(braw, 20)[0]
        ts = _U32.unpack_from(braw, 12)[0]
        pos = _unpack_f32(braw, 8)

        if dst == 0 or dst > 20 or src > 20:
            return None

        event_time = ts * self.DECISECONDS_TO_SECONDS

        return TripEvent(
            source_floor=src,
            dest_floor=dst,
            event_time=event_time,
            arrival_time=event_time,
            duration=0.0,
            direction=0,
            position=pos,
            time_delta=0.0,
            floor_delta=dst - src,
            total_floors=MAX_FLOOR,
            record_index=rec_idx,
            block=blk,
        )


DECODER_REGISTRY = {
    "A": GroupADecoder,
    "B": GroupBDecoder,
    "C": GroupCDecoder,
}


# ---------------------------------------------------------------------------
# Event → numpy conversion
# ---------------------------------------------------------------------------

def events_to_array(events: list[TripEvent]) -> np.ndarray:
    if not events:
        return np.zeros((0, len(EVENT_FEATURE_NAMES)), dtype=np.float32)

    arr = np.zeros((len(events), len(EVENT_FEATURE_NAMES)), dtype=np.float32)
    for i, e in enumerate(events):
        arr[i] = [
            e.source_floor, e.dest_floor, e.event_time, e.arrival_time,
            e.duration, e.direction, e.position, e.time_delta,
            e.floor_delta, e.total_floors,
        ]
    return arr


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

GLOBAL_FEATURE_NAMES = list(SectionFeatureExtractor.FEATURE_SPEC) + [
    "flags_le16", "n_records", "n_events"
]


class DatasetBuilder:
    """Aggregate extracted features across multiple .eet files."""

    def __init__(self, root_dir: str, label_scheme: str = "split_meeting"):
        self.root_dir = root_dir
        self.label_scheme = label_scheme

        self.global_features: list[np.ndarray] = []
        self.event_sequences: list[np.ndarray] = []
        self.labels: list[int] = []
        self.file_ids: list[str] = []
        self.metadata: list[dict] = []
        self._placeholder_masks: list[np.ndarray] = []

    def add_file(self, file_path: str):
        eet = EETFile(file_path)
        meta = ScenarioRegistry.classify(file_path, eet.header["flags_le16"])
        label = ScenarioRegistry.compute_label(meta, self.label_scheme)

        feats = SectionFeatureExtractor.extract(eet)
        placeholder_mask = SectionFeatureExtractor.is_placeholder(feats, meta)

        n_feat = len(SectionFeatureExtractor.FEATURE_SPEC)
        feat_vec = np.zeros(len(GLOBAL_FEATURE_NAMES), dtype=np.float32)
        for i, name in enumerate(SectionFeatureExtractor.FEATURE_SPEC):
            v = feats[name]
            feat_vec[i] = v if not (isinstance(v, float) and math.isnan(v)) else -1.0
        feat_vec[IDX_FLAGS] = float(eet.header["flags_le16"])
        feat_vec[IDX_N_RECORDS] = float(len(eet.records))

        decoder_cls = DECODER_REGISTRY.get(meta.group)
        if decoder_cls:
            events = decoder_cls().decode(eet)
        else:
            events = []

        event_arr = events_to_array(events)
        feat_vec[IDX_N_EVENTS] = float(len(events))

        self.global_features.append(feat_vec)
        self.event_sequences.append(event_arr)
        self.labels.append(label)
        self.file_ids.append(os.path.basename(file_path))
        self._placeholder_masks.append(np.array(placeholder_mask, dtype=bool))
        self.metadata.append({
            "file": os.path.basename(file_path),
            "scenario": meta.name,
            "scenario_id": meta.scenario_id,
            "label": label,
            "group": meta.group,
            "sub_type": meta.sub_type or "",
            "n_records": len(eet.records),
            "n_events": len(events),
            "filesize": len(eet.data),
        })

    def add_directory(self, dir_path: str):
        for fp in sorted(Path(dir_path).glob("*.eet")):
            self.add_file(str(fp))

    def add_all_scenarios(self):
        for entry in sorted(os.listdir(self.root_dir)):
            full = os.path.join(self.root_dir, entry)
            if os.path.isdir(full) and any(Path(full).glob("*.eet")):
                self.add_directory(full)

    def build(self) -> tuple:
        return (
            np.array(self.global_features, dtype=np.float32),
            np.array(self.event_sequences, dtype=object),
            np.array(self.labels, dtype=np.int32),
            np.array(self.file_ids, dtype=object),
            np.array(self._placeholder_masks, dtype=bool),
        )

    def save(self, output_dir: str, export_csv: bool = False):
        os.makedirs(output_dir, exist_ok=True)

        X_global, X_events, y, fids, masks = self.build()

        # Pad event sequences to uniform max length
        max_len = max(len(seq) for seq in X_events)
        n_features = len(EVENT_FEATURE_NAMES)
        n_files = len(X_events)
        event_lens = np.array([len(seq) for seq in X_events], dtype=np.int32)

        X_events_padded = np.zeros((n_files, max_len, n_features), dtype=np.float32)
        event_mask = np.zeros((n_files, max_len), dtype=bool)
        for i, seq in enumerate(X_events):
            L = len(seq)
            if L > 0:
                X_events_padded[i, :L] = seq
                event_mask[i, :L] = True

        np.savez_compressed(os.path.join(output_dir, "global_features.npz"), X_global)
        np.savez_compressed(os.path.join(output_dir, "event_sequences.npz"),
                            X_events_padded, event_mask)
        np.savez_compressed(os.path.join(output_dir, "event_lengths.npz"), event_lens)
        # Store filenames as fixed-width unicode (viewable in NPZ viewers)
        fids_fixed = np.array(fids, dtype='U50')
        np.savez_compressed(os.path.join(output_dir, "labels.npz"), y)
        np.savez_compressed(os.path.join(output_dir, "file_ids.npz"), fids_fixed)
        np.savez_compressed(os.path.join(output_dir, "placeholder_masks.npz"), masks)

        with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
            for name in GLOBAL_FEATURE_NAMES:
                f.write(name + "\n")
        with open(os.path.join(output_dir, "event_feature_names.txt"), "w") as f:
            for name in EVENT_FEATURE_NAMES:
                f.write(name + "\n")

        self._write_metadata_csv(output_dir)
        if export_csv:
            self._write_events_flat_csv(output_dir)

        self._print_summary(output_dir, X_global, X_events_padded, y, fids, masks,
                            event_lens, event_mask)

    def _print_summary(self, output_dir, X_global, X_events, y, fids, masks,
                        event_lens=None, event_mask=None):
        total_events = event_lens.sum() if event_lens is not None else sum(
            len(seq) if hasattr(seq, '__len__') and not isinstance(seq, np.ndarray)
            else (event_mask[i].sum() if event_mask is not None else 0)
            for i, seq in enumerate(X_events))
        print(f"\nSaved to {output_dir}:")
        print(f"  global_features.npz    → {X_global.shape}")
        print(f"  event_sequences.npz    → {X_events.shape} (padded, float32)")
        print(f"  event_lengths.npz      → ({len(event_lens)},) actual lengths")
        print(f"                           total events: {total_events}")
        print(f"  labels.npz             → {y.shape}, classes: {np.unique(y)}")
        print(f"  file_ids.npz           → {len(fids)} files")
        print(f"  placeholder_masks.npz  → {masks.shape}")
        print(f"  metadata.csv           → {len(self.metadata)} rows")
        print("\nPer-scenario breakdown:")
        for meta_dict in self.metadata:
            print(
                f"  label={meta_dict['label']:2d}  {meta_dict['scenario']:<16s}"
                f"  {meta_dict['sub_type']:<8s}"
                f"  events={meta_dict['n_events']:>4d}"
                f"  records={meta_dict['n_records']:>4d}"
                f"  {meta_dict['file']}"
            )

    def _write_metadata_csv(self, output_dir: str):
        path = os.path.join(output_dir, "metadata.csv")
        if not self.metadata:
            return
        keys = list(self.metadata[0])
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.metadata)

    def _write_events_flat_csv(self, output_dir: str):
        path = os.path.join(output_dir, "events_flat.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file_id", "event_idx"] + EVENT_FEATURE_NAMES)
            for fid, events in zip(self.file_ids, self.event_sequences):
                for i in range(len(events)):
                    w.writerow([fid, i] + list(events[i]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root_dir = sys.argv[1]
    output_dir = "./datasets"
    export_csv = False
    label_scheme = "split_meeting"

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] in ("-o", "--output-dir") and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif args[i] == "--csv":
            export_csv = True
            i += 1
        elif args[i] == "--label-scheme" and i + 1 < len(args):
            label_scheme = args[i + 1]
            i += 2
        else:
            i += 1

    print(f"Building dataset from: {root_dir}")
    print(f"Label scheme: {label_scheme}\n")

    builder = DatasetBuilder(root_dir, label_scheme=label_scheme)
    builder.add_all_scenarios()

    print(f"Found {len(builder.file_ids)} files across all scenarios")
    builder.save(output_dir, export_csv=export_csv)


if __name__ == "__main__":
    main()
