"""
SimScale Mesh Log Parser (v2 - matches actual SimScale UI log format)
======================================================================

The original simscale_log_parser.py in this repo was written against a
synthetic sample log (camelCase metric names like "nonOrthogonality", plus
stddev/median/99.9/99.999 percentiles) that does NOT match what SimScale's
UI actually outputs. Real logs use spaced names ("Non Orthogonality", "Edge
Ratio", etc.) grouped under "Element type: Tetrahedra / Pyramids / Prisms /
Hexahedra" headers, and only report min / max / average / 99.99-th
percentile - no stddev, no median, no 99.9 or 99.999 tiers.

This version parses the Tetrahedra block (the dominant element type in
every run collected so far) plus total cell count. Extend METRIC_NAMES /
add per-element-type parsing if you need Pyramids/Prisms/Hexahedra stats
too.

USAGE
-----
    python simscale_log_parser_v2.py ./logs mesh_dataset_v2.csv
"""

import re
import sys
import csv
from pathlib import Path

METRIC_NAMES = ["Non Orthogonality", "Edge Ratio", "Volume Ratio", "Aspect Ratio", "Skewness"]
METRIC_KEYS = {
    "Non Orthogonality": "nonortho",
    "Edge Ratio": "edge_ratio",
    "Volume Ratio": "volume_ratio",
    "Aspect Ratio": "aspect_ratio",
    "Skewness": "skewness",
}
STAT_PATTERNS = {
    "min": r"min:\s*([-\d.eE+]+)",
    "max": r"max:\s*([-\d.eE+]+)",
    "average": r"average:\s*([-\d.eE+]+)",
    "p99_99": r"99\.99-th percentile:\s*([-\d.eE+]+)",
}

# fineness / boundary_layers per run - fill in as you add real SimScale runs
RUN_METADATA = {
    "run_001": {"fineness": 5, "boundary_layers": 1},
    "run_002": {"fineness": 2, "boundary_layers": 1},
    "run_003": {"fineness": 8, "boundary_layers": 1},
    "run_004": {"fineness": 5, "boundary_layers": 0},
    "run_005": {"fineness": 1, "boundary_layers": 0},
    "run_006": {"fineness": 1, "boundary_layers": 1},
    "run_007": {"fineness": 3, "boundary_layers": 1},
    "run_008": {"fineness": 3, "boundary_layers": 0},
    "run_009": {"fineness": 6, "boundary_layers": 1},
    "run_010": {"fineness": 4, "boundary_layers": 1},
    "run_011": {"fineness": 7, "boundary_layers": 1},
}


def extract_tetrahedra_block(text: str) -> str:
    m = re.search(r"Element type: Tetrahedra(.*?)(?=Element type:|\Z)", text, re.DOTALL)
    return m.group(1) if m else ""


def parse_metric_block(block: str, metric: str) -> dict:
    other = [m for m in METRIC_NAMES if m != metric]
    stop = "|".join(re.escape(o) for o in other) + r"|Min Edge Length|Number of volumes"
    m = re.search(rf"{re.escape(metric)}\n(.*?)(?={stop}|\Z)", block, re.DOTALL)
    if not m:
        return {}
    sub = m.group(1)
    return {stat: float(sm.group(1)) for stat, pat in STAT_PATTERNS.items() if (sm := re.search(pat, sub))}


def parse_log_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    row = {"run_id": path.stem}
    tet_block = extract_tetrahedra_block(text)
    for metric in METRIC_NAMES:
        stats = parse_metric_block(tet_block, metric)
        key = METRIC_KEYS[metric]
        for stat_name, value in stats.items():
            row[f"tet_{key}_{stat_name}"] = value

    cells_m = re.search(r"Number of volumes:\s*(\d+)", text)
    if cells_m:
        row["total_cells"] = int(cells_m.group(1))
    return row


def label_run(row: dict) -> str:
    """3-class label using SimScale's own acceptable ranges as ground truth."""
    nonortho_max = row.get("tet_nonortho_max", 0)
    vol_max = row.get("tet_volume_ratio_max", 0)
    if nonortho_max > 88.0 or vol_max > 100.0:
        return "bad"
    if nonortho_max >= 85.0:
        return "marginal"
    return "good"


def build_dataset(log_dir: Path, output_csv: Path) -> None:
    log_files = sorted(log_dir.glob("*.txt"))
    if not log_files:
        print(f"No .txt log files found in {log_dir.resolve()}")
        return

    rows = []
    for path in log_files:
        parsed = parse_log_file(path)
        meta = RUN_METADATA.get(path.stem, {})
        if not meta:
            print(f"  ! No metadata for {path.stem} - add fineness/boundary_layers to RUN_METADATA.")
        merged = {**{"run_id": path.stem}, **meta, **{k: v for k, v in parsed.items() if k != "run_id"}}
        merged["label"] = label_run(merged)
        rows.append(merged)
        print(f"  parsed {path.name}: label={merged['label']}")

    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows x {len(all_keys)} columns to {output_csv}")


if __name__ == "__main__":
    log_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("./logs")
    output_csv = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("./mesh_dataset_v2.csv")
    build_dataset(log_dir, output_csv)
