"""
SimScale Mesh Log Parser
========================

Parses SimScale mesh log output and builds a CSV dataset for ML training
(e.g. Naive Bayes mesh quality classification).

USAGE
-----
1. Save each mesh log from SimScale as a separate .txt file inside a folder.
   (Copy the log text from the SimScale UI -> paste into a .txt file.)
   Suggested filename: run_001.txt, run_002.txt, etc.

2. Edit the RUN_METADATA dict below to record the mesh INPUT parameters
   (fineness, inflation layers, etc.) and your post-run LABEL for each run.
   The keys must match the .txt filenames (without the .txt extension).

3. Run:
       python simscale_log_parser.py /path/to/log/folder output.csv

   Or just:
       python simscale_log_parser.py
   ...to use the default folder ./logs and write ./mesh_dataset.csv

The script extracts every metric (mean, std, median, max, min, and the
99.9 / 99.99 / 99.999 percentiles) for every quality field SimScale reports,
plus the overall_quality score. It then merges your per-run metadata
and writes one tidy CSV row per simulation.
"""

import os
import re
import sys
import csv
from pathlib import Path


# ----------------------------------------------------------------------------
# EDIT THIS SECTION FOR EACH BATCH OF RUNS
# ----------------------------------------------------------------------------
# Keys = filename stems (e.g. "run_001" for run_001.txt).
# Values = whatever input parameters you varied + the label you assigned
# after looking at convergence + Cl/Cd vs your reference.
#
# Required label values: "good", "marginal", or "bad"
# Add or remove fields as you like - they all become CSV columns.
# ----------------------------------------------------------------------------
RUN_METADATA = {
    "run_001": {
        "mesh_fineness": 5,
        "num_inflation_layers": 3,
        "first_layer_thickness": 0.001,
        "growth_rate": 1.2,
        "cell_count": 250000,
        "final_Cl": 0.45,
        "final_Cd": 0.018,
        "converged": True,
        "label": "good",
    },
    # TODO: fill in the actual parameters you used for run_002 and assign a label.
    # The log shows overall_quality=0.607 (vs 0.673 for run_001) and higher
    # nonOrthogonality/skewness — likely "marginal" or "bad".
    # "run_002": {
    #     "mesh_fineness": ???,
    #     "num_inflation_layers": ???,
    #     "first_layer_thickness": ???,
    #     "growth_rate": ???,
    #     "cell_count": ???,
    #     "final_Cl": ???,
    #     "final_Cd": ???,
    #     "converged": ???,
    #     "label": "marginal",  # or "bad" — review Cl/Cd vs reference
    # },
}


# ----------------------------------------------------------------------------
# Parser - you usually don't need to edit below this line
# ----------------------------------------------------------------------------

# The metric names that appear in the SimScale mesh log
METRIC_NAMES = [
    "tetEdgeRatio",
    "quadMaxAngle",
    "triMaxAngle",
    "triMinAngle",
    "volumeRatio",
    "tetAspectRatio",
    "nonOrthogonality",
    "skewness",
    "aspectRatio",
]

# Statistics reported under each metric
STAT_PATTERNS = {
    "min": r"min:\s*([-\d.eE+]+)",
    "max": r"max:\s*([-\d.eE+]+)",
    "average": r"average:\s*([-\d.eE+]+)",
    "stddev": r"standard deviation\s*([-\d.eE+]+)",
    "median": r"median:\s*([-\d.eE+]+)",
    "p99_9": r"99\.9-th percentile:\s*([-\d.eE+]+)",
    "p99_99": r"99\.99-th percentile:\s*([-\d.eE+]+)",
    "p99_999": r"99\.999-th percentile:\s*([-\d.eE+]+)",
}


def parse_metric_block(log_text: str, metric: str) -> dict:
    """Extract every statistic for one named metric from the log."""
    other_metrics = [m for m in METRIC_NAMES if m != metric]
    next_metric_alt = "|".join(other_metrics + [r"Overall mesh quality"])
    block_re = (
        rf"^{re.escape(metric)}\s*\n"
        rf"(.*?)"
        rf"(?=^(?:{next_metric_alt})\b|\Z)"
    )
    m = re.search(block_re, log_text, re.MULTILINE | re.DOTALL)
    if not m:
        return {}
    block = m.group(1)

    results = {}
    for stat_name, pattern in STAT_PATTERNS.items():
        sm = re.search(pattern, block)
        if sm:
            try:
                results[stat_name] = float(sm.group(1))
            except ValueError:
                results[stat_name] = None
    return results


def parse_overall_quality(log_text: str) -> dict:
    """Extract the overall quality score and its components."""
    out = {}
    m = re.search(
        r"Overall mesh quality\s*\(based on the 99-percentile\):\s*([-\d.eE+]+)",
        log_text,
    )
    if m:
        out["overall_quality"] = float(m.group(1))

    components = {
        "nonOrtho_component": r"Non Orthogonality:\s*([-\d.eE+]+)",
        "skewness_component": r"Skewness:\s*([-\d.eE+]+)",
        "aspectRatio_component": r"Aspect Ratio:\s*([-\d.eE+]+)",
    }
    for key, pat in components.items():
        sm = re.search(pat, log_text)
        if sm:
            out[key] = float(sm.group(1))
    return out


def parse_log_file(path: Path) -> dict:
    """Parse a single mesh log file into a flat dict of features."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"^```.*?$", "", text, flags=re.MULTILINE)

    row = {"run_id": path.stem}

    for metric in METRIC_NAMES:
        stats = parse_metric_block(text, metric)
        for stat_name, value in stats.items():
            row[f"{metric}_{stat_name}"] = value

    row.update(parse_overall_quality(text))
    return row


def build_dataset(log_dir: Path, output_csv: Path) -> None:
    """Walk every .txt log in log_dir, merge with RUN_METADATA, write CSV."""
    log_files = sorted(log_dir.glob("*.txt"))
    if not log_files:
        print(f"No .txt log files found in {log_dir.resolve()}")
        print("Make sure each SimScale mesh log is saved as a .txt file there.")
        return

    rows = []
    for log_path in log_files:
        parsed = parse_log_file(log_path)
        meta = RUN_METADATA.get(log_path.stem, {})
        if not meta:
            print(
                f"  ! No metadata for {log_path.stem} - "
                f"add it to RUN_METADATA in the script."
            )
        merged = {**meta, **parsed}
        ordered = {"run_id": merged.pop("run_id", log_path.stem)}
        ordered.update({k: v for k, v in merged.items() if k != "label"})
        if "label" in meta:
            ordered["label"] = meta["label"]
        rows.append(ordered)
        print(f"  parsed {log_path.name} ({len(parsed) - 1} metric features)")

    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    if "label" in all_keys:
        all_keys.remove("label")
        all_keys.append("label")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {len(rows)} rows x {len(all_keys)} columns to {output_csv}")
    print("Open it in Excel or load into pandas:")
    print(f"    import pandas as pd; df = pd.read_csv('{output_csv}')")


# ----------------------------------------------------------------------------
# Self-test: parse the sample log
# ----------------------------------------------------------------------------

SAMPLE_LOG = """Maximum precision of model and its entities: 1e-08 m.
Absolute small feature tolerance: 1e-05 m.
Mesh quality metrics:
tetEdgeRatio
\tmin: 1.0000755085514497
\tmax: 27.94032903448313
\taverage: 1.6957980037475855
\tstandard deviation 0.30480053204851204
\tmedian: 1.675521962623866
\t\t99.9-th percentile: 3.305802333765203
\t\t99.99-th percentile: 4.613885116717533
\t\t99.999-th percentile: 18.0046623980038

quadMaxAngle
\tmin: 90
\tmax: 169.12941581241654
\taverage: 90.28347238842943
\tstandard deviation 2.673255825229437
\tmedian: 90
\t\t99.9-th percentile: 133.91851200328554
\t\t99.99-th percentile: 165.31547427734534
\t\t99.999-th percentile: 169.11911814357978

triMaxAngle
\tmin: 60.00000000000001
\tmax: 156.63576204272016
\taverage: 79.83175662980409
\tstandard deviation 12.801729867784928
\tmedian: 77.06190491206804
\t\t99.9-th percentile: 117.52939492622025
\t\t99.99-th percentile: 124.06728101990065
\t\t99.999-th percentile: 129.78996778810316

triMinAngle
\tmin: 1.6230726938805564
\tmax: 60.00000000000001
\taverage: 45.21734656619261
\tstandard deviation 8.306105194774114
\tmedian: 45.09489721761922
\t\t99.9-th percentile: 59.99750216578822
\t\t99.99-th percentile: 59.99999999999991
\t\t99.999-th percentile: 60.00000000000001

volumeRatio
\tmin: 1
\tmax: 227.58443397804828
\taverage: 1.3421724428090576
\tstandard deviation 0.6991048218264179
\tmedian: 1.0983691799713413
\t\t99.9-th percentile: 6.000000000000112
\t\t99.99-th percentile: 10.000000000000016
\t\t99.999-th percentile: 11.472912995279165

tetAspectRatio
\tmin: 1.0000503472711662
\tmax: 174.88030382495447
\taverage: 1.5692659999156937
\tstandard deviation 0.3450655906048141
\tmedian: 1.588400839252388
\t\t99.9-th percentile: 2.272764045933034
\t\t99.99-th percentile: 3.1114586713359933
\t\t99.999-th percentile: 11.273662383183506

nonOrthogonality
\tmin: 0
\tmax: 89.76684687521428
\taverage: 12.506651704581033
\tstandard deviation 11.924319450241184
\tmedian: 10.627448376636563
\t\t99.9-th percentile: 55.26405779820152
\t\t99.99-th percentile: 64.5615260564646
\t\t99.999-th percentile: 77.50091877516007

skewness
\tmin: 0
\tmax: 3.5147116799818288
\taverage: 0.14167662845037743
\tstandard deviation 0.13593318386263833
\tmedian: 0.12499999999998249
\t\t99.9-th percentile: 1.0978856816171634
\t\t99.99-th percentile: 1.1582305177551766
\t\t99.999-th percentile: 2.465293791847433

aspectRatio
\tmin: 0.03477709091076951
\tmax: 174.88030382495447
\taverage: 1.4043321902020858
\tstandard deviation 0.44357222482390024
\tmedian: 1.4711912779519103
\t\t99.9-th percentile: 2.247312498266706
\t\t99.99-th percentile: 2.545242394567671
\t\t99.999-th percentile: 9.230673544938659


Overall mesh quality (based on the 99-percentile): 0.673940

Overall mesh quality is computed from:
\tNon Orthogonality: 40.609775 (normalized value: 0.75916, weight: 0.60)
\tSkewness: 0.501160 (normalized value: 0.605827, weight: 0.30)
\tAspect Ratio: 2.002511 (normalized value: 0.366957, weight: 0.10)
"""


def self_test() -> None:
    """Verify the parser against the user's sample log."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(SAMPLE_LOG)
        tmp_path = Path(f.name)

    try:
        row = parse_log_file(tmp_path)
        checks = {
            "nonOrthogonality_average": 12.506651704581033,
            "nonOrthogonality_max": 89.76684687521428,
            "nonOrthogonality_p99_9": 55.26405779820152,
            "skewness_average": 0.14167662845037743,
            "skewness_p99_999": 2.465293791847433,
            "aspectRatio_max": 174.88030382495447,
            "tetEdgeRatio_p99_99": 4.613885116717533,
            "overall_quality": 0.673940,
            "nonOrtho_component": 40.609775,
        }
        all_ok = True
        for key, expected in checks.items():
            got = row.get(key)
            ok = got is not None and abs(got - expected) < 1e-6
            mark = "OK" if ok else "FAIL"
            print(f"  [{mark}] {key}: got {got}, expected {expected}")
            if not ok:
                all_ok = False
        print()
        print(f"Total features extracted: {len(row) - 1}")
        print(f"Self-test {'PASSED' if all_ok else 'FAILED'}")
    finally:
        tmp_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--self-test":
        self_test()
    else:
        log_dir = Path(args[0]) if len(args) >= 1 else Path("./logs")
        output_csv = Path(args[1]) if len(args) >= 2 else Path("./mesh_dataset.csv")
        build_dataset(log_dir, output_csv)

        