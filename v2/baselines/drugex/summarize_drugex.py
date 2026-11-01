#!/usr/bin/env python3
"""Aggregate completed DrugEx seed evaluations into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.eval_root.glob("*/metrics.json"))
    runs: Dict[str, Dict] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[path.parent.name] = payload.get("metrics", payload)
    metric_names = sorted({key for metrics in runs.values() for key, value in metrics.items() if isinstance(value, (int, float))})
    aggregate = {}
    for name in metric_names:
        values = [float(metrics[name]) for metrics in runs.values() if name in metrics]
        if not values:
            continue
        aggregate[name] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    summary = {"runs": runs, "aggregate": aggregate, "num_runs": len(runs)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    preferred = [
        "validity",
        "unique_rate",
        "novelty",
        "qed_mean",
        "sas_mean",
        "diversity",
        "mean_within_record_diversity",
        "scaffold_core_match_rate",
        "scaffold_core_retention_rate",
        "mean_unique_per_record",
    ]
    names = [name for name in preferred if name in aggregate] + [name for name in metric_names if name not in preferred and name in aggregate]
    lines = [
        "# DrugEx baseline summary",
        "",
        f"Completed evaluation runs: {len(runs)}",
        "",
        "| Metric | Mean | SD |",
        "|---|---:|---:|",
    ]
    for name in names:
        item = aggregate[name]
        lines.append(f"| {name} | {item['mean']:.6f} | {item['std']:.6f} |")
    lines.extend(["", "## Runs", ""])
    for name in sorted(runs):
        lines.append(f"- `{name}`")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
