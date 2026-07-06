"""Generate a plain-text benchmark report from pytest-benchmark output.

Reads ``output.json`` (pytest-benchmark) and ``equivalence_result.json``
(written by ``benchmark.py``) and writes a human-readable text
report to ``benchmark_report.txt``.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def _load_json(path):
    """Load a JSON file, return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def generate_report(benchmark_path, equivalence_path, output_path):
    """Build the text report and write it to *output_path*."""

    data = _load_json(benchmark_path)
    if data is None:
        print(f"ERROR: {benchmark_path} not found", file=sys.stderr)
        sys.exit(1)

    eq = _load_json(equivalence_path)

    rounds = data["benchmarks"][0]["stats"]["rounds"]
    now = datetime.now().isoformat(timespec='microseconds')

    lines = []
    lines.append("=================================================================")
    lines.append("  Meshdata() Profiling Report")
    lines.append("=================================================================\n")
    lines.append(f"Date            : {now}")
    lines.append(f"Rounds          : {rounds}\n\n")

    # Timing section per mode
    mode_stats = {}
    for b in data["benchmarks"]:
        name = b["name"].replace("test_meshdata_pipeline[", "").replace("]", "")
        s = b["stats"]
        mode_stats[name] = s
        
        lines.append("-----------------------------------------------------------------")
        lines.append(f"  MODE: {name.upper()}")
        lines.append("-----------------------------------------------------------------")
        lines.append(f"  Time (Mean) : {s['mean']:.4f} s")
        lines.append(f"  Min         : {s['min']:.4f} s")
        lines.append(f"  Max         : {s['max']:.4f} s")
        lines.append(f"  StdDev      : {s['stddev']:.4f}")
        lines.append(f"  OPS         : {s['ops']:.4f}\n")

    # Comparison
    lines.append("=================================================================")
    lines.append("  COMPARISON")
    lines.append("=================================================================")
    lines.append(f"  {'Mode':<45} {'Time'}")
    lines.append(f"  {'-'*45} {'--------'}")
    
    for name, s in mode_stats.items():
        lines.append(f"  {name:<45} {s['mean']:.2f}s")
    lines.append("")

    # Speedup (assuming baseline is first and we compare others to it)
    if len(mode_stats) > 1:
        baseline_name = list(mode_stats.keys())[0]
        baseline_mean = mode_stats[baseline_name]['mean']
        
        for name, s in mode_stats.items():
            if name != baseline_name:
                speedup = baseline_mean / s['mean']
                lines.append(f"  Speedup ({name} vs {baseline_name}):")
                lines.append(f"    {speedup:.2f}x ({baseline_mean:.2f}s -> {s['mean']:.2f}s)\n")

    # Equivalence checks
    if eq is not None:
        lines.append(f"  Values match ({eq['baseline']} vs others) : {eq['status'].upper()}")
        for mode, checks in eq["comparisons"].items():
            c = checks
            match_str = "YES" if c["num_nodes"]["match"] else "NO"
            lines.append(f"    {mode} node count matches : {match_str}")
            for metric in ("min", "max", "mean"):
                if c[metric]['rel_diff'] <= eq['rtol']:
                    lines.append(f"    {mode} {metric:<4} matches : YES (rel diff {c[metric]['rel_diff']:.2e})")
                else:
                    lines.append(f"    {mode} {metric:<4} matches : NO (rel diff {c[metric]['rel_diff']:.2e})")
    else:
        lines.append("  Values match  : UNKNOWN (equivalence_result.json not found)")

    lines.append("=================================================================\n")

    report = "\n".join(lines)
    Path(output_path).write_text(report)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    generate_report(
        benchmark_path="output.json",
        equivalence_path="equivalence_result.json",
        output_path="benchmark_report.txt",
    )
