"""Turn `ncu --import <rep> --csv` output into a per-kernel markdown table.

One ncu capture holds several kernel launches (the fused custom kernel plus
OpenCV's internal LK kernels), each reported as many one-metric-per-line rows.
This pivots them to one row per launch with the memory-bound counters as
columns, and labels which arm each launch belongs to so the fused kernel's
traffic reads directly against OpenCV's.

Usage: parse_ncu.py <counters.csv>   (or piped on stdin)
"""
import csv
import re
import sys

# Column order and the short headers they print under. Keys are ncu metric
# names as they appear in the CSV's "Metric Name" field.
METRIC_LABELS = {
    "dram__bytes_read.sum": "DRAM read",
    "dram__bytes_write.sum": "DRAM write",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM %peak",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM %peak",
    "l1tex__t_sector_hit_rate.pct": "L1 hit%",
    "lts__t_sector_hit_rate.pct": "L2 hit%",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Occupancy%",
    "launch__registers_per_thread": "Regs/thread",
}

CUSTOM_KERNEL = "lk_warp_kernel"


def short_kernel_name(name: str) -> str:
    stem = name.split("<")[0] if "<" in name else name.split("(")[0]
    stem = stem.strip()
    if stem.startswith("void "):
        stem = stem[len("void "):]
    return stem.strip().split("::")[-1]


def load_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip('"').startswith("ID"):
            return list(csv.DictReader(iter(lines[index:])))
    return []


def pivot(rows: list[dict[str, str]]) -> list[dict]:
    launches: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        metric = row.get("Metric Name")
        if metric not in METRIC_LABELS:
            continue
        key = row.get("ID", row.get("Kernel Name", ""))
        if key not in launches:
            launches[key] = {"kernel": short_kernel_name(row["Kernel Name"]),
                             "metrics": {}}
            order.append(key)
        value = (row.get("Metric Value") or "").replace(",", "")
        launches[key]["metrics"][metric] = value
    return [launches[key] for key in order]


def format_value(raw: str) -> str:
    if raw == "":
        return "-"
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number >= 1000:
        return f"{number:,.0f}"
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}"


def render(launches: list[dict]) -> str:
    headers = ["Arm", "Kernel", *METRIC_LABELS.values()]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for launch in launches:
        arm = "custom" if CUSTOM_KERNEL in launch["kernel"] else "opencv"
        cells = [format_value(launch["metrics"].get(metric, ""))
                 for metric in METRIC_LABELS]
        lines.append("| " + " | ".join([arm, launch["kernel"], *cells]) + " |")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as handle:
            text = handle.read()
    else:
        text = sys.stdin.read()

    launches = pivot(load_rows(text))
    if not launches:
        print("no kernel counters found in input", file=sys.stderr)
        sys.exit(1)
    print(render(launches))


if __name__ == "__main__":
    main()
