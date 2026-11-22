"""Energy and carbon footprint of our computations (training, benchmarks, sweeps, demo), for the environmental
impact analysis required at the defense.

    from tools.carbon import track
    with track("resolution-sweep"):
        ...

Each block adds one row to runs/emissions.csv (codecarbon's format: duration in s, energy_consumed in kWh,
emissions in kg CO2eq, cpu_model, gpu_model, ...). The tracker is codecarbon's OfflineEmissionsTracker: no
network call, carbon intensity of the French grid. tracking_mode="process" counts only our process, which is
what we want on a shared server (deepeirb); "machine" counts the whole machine.

To run an existing script under the tracker without editing it, see run_tracked.py.
"""
import csv
import logging
import os
from contextlib import contextmanager
from pathlib import Path

DEFAULT_CSV = Path("runs/emissions.csv")


def visible_gpus():
    """GPUs this process may use, from CUDA_VISIBLE_DEVICES (physical indices or UUIDs, as NVML numbers them):
    None if the variable is not set (every GPU), [] if it is empty (no GPU)."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    return [int(x) if x.strip().isdigit() else x.strip() for x in value.split(",") if x.strip()]


@contextmanager
def track(name, out=DEFAULT_CSV, tracking_mode="process", measure_power_secs=15, gpu_ids="visible"):
    """Measure the block and append one row named `name` to the CSV `out`, even if the block raises.

    Yields the codecarbon tracker. measure_power_secs: sampling period of the power (s).
    gpu_ids: GPUs to measure; "visible" (default): those of CUDA_VISIBLE_DEVICES, every GPU if it is not set.
    codecarbon measures every GPU of the machine otherwise, including GPUs used by other jobs or idle: on a
    2-GPU server, a job on one GPU was charged the idle power of the other.
    """
    if gpu_ids == "visible":
        gpu_ids = visible_gpus()
    from codecarbon import OfflineEmissionsTracker  # imported here: only the measured runs need it

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tracker = OfflineEmissionsTracker(
        country_iso_code="FRA", project_name=name, output_dir=str(out.parent), output_file=out.name,
        tracking_mode=tracking_mode, measure_power_secs=measure_power_secs, gpu_ids=gpu_ids,
        log_level="error", allow_multiple_runs=True)
    logging.getLogger("codecarbon").setLevel(logging.ERROR)  # its warnings would clutter the progress bars
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()


def read_runs(path=DEFAULT_CSV):
    """Rows of the CSV as dicts: name, duration_s, energy_kwh, co2_g, cpu, gpu (one per measured block)."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{
            "name": row["project_name"],
            "duration_s": float(row["duration"]),
            "energy_kwh": float(row["energy_consumed"]),
            "co2_g": 1000 * float(row["emissions"]),
            "cpu": row["cpu_model"],
            "gpu": row["gpu_model"],
        } for row in csv.DictReader(f)]


def markdown_table(runs):
    """Markdown table of read_runs(), with a total line: the table of docs/notes.md, section Impact."""
    lines = ["| Expérience | Machine | Durée | Énergie (kWh) | CO₂eq (g) |", "|---|---|---|---|---|"]
    for r in runs:
        machine = r["gpu"] or r["cpu"]
        lines.append(f"| {r['name']} | {machine} | {_duration(r['duration_s'])} | {r['energy_kwh']:.4f} | {r['co2_g']:.2f} |")
    lines.append(f"| **Total** | | {_duration(sum(r['duration_s'] for r in runs))} | "
                 f"{sum(r['energy_kwh'] for r in runs):.4f} | {sum(r['co2_g'] for r in runs):.2f} |")
    return "\n".join(lines)


def _duration(seconds):
    minutes, s = divmod(round(seconds), 60)
    h, m = divmod(minutes, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


if __name__ == "__main__":
    import sys
    print(markdown_table(read_runs(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)))
