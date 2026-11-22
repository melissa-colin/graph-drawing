"""benchmark/resolution_sweep.py report: tables, paired tests and figures from a synthetic rows.csv."""
import csv
import importlib.util
import random
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("resolution_sweep", ROOT / "benchmark" / "resolution_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_end_to_end(tmp_path):
    script = load_script()
    script.OUT = tmp_path
    rng = random.Random(0)
    methods = ["random", "kk", "deepgd", "coregd", "ours", "ours500",
               "kk+screen", "kk+geometric", "ours+screen", "ours+geometric"]
    rows = []
    for size in (100, 500):
        for radius in (2.0, 4.0):
            for method in methods:
                for graph in range(12):
                    better = 0.5 if method.endswith("+screen") else 1.0  # screen halves the defects
                    rows.append({"size": size, "radius": radius, "width": 1.0, "method": method, "graph": graph,
                                 **{m: better * rng.uniform(0, 1) * 100 / size for m in
                                    ("hidden_nodes", "merged_nodes", "ambiguous_edges", "edge_overlap")},
                                 "node_visibility": rng.uniform(0.5, 1), "stress": rng.uniform(0.05, 0.07),
                                 "crossings": rng.randint(0, 40), "ink_density": 0.1, "ink_peak": 0.5})
    with open(tmp_path / "rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    script.report(types.SimpleNamespace(quick=False))

    md = (tmp_path / "results_resolution.md").read_text()
    assert md.count("## s = ") == 4
    assert "Écran contre les témoins géométriques" in md
    assert md.count("| kk | géométrique | Nœuds cachés |") == 4  # one paired test per setting, start, control, measure
    for m in ("hidden_nodes", "merged_nodes", "ambiguous_edges", "stress"):
        for name in ("methods", "screen_vs_geometric"):
            assert (tmp_path / f"{m}_{name}.png").stat().st_size > 10_000


def test_by_setting_sorts_by_graph():
    script = load_script()
    rows = [{"size": 100.0, "radius": 2.0, "width": 1.0, "method": "kk", "graph": g, "stress": float(g)}
            for g in (2, 0, 1)]
    table = script.by_setting(rows)
    assert [r["graph"] for r in table[(100.0, 2.0, 1.0)]["kk"]] == [0, 1, 2]
