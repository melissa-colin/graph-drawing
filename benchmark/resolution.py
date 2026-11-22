"""Resolution sweep: which drawing method keeps a readable IMAGE when the image gets small or the nodes big,
and does optimizing in screen space (rather than geometrically) make the difference?

Settings: image side in SIZES x node radius in RADII (edges 1 px), plus one with thick edges.
Methods, all put in the frame the same way (resolution.rescale_to_frame):
  random, kk (Kamada-Kawai), deepgd, coregd, ours (our model), ours500 (our model + the team's refinement
  done at 500 px / 4 px), and four refinements AT THE TARGET SETTING:
  {kk, ours} + {screen (the team's SDF terms), geometric (the control, metrics/geometric.py)}.
For each setting and kind, the refinement weights and learning rate are chosen on VALIDATION graphs (rule
resolution.choose, the same for both kinds); the test graphs are only measured.

    python benchmark/resolution.py run runs/model.pt [--workers 48] [--val 200] [--quick]
    python benchmark/resolution.py report

`run` needs benchmark/out/positions.pt (benchmark/compare.py: the test drawings of every method) and writes
benchmark/out/resolution/: rows.csv (one line per test drawing and setting), selection.json, val_starts.pt.
`report` reads them and writes results_resolution.md (means with 95 % bootstrap intervals, paired Wilcoxon
tests screen vs geometric with Holm correction) and the figures *.png.
Run it under the carbon tracker on deepeirb:
    python src/tools/run_tracked.py --name resolution benchmark/resolution.py run runs/model.pt
"""
import argparse
import csv
import json
import multiprocessing
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = HERE / "out" / "resolution"
DATA = HERE.parent / "data" / "rome"
SIZES = (100, 150, 200, 300, 500)
RADII = (2.0, 4.0, 8.0)
SETTINGS = [(s, r, 1.0) for s in SIZES for r in RADII] + [(200, 4.0, 3.0)]
COMPARE_NAMES = {"Aléatoire": "random", "Kamada-Kawai": "kk", "DeepGD": "deepgd", "CoRe-GD": "coregd",
                 "Notre modèle": "ours", "Notre modèle + affinage": "ours500"}
REFINED = ["kk", "ours"]


# -----------------------------------------------------------------------------------------------------------
# run
# -----------------------------------------------------------------------------------------------------------
def kamada_kawai(g):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(g.num_nodes))
    G.add_edges_from(g.edge_index.t().tolist())
    p = nx.kamada_kawai_layout(G)
    return torch.tensor(np.array([p[i] for i in range(g.num_nodes)]), dtype=torch.float32)


def validation_starts(checkpoint, graphs, device):
    """Starting drawings of the refined methods on the validation graphs (cached: KK is slow)."""
    cache = OUT / "val_starts.pt"
    names = [g.name for g in graphs]
    if cache.exists():
        saved = torch.load(cache)
        if saved["names"] == names:
            return saved["starts"]
    from infer import load_model
    from loss import Loss
    from refine import ROUNDS, draw
    model = load_model(Path(checkpoint)).to(device)
    starts = {"kk": [kamada_kawai(g) for g in graphs],
              "ours": [draw(model, g.to(device), Loss({}), ROUNDS, 0).positions.cpu() for g in graphs]}
    torch.save({"names": names, "starts": starts}, cache)
    return starts


def run(args):
    from data.input import load_dataset, split_dataset
    from resolution import FRACTIONS, LEARNING_RATES, STEPS, sweep_setting

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = load_dataset(DATA)
    _, val_files, test_files = split_dataset(DATA)
    val_files = sorted(random.Random(0).sample(val_files, min(args.val, len(val_files))))
    val_graphs = [graphs[p.stem] for p in val_files]
    test_graphs = [graphs[p.stem] for p in test_files]

    positions = torch.load(HERE / "out" / "positions.pt")
    test = {short: [(positions[long][g.name], g) for g in test_graphs] for long, short in COMPARE_NAMES.items()}
    val_start = validation_starts(args.checkpoint, val_graphs, device)
    val = {m: list(zip(val_start[m], val_graphs)) for m in REFINED}

    settings, fractions, rates, steps = SETTINGS, FRACTIONS, LEARNING_RATES, STEPS
    if args.quick:  # a few minutes: checks the whole chain
        settings, fractions, rates = [(150, 4.0, 1.0), (500, 4.0, 1.0)], (0.1,), (1.0,)
        test = {m: d[:20] for m, d in test.items()}
        val = {m: d[:20] for m, d in val.items()}

    print(f"device {device} | {len(val[REFINED[0]])} validation / {len(test['kk'])} test graphs | "
          f"{len(settings)} settings | {len(fractions) * len(rates)} candidates per kind")
    rows_path, selection = OUT / ("rows_quick.csv" if args.quick else "rows.csv"), {}
    writer, rows_file = None, open(rows_path, "w", newline="")
    with multiprocessing.get_context("spawn").Pool(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for size, radius, width in settings:
            t = time.perf_counter()
            rows, chosen = sweep_setting(size, radius, width, val, test, REFINED, device, pool,
                                         fractions, rates, steps)
            key = f"{size}px_r{radius:g}_w{width:g}"
            selection[key] = chosen
            for row in rows:
                row = {"size": size, "radius": radius, "width": width, **row}
                if writer is None:
                    writer = csv.DictWriter(rows_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            rows_file.flush()
            print(f"{key}: {time.perf_counter() - t:.0f} s | chosen " +
                  ", ".join(f"{k}: fraction {c['fraction']}, lr {c['lr']}" for k, c in chosen.items()), flush=True)
    rows_file.close()
    (OUT / ("selection_quick.json" if args.quick else "selection.json")).write_text(json.dumps(selection, indent=1))


# -----------------------------------------------------------------------------------------------------------
# report
# -----------------------------------------------------------------------------------------------------------
LABELS = {"random": "Aléatoire", "kk": "Kamada-Kawai", "deepgd": "DeepGD", "coregd": "CoRe-GD",
          "ours": "Notre modèle", "ours500": "Notre modèle + affinage à 500 px",
          "kk+screen": "KK + affinage écran", "kk+geometric": "KK + affinage géométrique",
          "ours+screen": "Notre modèle + affinage écran", "ours+geometric": "Notre modèle + affinage géométrique"}
MEASURES = {"hidden_nodes": "Nœuds cachés", "merged_nodes": "Nœuds fusionnés",
            "ambiguous_edges": "Arêtes ambiguës / arête", "edge_overlap": "Arêtes confondues",
            "node_visibility": "Visibilité des nœuds", "stress": "Stress", "crossings": "Croisements"}
# Figures: categorical slots of the dataviz reference palette, in its validated order; each method keeps its
# color in every figure. Markers differ too: identity never rests on color alone.
STYLE = {"ours+screen": ("#2a78d6", "o"), "kk": ("#eb6834", "s"), "coregd": ("#1baf7a", "^"),
         "deepgd": ("#eda100", "D"), "ours": ("#e87ba4", "v")}
START_STYLE = {"ours": ("#2a78d6", "o"), "kk": ("#eb6834", "s")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k != "method":
                r[k] = float(r[k])
    return rows


def by_setting(rows):
    table = {}
    for r in rows:
        table.setdefault((r["size"], r["radius"], r["width"]), {}).setdefault(r["method"], []).append(r)
    for methods in table.values():
        for drawings in methods.values():
            drawings.sort(key=lambda r: r["graph"])
    return table


def report(args):
    from stats import bootstrap_ci, holm, paired_test

    rows_path = OUT / ("rows_quick.csv" if args.quick else "rows.csv")
    table = by_setting(load_rows(rows_path))
    lines = [f"# Balayage en résolution ({rows_path.name})", "",
             "Moyenne sur les graphes de test [IC à 95 % par bootstrap]. Nœuds de rayon r, arêtes de largeur w, "
             "image s × s. Flèche : sens du mieux.", ""]
    arrows = {"node_visibility": "↑"}
    for (size, radius, width), methods in sorted(table.items()):
        lines += [f"## s = {size:g} px, r = {radius:g} px, w = {width:g} px", "",
                  "| méthode | " + " | ".join(f"{MEASURES[m]} {arrows.get(m, '↓')}" for m in MEASURES) + " |",
                  "|---|" + "---|" * len(MEASURES)]
        for method in LABELS:
            if method in methods:
                cells = []
                for m in MEASURES:
                    mean, lo, hi = bootstrap_ci([r[m] for r in methods[method]])
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]" if m != "crossings" else f"{mean:.1f} [{lo:.1f}, {hi:.1f}]")
                lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
        lines.append("")

    # screen vs geometric, paired by graph, Holm over every (setting, start, measure)
    tests = []
    for setting, methods in sorted(table.items()):
        for start in REFINED:
            for m in ("hidden_nodes", "merged_nodes", "ambiguous_edges", "stress"):
                a = [r[m] for r in methods[f"{start}+screen"]]
                b = [r[m] for r in methods[f"{start}+geometric"]]
                diff, p = paired_test(a, b)
                tests.append((setting, start, m, diff, p))
    adjusted = holm([t[4] for t in tests])
    lines += ["## Écran contre géométrique (témoin)", "",
              "Différence moyenne écran − géométrique, appariée par graphe (négatif : l'écran fait mieux sur ces "
              f"mesures, toutes à minimiser). Wilcoxon signé, p corrigé par Holm sur les {len(tests)} tests.", "",
              "| s | r | w | départ | mesure | écran − géométrique | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for (setting, start, m, diff, _), p in zip(tests, adjusted):
        mark = " *" if p < 0.05 else ""
        lines.append(f"| {setting[0]:g} | {setting[1]:g} | {setting[2]:g} | {start} | {MEASURES[m]} | {diff:+.4f} | {p:.2g}{mark} |")
    (OUT / ("results_resolution_quick.md" if args.quick else "results_resolution.md")).write_text("\n".join(lines) + "\n")
    figures(table, args.quick)
    print("\n".join(lines[:40]))


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def figures(table, quick):
    """Per measure: one panel per node radius, image size on x; methods as lines with their 95 % interval.
    And the difference screen - geometric per start, with its interval, around 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stats import bootstrap_ci

    suffix = "_quick" if quick else ""
    radii = sorted({k[1] for k in table if k[2] == 1.0})
    sizes = sorted({k[0] for k in table if k[2] == 1.0})
    for m in ("hidden_nodes", "merged_nodes", "ambiguous_edges", "stress"):
        for name, series in (("methods", STYLE), ("screen_vs_geometric", START_STYLE)):
            fig, axes = plt.subplots(1, len(radii), figsize=(3.6 * len(radii), 3.2), sharey=True, squeeze=False)
            fig.patch.set_facecolor(SURFACE)
            for ax, radius in zip(axes[0], radii):
                _axes_style(ax)
                for method, (color, marker) in series.items():
                    stats_ = []
                    for s in sizes:
                        methods = table[(s, radius, 1.0)]
                        if name == "methods":
                            values = [r[m] for r in methods[method]]
                        else:
                            values = [a[m] - b[m] for a, b in zip(methods[f"{method}+screen"], methods[f"{method}+geometric"])]
                        stats_.append(bootstrap_ci(values))
                    mean, lo, hi = map(np.array, zip(*stats_))
                    label = LABELS[method] if name == "methods" else f"{LABELS[method]} : écran − géométrique"
                    ax.fill_between(sizes, lo, hi, color=color, alpha=0.10, linewidth=0)
                    ax.plot(sizes, mean, color=color, linewidth=2, marker=marker, markersize=6,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, solid_capstyle="round")
                if name != "methods":
                    ax.axhline(0, color=MUTED, linewidth=1)
                    ax.annotate("sous 0 : l'écran fait mieux", xy=(0.98, 0.02), xycoords="axes fraction",
                                ha="right", va="bottom", fontsize=8, color=MUTED)
                ax.set_title(f"nœuds de rayon {radius:g} px", color=INK, fontsize=10)
                ax.set_xlabel("côté de l'image (px)", color=MUTED, fontsize=9)
                ax.set_xticks(sizes)
            axes[0][0].set_ylabel(MEASURES[m] if name == "methods" else f"Δ {MEASURES[m]}", color=MUTED, fontsize=9)
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False, fontsize=9,
                       labelcolor=INK, bbox_to_anchor=(0.5, 1.08))
            fig.tight_layout()
            fig.savefig(OUT / f"{m}_{name}{suffix}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("checkpoint")
    p_run.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 4))
    p_run.add_argument("--val", type=int, default=200, help="validation graphs used to choose the refinement")
    p_run.add_argument("--quick", action="store_true")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--quick", action="store_true")
    a = parser.parse_args()
    run(a) if a.command == "run" else report(a)
