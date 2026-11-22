"""Curved vs straight edges on hard settings: from the drawing of our model refined in screen space at the
setting (the refinement chosen by benchmark/resolution_sweep.py), the nodes stay where they are and only the
edges change, straight or bent in screen space (src/curves). Measures on the rendered image.

    python benchmark/curves.py run runs/model.pt [--workers 48] [--quick]
    python benchmark/curves.py report

Needs benchmark/out/positions.pt (compare.py) and benchmark/out/resolution/{selection.json, val_starts.pt}
(resolution_sweep.py run). Writes benchmark/out/curves/: rows.csv, selection.json, results_curves.md (means with
95 % bootstrap intervals, paired Wilcoxon tests curved vs straight with Holm correction) and sheet.png
(three test graphs, straight and curved, defects of the image in red).
"""
import argparse
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = HERE / "out" / "curves"
SWEEP = HERE / "out" / "resolution"
DATA = HERE.parent / "data" / "rome"
SETTINGS = [(200, 4.0, 1.0), (200, 4.0, 3.0), (300, 8.0, 1.0)]  # where straight edges hit nodes most
MEASURES = {"ambiguous_edges": "Arêtes ambiguës / arête", "edge_overlap": "Arêtes confondues",
            "crossings": "Croisements", "hidden_nodes": "Nœuds cachés", "ink_peak": "Encre (pic)", "bend": "Courbure"}


def refined_drawings(starts, graphs, size, radius, width, chosen, device):
    """Our model's drawings, in the frame, refined with the screen loss chosen by the sweep for this setting."""
    from resolution import make_loss, refine_all, rescale_to_frame, with_style
    styled = [with_style(g, radius, width) for g in graphs]
    framed = [rescale_to_frame(p, size, radius) for p in starts]
    P = refine_all(framed, styled, size, make_loss("screen", chosen["weights"]), 25, chosen["lr"], radius, device)
    return list(zip(P, styled))


def run(args):
    from curves.experiment import curves_setting
    from data.input import load_dataset, split_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = load_dataset(DATA)
    _, val_files, test_files = split_dataset(DATA)
    val_saved = torch.load(SWEEP / "val_starts.pt")
    val_graphs = [graphs[name] for name in val_saved["names"]]
    test_graphs = [graphs[p.stem] for p in test_files]
    positions = torch.load(HERE / "out" / "positions.pt")["Notre modèle"]
    test_starts = [positions[g.name] for g in test_graphs]
    val_starts = val_saved["starts"]["ours"]
    selection_sweep = json.loads((SWEEP / "selection.json").read_text())
    settings = SETTINGS
    if args.quick:
        settings, val_graphs, val_starts = SETTINGS[:1], val_graphs[:20], val_starts[:20]
        test_graphs, test_starts = test_graphs[:20], test_starts[:20]

    rows_path = OUT / ("rows_quick.csv" if args.quick else "rows.csv")
    selection, writer = {}, None
    with open(rows_path, "w", newline="") as f, \
            multiprocessing.get_context("spawn").Pool(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for size, radius, width in settings:
            t = time.perf_counter()
            key = f"{size}px_r{radius:g}_w{width:g}"
            chosen = selection_sweep[key]["screen"]
            val = refined_drawings(val_starts, val_graphs, size, radius, width, chosen, device)
            test = refined_drawings(test_starts, test_graphs, size, radius, width, chosen, device)
            kwargs = dict(fractions=(1.0,), learning_rates=(1.0,), steps=20) if args.quick else {}
            rows, chosen_curves, offsets = curves_setting(size, val, test, device, pool, **kwargs)
            selection[key] = chosen_curves
            torch.save({"positions": [p for p, _ in test], "offsets": offsets, "names": [g.name for g in test_graphs]},
                       OUT / f"drawings_{key}{'_quick' if args.quick else ''}.pt")
            for row in rows:
                row = {"size": size, "radius": radius, "width": width, **row}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            f.flush()
            print(f"{key}: {time.perf_counter() - t:.0f} s | fraction {chosen_curves['fraction']}, "
                  f"lr {chosen_curves['lr']}", flush=True)
    (OUT / ("selection_quick.json" if args.quick else "selection.json")).write_text(json.dumps(selection, indent=1))


def report(args):
    from stats import bootstrap_ci, holm, paired_test
    suffix = "_quick" if args.quick else ""
    with open(OUT / f"rows{suffix}.csv", newline="") as f:
        rows = [{k: (v if k == "edges" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    table = {}
    for r in rows:
        table.setdefault((r["size"], r["radius"], r["width"]), {}).setdefault(r["edges"], []).append(r)
    lines = [f"# Arêtes courbes contre arêtes droites ({OUT.name}/rows{suffix}.csv)", "",
             "Mêmes positions des nœuds (notre modèle affiné à l'écran au réglage) ; seules les arêtes changent. "
             "Moyenne sur les graphes de test [IC à 95 %].", ""]
    tests = []
    for setting, by_edges in sorted(table.items()):
        for kind in by_edges.values():
            kind.sort(key=lambda r: r["graph"])
        lines += [f"## s = {setting[0]:g} px, r = {setting[1]:g} px, w = {setting[2]:g} px", "",
                  "| arêtes | " + " | ".join(MEASURES.values()) + " |", "|---|" + "---|" * len(MEASURES)]
        for kind in ("straight", "curved"):
            cells = []
            for m in MEASURES:
                mean, lo, hi = bootstrap_ci([r[m] for r in by_edges[kind]])
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {'droites' if kind == 'straight' else 'courbes'} | " + " | ".join(cells) + " |")
        lines.append("")
        for m in ("ambiguous_edges", "edge_overlap", "crossings"):
            diff, p = paired_test([r[m] for r in by_edges["curved"]], [r[m] for r in by_edges["straight"]])
            tests.append((setting, m, diff, p))
    adjusted = holm([t[3] for t in tests])
    lines += ["## Courbes − droites, apparié par graphe", "",
              f"Wilcoxon signé, p corrigé par Holm sur les {len(tests)} tests (négatif : les courbes font mieux).", "",
              "| s | r | w | mesure | courbes − droites | p (Holm) |", "|---|---|---|---|---|---|"]
    for (setting, m, diff, _), p in zip(tests, adjusted):
        lines.append(f"| {setting[0]:g} | {setting[1]:g} | {setting[2]:g} | {MEASURES[m]} | {diff:+.4f} | {p:.2g}{' *' if p < 0.05 else ''} |")
    (OUT / f"results_curves{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sheet(suffix)


def defect_map(positions, offsets, graph, size):
    """(size, size) map of the pixels of an edge that touch a node it does not connect (for the sheet)."""
    from curves.render import curve_elements
    from metrics.image_quality import elements
    nodes, _ = elements(positions, graph, size, size)
    curves = curve_elements(positions, offsets, graph, size, size)
    owner = {}
    for k, e in enumerate(nodes):
        for p in e.pixels[e.coverage > 0.1]:
            owner.setdefault(int(p), set()).add(k)
    ends = graph.edge_index.t().tolist()
    hit = np.zeros(size * size)
    for (u, v), e in zip(ends, curves):
        for p, c in zip(e.pixels, e.coverage):
            if c > 0.1 and owner.get(int(p), set()) - {u, v}:
                hit[p] = 1.0
    from scipy import ndimage
    return ndimage.binary_dilation(hit.reshape(size, size), iterations=2).astype(float)


def sheet(suffix=""):
    """Three test graphs of the first setting: straight edges | curved edges, defects in red."""
    from PIL import Image, ImageDraw
    from curves.render import render_curved
    from data.input import load_dataset
    from data.raster import raster
    from resolution import with_style

    size, radius, width = SETTINGS[0]
    key = f"{size}px_r{radius:g}_w{width:g}"
    path = OUT / f"drawings_{key}{suffix}.pt"
    if not path.exists():
        return
    saved = torch.load(path)
    graphs = load_dataset(DATA)
    picks = sorted(range(len(saved["names"])), key=lambda i: graphs[saved["names"][i]].num_nodes)
    picks = [picks[len(picks) // 4], picks[len(picks) // 2], picks[-1]]
    scale, pad = 2, 12
    tiles = []
    for i in picks:
        g = with_style(graphs[saved["names"][i]], radius, width)
        P, off = saved["positions"][i], saved["offsets"][i]
        row = []
        for offsets in (torch.zeros_like(off), off):
            img = render_curved(P, offsets, g, size, size, highlight=defect_map(P, offsets, g, size))
            row.append(Image.fromarray(raster.to_uint8(img)).resize((size * scale, size * scale), Image.NEAREST))
        tiles.append(row)
    W = 2 * size * scale + 3 * pad
    H = len(tiles) * (size * scale + pad) + pad + 24
    canvas = Image.new("RGB", (W, H), "#fcfcfb")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), "arêtes droites", fill="#0b0b0b")
    draw.text((2 * pad + size * scale, 6), "arêtes courbes (mêmes nœuds)", fill="#0b0b0b")
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            canvas.paste(tile, (pad + c * (size * scale + pad), 24 + r * (size * scale + pad)))
    canvas.save(OUT / f"sheet{suffix}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("checkpoint")
    p_run.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 4))
    p_run.add_argument("--quick", action="store_true")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--quick", action="store_true")
    a = parser.parse_args()
    run(a) if a.command == "run" else report(a)
