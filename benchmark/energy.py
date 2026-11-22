"""Energy per drawing of each method, for the environmental impact analysis (docs/impact.md).

benchmark/compare.py measures all the methods in one block; here each one runs alone under the carbon tracker
(src/tools/carbon.py) on the same test graphs, one graph at a time as a user would draw them:
  kk               Kamada-Kawai (networkx, CPU)
  ours             our model, 12 rounds (GPU if available)
  ours+refine      our model, then the team's refinement at 500 px (refine.draw)
  refine@S         the refinement in screen space alone, at image side S (100, 200, 500 px), from KK drawings,
                   by batches of 64 graphs (the cost of drawing for a given screen)

    python benchmark/energy.py runs/model.pt [--graphs 200]
Writes benchmark/out/energy/energy.csv (codecarbon rows) and energy.md (Wh and seconds per drawing).
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

OUT = HERE / "out" / "energy"
DATA = HERE.parent / "data" / "rome"


def main():
    from data.input import load_dataset, split_dataset
    from infer import load_model
    from loss import Loss
    from refine import ROUNDS, STEPS, draw
    from resolution import (anchor, balanced_weights, make_loss, refine_all, rescale_to_frame, term_means,
                            with_style)
    from tools.carbon import read_runs, track

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--graphs", type=int, default=200)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "energy.csv"
    csv_path.unlink(missing_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = load_dataset(DATA)
    test = [graphs[p.stem] for p in split_dataset(DATA)[2][:args.graphs]]
    model = load_model(Path(args.checkpoint)).to(device)
    from train import REFINE_WEIGHTS
    refine_loss = Loss(REFINE_WEIGHTS)
    sys.path.insert(0, str(HERE))
    from resolution_sweep import kamada_kawai

    seconds, kk = {}, []

    def measured(name, fn):
        t = time.perf_counter()
        with track(f"energy-{name}", csv_path, measure_power_secs=1):
            result = fn()
            if device == "cuda":
                torch.cuda.synchronize()
        seconds[name] = time.perf_counter() - t
        return result

    kk = measured("kk", lambda: [kamada_kawai(g) for g in test])
    measured("ours", lambda: [draw(model, g.clone().to(device), refine_loss, ROUNDS, 0) for g in test])
    measured("ours+refine", lambda: [draw(model, g.clone().to(device), refine_loss, ROUNDS, STEPS) for g in test])
    for size in (100, 200, 500):
        framed = [rescale_to_frame(p, size, 4.0) for p in kk]
        styled = [anchor(with_style(g, 4.0, 1.0), p) for g, p in zip(test, framed)]
        loss = make_loss("screen", balanced_weights(term_means(list(zip(framed, styled)), "screen", size), 0.3))
        measured(f"refine@{size}", lambda: refine_all(framed, styled, size, loss, 25, 1.0, 4.0, device))

    runs = {r["name"].removeprefix("energy-"): r for r in read_runs(csv_path)}
    n = len(test)
    lines = [f"# Énergie par dessin ({n} graphes de test de Rome, {device}, {runs['kk']['gpu'] or runs['kk']['cpu']})", "",
             "| méthode | temps / dessin | énergie / dessin (mWh) | CO₂eq / dessin (mg) |", "|---|---|---|---|"]
    for name, r in runs.items():
        lines.append(f"| {name} | {1000 * seconds[name] / n:.1f} ms | {1e6 * r['energy_kwh'] / n:.3f} | "
                     f"{1000 * r['co2_g'] / n:.3f} |")
    (OUT / "energy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
