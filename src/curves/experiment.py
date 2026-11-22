"""Curved edges vs straight edges, on settings where straight edges cannot avoid the nodes (benchmark/curves_experiment.py).

From the same refined drawing (nodes fixed), only the edges change: straight, or bent by optimize_curves.
Curve-term weights: bend at 1, every other term at `fraction` / its mean value on the straight drawings
(as resolution.balanced_weights); fraction and learning rate chosen on validation graphs by `choose_curves`,
declared before looking at the test.
"""
import torch
from torch_geometric.data import Batch

from curves.bezier import bend
from curves.losses import CURVE_TERMS, count_curve_crossings
from curves.optimize import curve_loss, optimize_curves
from curves.render import curve_elements
from data.output import Layout
from metrics.image_quality import image_quality

FRACTIONS = (0.1, 0.3, 1.0, 3.0)
LEARNING_RATES = (0.5, 1.0, 2.0)
STEPS = 50
CROSSING_TOLERANCE = 1.05


@torch.no_grad()
def curve_term_means(drawings, size):
    """Mean of each curve term on straight edges (offsets 0) over drawings = [(positions, graph)]."""
    sums = dict.fromkeys(CURVE_TERMS, 0.0)
    for P, g in drawings:
        layout = Layout(torch.as_tensor(P, dtype=torch.float32), size, size)
        zero = torch.zeros(g.edge_index.shape[1], 2)
        for name in sums:
            sums[name] += float(CURVE_TERMS[name](layout, zero, g))
    return {k: v / len(drawings) for k, v in sums.items()}


def curve_weights(means, fraction):
    weights = {"curve_bend": 1.0}
    for name, mean in means.items():
        if name != "curve_bend":
            weights[name] = fraction / mean if mean > 1e-12 else 0.0
    return weights


def bend_all(drawings, size, weights, steps, lr, device="cpu", batch_size=64):
    """optimize_curves on every drawing (nodes fixed), by batches. Returns the offsets of each drawing."""
    out = []
    for start in range(0, len(drawings), batch_size):
        chunk = drawings[start:start + batch_size]
        batch = Batch.from_data_list([g for _, g in chunk]).to(device)
        P = torch.cat([torch.as_tensor(p, dtype=torch.float32) for p, _ in chunk]).to(device)
        _, off = optimize_curves(Layout(P, size, size), batch, weights, steps=steps, lr=lr)
        out.extend(off.cpu().split([g.edge_index.shape[1] for _, g in chunk]))
    return out


def evaluate_curved(positions, offsets, graph, size):
    """Image measures with curved edges, curve crossings and mean bend of one drawing."""
    P = torch.as_tensor(positions, dtype=torch.float32)
    off = torch.as_tensor(offsets, dtype=torch.float32)
    measures = image_quality(P, graph, size, size, edges=curve_elements(P, off, graph, size, size))
    measures["crossings"] = count_curve_crossings(Layout(P, size, size), off, graph)
    measures["bend"] = float(bend(P, graph.edge_index, off).mean()) if graph.edge_index.shape[1] else 0.0
    return measures


def evaluate_all_curved(jobs, size, pool=None):
    args = [(P, off, g, size) for P, off, g in jobs]
    if pool is None:
        return [evaluate_curved(*a) for a in args]
    return list(pool.starmap(evaluate_curved, args, chunksize=max(1, len(args) // (4 * pool._processes))))


def mean(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def choose_curves(candidates, straight):
    """Lowest mean ambiguous_edges among the candidates with at most CROSSING_TOLERANCE x the crossings of the
    straight edges; if none, the fewest crossings. candidates: [(params, mean measures)]."""
    allowed = [c for c in candidates if c[1]["crossings"] <= CROSSING_TOLERANCE * straight["crossings"] + 1e-9]
    if allowed:
        return min(allowed, key=lambda c: (c[1]["ambiguous_edges"], c[1]["edge_overlap"]))
    return min(candidates, key=lambda c: c[1]["crossings"])


def curves_setting(size, val, test, device="cpu", pool=None, fractions=FRACTIONS, learning_rates=LEARNING_RATES,
                   steps=STEPS, batch_size=64):
    """val, test: [(positions in the frame, styled graph)], the refined straight drawings of one setting.
    Returns (rows, selection): rows = [{"edges": "straight" | "curved", "graph": i, measures...}]."""
    zero = lambda d: [(p, torch.zeros(g.edge_index.shape[1], 2), g) for p, g in d]
    straight_val = mean(evaluate_all_curved(zero(val), size, pool))
    means = curve_term_means(val, size)
    candidates = []
    for fraction in fractions:
        for lr in learning_rates:
            off = bend_all(val, size, curve_weights(means, fraction), steps, lr, device, batch_size)
            scores = mean(evaluate_all_curved([(p, o, g) for (p, g), o in zip(val, off)], size, pool))
            candidates.append(({"fraction": fraction, "lr": lr}, scores))
    params, scores = choose_curves(candidates, straight_val)
    weights = curve_weights(means, params["fraction"])
    selection = {**params, "weights": weights, "val": scores, "val_straight": straight_val,
                 "candidates": [{**p, **s} for p, s in candidates]}

    rows = [{"edges": "straight", "graph": i, **m} for i, m in enumerate(evaluate_all_curved(zero(test), size, pool))]
    off = bend_all(test, size, weights, steps, params["lr"], device, batch_size)
    curved = evaluate_all_curved([(p, o, g) for (p, g), o in zip(test, off)], size, pool)
    rows += [{"edges": "curved", "graph": i, **m} for i, m in enumerate(curved)]
    return rows, selection, off


__all__ = ["curves_setting", "curve_loss"]
