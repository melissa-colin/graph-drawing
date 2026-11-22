"""Tools of the resolution sweep (benchmark/resolution.py): does optimizing in screen space pay off, and when?

Every drawing is put in a size x size frame with nodes of radius r and edges of width w, then measured on its
rendered image (metrics/image_quality.py) and with the geometric measures (stress, crossings). Geometric
methods (Kamada-Kawai, DeepGD, CoRe-GD) do not know at which size they will be displayed; a refinement in
screen space at the target size can adapt to it. To attribute a gain to the screen space itself, the same
refinement runs with geometric versions of its three screen terms (metrics/geometric.py): the control.

Pieces:
- TermLoss / make_loss(kind, weights): the refinement loss of train.REFINE_WEIGHTS, with kind = "screen"
  (the team's SDF terms) or "geometric" (the control); stress, crossings and angles are shared.
- rescale_to_frame: any drawing into the frame, same treatment for every method.
- with_style: a copy of a graph with another node radius and edge width.
- balanced_weights: weight of each term = fraction / its mean value on the starting drawings (the rule used to
  choose train.PHASE_2_WEIGHTS, DeepGD's Adaptive Weight), recomputed for each setting.
- refine_all: src/refine.refine on many graphs, by batches of graphs of the same setting.
- evaluate: every measure of one drawing.
"""
import copy

import torch
from torch_geometric.data import Batch

from data.output import Layout
from metrics import geometric
from metrics.edge_angles import edge_angles
from metrics.edge_crossings import count_crossings, edge_crossings
from metrics.image_quality import image_quality
from metrics.overlaps_vectorised import edge_node_overlaps, edge_overlaps, node_overlaps
from metrics.stress import stress
from refine import refine

SHARED_TERMS = {"stress": stress, "angles": edge_angles, "crossings": edge_crossings}
SCREEN_TERMS = {"node_overlaps": node_overlaps, "edge_overlaps": edge_overlaps, "edge_node_overlaps": edge_node_overlaps}
TERMS = {
    "screen": {**SHARED_TERMS, **SCREEN_TERMS},
    "geometric": {**SHARED_TERMS, **geometric.GEOMETRIC},
}


class TermLoss:
    """Weighted sum of terms, called like loss.Loss: loss(layout, graph) -> (total, {name: value})."""

    def __init__(self, terms, weights):
        unknown = set(weights) - set(terms)
        if unknown:
            raise ValueError(f"unknown terms {sorted(unknown)}, expected some of {sorted(terms)}")
        self.terms, self.weights = terms, dict(weights)

    def __call__(self, layout, graph):
        total = layout.positions.new_zeros(())
        values = {}
        for name, weight in self.weights.items():
            if weight == 0:
                continue
            value = self.terms[name](layout, graph)
            total = total + weight * value
            values[name] = value.detach()
        return total, values


def make_loss(kind, weights):
    return TermLoss(TERMS[kind], weights)


def rescale_to_frame(positions, size, margin):
    """Center the bounding box of the drawing in the size x size frame and scale it uniformly so that its
    largest side spans size - 2 margin (every node inside [margin, size - margin]). A single node, or nodes all
    at the same place, go to the center."""
    P = torch.as_tensor(positions, dtype=torch.float32)
    lo, hi = P.min(0).values, P.max(0).values
    span = float((hi - lo).max())
    scale = (size - 2 * margin) / span if span > 1e-9 else 0.0
    return (P - (lo + hi) / 2) * scale + size / 2


def with_style(graph, radius, edge_width):
    """Shallow copy of graph whose nodes all have radius `radius` and edges width `edge_width`."""
    styled = copy.copy(graph)
    styled.node_size = torch.full((graph.num_nodes,), float(radius))
    styled.edge_width = float(edge_width)
    return styled


@torch.no_grad()
def term_means(starts, kind, size):
    """Mean value of every term of `kind` over the starting drawings starts = [(positions, styled graph)]."""
    sums = dict.fromkeys(TERMS[kind], 0.0)
    for P, g in starts:
        layout = Layout(torch.as_tensor(P, dtype=torch.float32), size, size)
        for name, fn in TERMS[kind].items():
            sums[name] += float(fn(layout, g))
    return {name: s / max(len(starts), 1) for name, s in sums.items()}


def balanced_weights(means, fraction):
    """Stress at 1, every other term at `fraction` of its balanced weight 1 / mean (so that it weighs about
    `fraction` of the stress on the starting drawings). A term absent from the starting drawings (mean 0) gets
    the weight of its smallest nonzero sibling, so that it still acts if the refinement creates such defects."""
    weights = {"stress": 1.0}
    for name, mean in means.items():
        if name != "stress" and mean > 1e-12:
            weights[name] = fraction / mean
    fallback = min((w for n, w in weights.items() if n != "stress"), default=fraction)
    for name in means:
        weights.setdefault(name, fallback)
    return weights


def refine_all(positions, graphs, size, loss_fn, steps, lr, margin, device="cpu", batch_size=64):
    """refine() on every drawing (positions[i] of graphs[i]), by batches of `batch_size` graphs.
    Returns the refined positions, on the CPU, in the same order."""
    out = []
    for start in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
        P = torch.cat([torch.as_tensor(p, dtype=torch.float32) for p in positions[start:start + batch_size]]).to(device)
        refined = refine(Layout(P, size, size), batch, loss_fn, steps=steps, lr=lr, margin=margin).positions.cpu()
        sizes = [g.num_nodes for g in graphs[start:start + batch_size]]
        out.extend(refined.split(sizes))
    return out


def scale_normalized_stress(P, graph):
    """Stress after the best uniform scaling, divided by the number of pairs: benchmark/compare.stress."""
    i, j = graph.dist_index
    d = graph.dist_value.double()
    e = (P[i] - P[j]).double().norm(dim=1)
    w = d ** -2
    s = (w * e * d).sum() / (w * e * e).sum().clamp_min(1e-12)
    return float((w * (s * e - d) ** 2).sum() / len(d))


def evaluate(positions, graph, size):
    """Image measures at this size (graph carries the node radius and edge width), stress and crossings."""
    P = torch.as_tensor(positions, dtype=torch.float32)
    measures = image_quality(P, graph, size, size)
    measures["stress"] = scale_normalized_stress(P, graph)
    measures["crossings"] = float(count_crossings(Layout(P, size, size), graph))
    return measures


def defect_score(measures):
    """Selection criterion of the sweep (declared before looking at the test): hidden + merged nodes +
    ambiguous edges, the three defects that a reader cannot repair by looking closer."""
    return measures["hidden_nodes"] + measures["merged_nodes"] + measures["ambiguous_edges"]


def mean_measures(rows):
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
