"""Bend the edges of a drawing, by gradient descent in screen space, so that they avoid the nodes and each
other: after the refinement of the positions (src/refine.py), Adam on the control-point offsets of the edges
(curves/bezier.py), the positions fixed; or both together (joint=True).

The control point is kept inside the frame, which keeps the whole curve inside (a quadratic Bezier curve lies
in the triangle of its three points).
"""
import torch

from curves.bezier import control_points
from curves.losses import CURVE_TERMS
from data.output import Layout
from metrics.batching import static_cache

STEPS = 50
LR = 1.0  # pixels per step, roughly


def curve_loss(layout, offsets, graph, weights):
    """Weighted sum of the terms of curves/losses.CURVE_TERMS: (total, {name: value})."""
    total = layout.positions.new_zeros(())
    values = {}
    for name, weight in weights.items():
        if weight == 0:
            continue
        value = CURVE_TERMS[name](layout, offsets, graph)
        total = total + weight * value
        values[name] = value.detach()
    return total, values


def _clamp_controls(positions, edge_index, offsets, width, height, margin):
    """Move the control points back inside [margin, size - margin] (in place on offsets)."""
    with torch.no_grad():
        mid = control_points(positions, edge_index, torch.zeros_like(offsets))
        c = mid + offsets
        c[:, 0].clamp_(margin, width - margin)
        c[:, 1].clamp_(margin, height - margin)
        offsets.copy_(c - mid)


def optimize_curves(layout, graph, weights, steps=STEPS, lr=LR, offsets=None, joint=False, position_loss=None,
                    margin=None):
    """Returns (layout, offsets). weights: {term of CURVE_TERMS: weight}. offsets: start (straight if None).
    joint: also move the nodes, with position_loss(layout, graph) -> (total, values) added to the curve loss
    (e.g. resolution.make_loss with stress, node overlaps and angles; its straight-edge terms make no sense
    for curves). margin: frame margin of the nodes and control points (largest node radius by default).
    graph can be a PyG batch."""
    positions = layout.positions.detach().clone()
    edge_index = graph.edge_index.to(positions.device)
    offsets = (torch.zeros(edge_index.shape[1], 2, dtype=positions.dtype, device=positions.device)
               if offsets is None else offsets.detach().clone().to(positions))
    margin = float(graph.node_size.max()) if margin is None else margin
    params = [offsets.requires_grad_(True)]
    if joint:
        if position_loss is None:
            raise ValueError("joint=True needs a position_loss")
        params.append(positions.requires_grad_(True))
    optimizer = torch.optim.Adam(params, lr=lr)
    with torch.enable_grad(), static_cache(graph):
        for _ in range(steps):
            optimizer.zero_grad()
            current = Layout(positions, layout.width, layout.height)
            loss = curve_loss(current, offsets, graph, weights)[0]
            if joint:
                loss = loss + position_loss(current, graph)[0]
            if not loss.requires_grad:
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if joint:
                    positions[:, 0].clamp_(margin, layout.width - margin)
                    positions[:, 1].clamp_(margin, layout.height - margin)
            _clamp_controls(positions, edge_index, offsets, layout.width, layout.height, margin)
    return Layout(positions.detach(), layout.width, layout.height), offsets.detach()
