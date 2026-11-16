"""Evaluation measures of the graph drawing literature, to compare with published methods.

Definitions of (GD)^2 (Ahmed et al. 2020, section 4), also used by (SGD)^2, DeepGD and SmartGD on Rome.
For evaluation only (not differentiable, one graph at a time). All are invariant to the scale of the
drawing, so methods that do not draw in pixels (DeepGD, CoRe-GD, Kamada-Kawai) compare fairly with ours.
P: (N, 2) positions; graph: a GraphInput.
"""
import math

import torch


def _edges(graph):
    return graph.edge_index[0], graph.edge_index[1]


def ideal_edge_length(P, graph) -> float:
    """Q_IL, lower is better: relative deviation of the edge lengths from their mean (root mean square)."""
    u, v = _edges(graph)
    length = (P[u] - P[v]).norm(dim=1)
    mean = length.mean().clamp_min(1e-12)
    return float((((length - mean) / mean) ** 2).mean().sqrt())


def neighborhood_preservation(P, graph) -> float:
    """Q_NP, higher is better (in [0, 1]): Jaccard index between the graph neighbors of each node and its
    k nearest nodes in the drawing, k = its degree (as in t-SNET, Kruiger et al.)."""
    n = P.shape[0]
    u, v = _edges(graph)
    adjacency = torch.zeros(n, n, dtype=torch.bool)
    adjacency[u, v] = True
    adjacency[v, u] = True
    degree = adjacency.sum(1)
    D = torch.cdist(P, P)
    D.fill_diagonal_(float("inf"))
    rank = D.argsort(dim=1).argsort(dim=1)  # rank of j among the nearest nodes of i
    nearest = rank < degree[:, None]
    union = (nearest | adjacency).sum()
    return float((nearest & adjacency).sum() / union.clamp_min(1))


def _crossings(P, graph):
    """Pairs of crossing edges (without a common node): (a, b, c, d) endpoints of both edges."""
    u, v = _edges(graph)
    i, j = torch.triu_indices(u.numel(), u.numel(), offset=1)
    keep = (u[i] != u[j]) & (u[i] != v[j]) & (v[i] != u[j]) & (v[i] != v[j])
    i, j = i[keep], j[keep]
    a, b, c, d = P[u[i]], P[v[i]], P[u[j]], P[v[j]]

    def orient(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
    cross = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)
    return a[cross], b[cross], c[cross], d[cross]


def crossing_angle(P, graph) -> float:
    """Q_CAM, lower is better (in [0, 1]): worst deviation of a crossing angle from 90 degrees,
    max |theta - pi/2| / (pi/2) over the crossings; 0 without crossings."""
    a, b, c, d = _crossings(P, graph)
    if a.shape[0] == 0:
        return 0.0
    e1, e2 = b - a, d - c
    cos = ((e1 * e2).sum(1) / (e1.norm(dim=1) * e2.norm(dim=1)).clamp_min(1e-12)).abs().clamp(max=1)
    theta = torch.acos(cos)  # acute angle between the two edges
    return float(((theta - math.pi / 2).abs() / (math.pi / 2)).max())


def aspect_ratio(P, graph, rotations: int = 7) -> float:
    """Q_AR, higher is better (in [0, 1]): worst min(w, h) / max(w, h) of the bounding box over
    `rotations` rotations of the drawing."""
    P = P - P.mean(0)
    worst = 1.0
    for k in range(rotations):
        angle = 2 * math.pi * k / rotations
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=P.dtype)
        Q = P @ rotation.T
        w, h = (Q.max(0).values - Q.min(0).values).tolist()
        worst = min(worst, min(w, h) / max(max(w, h), 1e-12))
    return worst


def angular_resolution(P, graph) -> float:
    """Q_ANR, higher is better (in [0, 1]): smallest angle between two edges of a same node in the whole
    drawing, divided by its upper bound 2 pi / max degree."""
    u, v = _edges(graph)
    src, dst = torch.cat((u, v)), torch.cat((v, u))
    degree = torch.bincount(src, minlength=P.shape[0])
    if degree.max() < 2:
        return 1.0
    vec = P[dst] - P[src]
    angle = torch.atan2(vec[:, 1], vec[:, 0])
    smallest = 2 * math.pi
    for node in (degree >= 2).nonzero().flatten().tolist():
        a = angle[src == node].sort().values
        gaps = torch.cat((a[1:] - a[:-1], (a[:1] + 2 * math.pi) - a[-1:]))
        smallest = min(smallest, float(gaps.min()))
    return smallest / (2 * math.pi / int(degree.max()))


def vertex_resolution(P, graph) -> float:
    """VR, higher is better (in [0, 1]): shortest distance between two nodes / longest one."""
    D = torch.cdist(P, P)
    longest = D.max()
    D.fill_diagonal_(float("inf"))
    return float(D.min() / longest.clamp_min(1e-12))


MEASURES = {  # name: (function, True if higher is better)
    "ideal_edge_length": (ideal_edge_length, False),
    "neighborhood_preservation": (neighborhood_preservation, True),
    "crossing_angle": (crossing_angle, False),
    "aspect_ratio": (aspect_ratio, True),
    "angular_resolution": (angular_resolution, True),
    "vertex_resolution": (vertex_resolution, True),
}
