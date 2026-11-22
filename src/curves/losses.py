"""Screen-space losses for curved edges (curves/bezier.py), built like metrics/overlaps_vectorised.py: sums of
products of SDF coverages over the pixels where two shapes meet, batched over all pairs at once.

Every loss takes (layout, offsets, graph): offsets (E, 2) are the control-point offsets of the edges of
graph.edge_index. graph can be a PyG batch: sum per graph, mean over the graphs.

- curve_node_overlaps: area (px) where a curve covers a node that is not one of its ends. Exact coverage of
  the whole curve (min of the capsule SDFs) inside the box of each (curve, node) pair.
- curve_curve_overlaps: area where two curves without a common node overlap, with the exact coverage of
  each whole curve. The pixels are visited by pairs of SUB-SEGMENTS whose boxes touch (small boxes, instead of
  the box of two whole curves), and a pixel is counted only in the pair made of the segment of each curve
  nearest to it: every pixel once, even where several segments meet (joints).
- curve_crossings: for each pair of crossing curves without a common node, the smallest distance from an end
  of one curve to the other curve. For straight edges it is close to metrics/edge_crossings.py (distance of an
  end to the other edge). Not the smallest move of a sub-segment end, which is 0 when the crossing is at a
  joint and only slides the crossing to the next segment: here, with the nodes fixed, the gradient bends a
  curve beyond the end of the other one, which does uncross them. Crossings are found on sub-segments,
  half-open, so a crossing exactly at a joint is seen once.
- curve_bend: sum of |o|^2 / |p_u - p_v|^2: keeps edges straight unless curving pays off.
"""
import torch

from curves.bezier import K, bend, sample, segment_distance
from data.raster.sdf import coverage_margin, node_coverage_torch
from metrics.batching import cached, edge_width, node_graph
from metrics.edge_crossings import _candidate_pairs, _side


def _crosses(a, b, c, d):
    """Segments [a, b) and [c, d) cross: each one has its two ends on both sides of the other's line, a point
    ON the line counting as the positive side (half-open segments: a crossing exactly at a joint of a
    polyline is seen by exactly one of the two segments that meet there)."""
    return (((_side(a, b, c) >= 0) != (_side(a, b, d) >= 0)) & ((_side(c, d, a) >= 0) != (_side(c, d, b) >= 0)))
from metrics.overlaps_vectorised import _boxes, _col, batched_overlap


def _zero(pos, offsets):
    return pos.sum() * 0.0 + offsets.sum() * 0.0  # linked to both: backward always works


def curve_node_overlaps(layout, offsets, graph, softness=0.5, k=K):
    pos = layout.positions
    r = graph.node_size.to(device=pos.device, dtype=pos.dtype)
    margin = coverage_margin(softness)
    node_graph_index, num_graphs = node_graph(graph, pos.device)

    def static():  # (edge e, node k) of the same graph, k not an end of e
        u, v = graph.edge_index.to(pos.device)
        edge_graph = node_graph_index[u]
        k_all = torch.arange(pos.shape[0], device=pos.device)
        candidates = (edge_graph[:, None] == node_graph_index[None, :]) \
            & (u[:, None] != k_all[None, :]) & (v[:, None] != k_all[None, :])
        e, n = candidates.nonzero(as_tuple=True)
        return u, v, edge_width(graph, edge_graph, pos), e, n
    u, v, w, e, n = cached(graph, ("curve_node_pairs", str(pos.device)), static)
    pts = sample(pos, (u, v), offsets, k)  # (E, k + 1, 2), with gradient

    with torch.no_grad():
        reach = r[n] + w[e] / 2 + 2 * margin
        lo_c, hi_c = pts.min(dim=1).values, pts.max(dim=1).values
        near = ((lo_c[e] - reach[:, None] < pos[n]) & (pos[n] < hi_c[e] + reach[:, None])).all(dim=1)
        e, n, reach = e[near], n[near], reach[near]
        a = (pts[e, :-1, 0], pts[e, :-1, 1])
        b = (pts[e, 1:, 0], pts[e, 1:, 1])
        dist = segment_distance(pos[n, 0, None], pos[n, 1, None], a, b).amin(dim=1)
        keep = dist < reach
        e, n = e[keep], n[keep]
        node_lo, node_hi = pos[n] - (r[n] + margin)[:, None], pos[n] + (r[n] + margin)[:, None]
        curve_lo = lo_c[e] - (w[e] / 2 + margin)[:, None]
        curve_hi = hi_c[e] + (w[e] / 2 + margin)[:, None]
        x0, y0, x1, y1, keep = _boxes(torch.maximum(node_lo, curve_lo), torch.minimum(node_hi, curve_hi),
                                      layout.width, layout.height)
    e, n, x0, y0, x1, y1 = (t[keep] for t in (e, n, x0, y0, x1, y1))
    if len(e) == 0:
        return _zero(pos, offsets)
    curve_pts, we, pn, rn = pts[e], w[e], pos[n], r[n]

    def cov_curve(X, Y, idx):
        p = curve_pts[idx][:, None, None]  # (k, 1, 1, K + 1, 2)
        a = (p[..., :-1, 0], p[..., :-1, 1])
        b = (p[..., 1:, 0], p[..., 1:, 1])
        dist = segment_distance(X[..., None], Y[..., None], a, b).amin(dim=-1)
        return torch.sigmoid(-(dist - _col(we[idx]) / 2) / softness)

    def cov_node(X, Y, idx):
        return node_coverage_torch(X, Y, (_col(pn[idx, 0]), _col(pn[idx, 1])), _col(rn[idx]), softness)

    return batched_overlap(x0, y0, x1, y1, cov_curve, cov_node, pos) / num_graphs + _zero(pos, offsets)


def _subsegment_pairs(pts, w, graph, device, margin):
    """Pairs of sub-segments (e1, s1, e2, s2) of two curves of the same graph without a common node whose
    boxes (grown by the reach of their coverage) touch. Selection only: no gradient."""
    u, v, i, j = _candidate_pairs(graph, device)
    with torch.no_grad():
        rho = w / 2 + margin
        lo = pts.min(dim=1).values - rho[:, None]
        hi = pts.max(dim=1).values + rho[:, None]
        # <=: a straight horizontal or vertical curve has a flat box, which may touch the other one at its edge
        touch = ((lo[i] <= hi[j]) & (lo[j] <= hi[i])).all(dim=1)
        i, j = i[touch], j[touch]
        k = pts.shape[1] - 1
        s = torch.arange(k, device=device)
        s1 = s.repeat_interleave(k).repeat(len(i))
        s2 = s.repeat(k).repeat(len(i))
        e1, e2 = i.repeat_interleave(k * k), j.repeat_interleave(k * k)
        a1, b1, a2, b2 = pts[e1, s1], pts[e1, s1 + 1], pts[e2, s2], pts[e2, s2 + 1]
        lo1 = torch.minimum(a1, b1) - rho[e1, None]
        hi1 = torch.maximum(a1, b1) + rho[e1, None]
        lo2 = torch.minimum(a2, b2) - rho[e2, None]
        hi2 = torch.maximum(a2, b2) + rho[e2, None]
        keep = ((lo1 <= hi2) & (lo2 <= hi1)).all(dim=1)
    return e1[keep], s1[keep], e2[keep], s2[keep]


def curve_curve_overlaps(layout, offsets, graph, softness=0.5, k=K):
    pos = layout.positions
    margin = coverage_margin(softness)
    node_graph_index, num_graphs = node_graph(graph, pos.device)
    u, v = graph.edge_index.to(pos.device)
    w = edge_width(graph, node_graph_index[u], pos)
    pts = sample(pos, (u, v), offsets, k)
    e1, s1, e2, s2 = _subsegment_pairs(pts, w, graph, pos.device, margin)
    if len(e1) == 0:
        return _zero(pos, offsets)
    with torch.no_grad():
        a1, b1, a2, b2 = pts[e1, s1], pts[e1, s1 + 1], pts[e2, s2], pts[e2, s2 + 1]
        w1, w2 = w[e1], w[e2]
        lo = torch.maximum(torch.minimum(a1, b1) - (w1 / 2 + margin)[:, None], torch.minimum(a2, b2) - (w2 / 2 + margin)[:, None])
        hi = torch.minimum(torch.maximum(a1, b1) + (w1 / 2 + margin)[:, None], torch.maximum(a2, b2) + (w2 / 2 + margin)[:, None])
        x0, y0, x1, y1, keep = _boxes(lo, hi, layout.width, layout.height)
    e1, s1, e2, s2, x0, y0, x1, y1 = (t[keep] for t in (e1, s1, e2, s2, x0, y0, x1, y1))

    def cov(edge, segment):
        def coverage(X, Y, idx):
            p = pts[edge[idx]][:, None, None]  # (k, 1, 1, K + 1, 2), with gradient
            dist = segment_distance(X[..., None], Y[..., None], (p[..., :-1, 0], p[..., :-1, 1]),
                                    (p[..., 1:, 0], p[..., 1:, 1]))  # (k, Sy, Sx, K)
            nearest, owner = dist.min(dim=-1)
            mine = (owner == _col(segment[idx])).to(dist.dtype)  # the pixel belongs to this segment's pair
            return torch.sigmoid(-(nearest - _col(w[edge[idx]]) / 2) / softness) * mine
        return coverage

    return batched_overlap(x0, y0, x1, y1, cov(e1, s1), cov(e2, s2), pos) / num_graphs + _zero(pos, offsets)


def _crossing_curve_pairs(pts, graph, device):
    """(E', 2) unique pairs of curves (no common node) with at least one crossing pair of sub-segments."""
    zero_width = torch.zeros(pts.shape[0], dtype=pts.dtype, device=device)
    e1, s1, e2, s2 = _subsegment_pairs(pts, zero_width, graph, device, 0.0)
    with torch.no_grad():
        a, b, c, d = pts[e1, s1], pts[e1, s1 + 1], pts[e2, s2], pts[e2, s2 + 1]
        crossing = _crosses(a, b, c, d)
    return torch.unique(torch.stack((e1[crossing], e2[crossing]), dim=1), dim=0)


def _point_polyline_distance(p, polylines):
    """(P,) distance from each point p (P, 2) to the polyline polylines (P, K + 1, 2)."""
    a = (polylines[:, :-1, 0], polylines[:, :-1, 1])
    b = (polylines[:, 1:, 0], polylines[:, 1:, 1])
    return segment_distance(p[:, 0, None], p[:, 1, None], a, b).amin(dim=1)


def curve_crossings(layout, offsets, graph, k=K):
    pos = layout.positions
    _, num_graphs = node_graph(graph, pos.device)
    u, v = graph.edge_index.to(pos.device)
    pts = sample(pos, (u, v), offsets, k)
    pairs = _crossing_curve_pairs(pts.detach(), graph, pos.device)
    if len(pairs) == 0:
        return _zero(pos, offsets)
    e, f = pairs[:, 0], pairs[:, 1]
    moves = torch.stack((_point_polyline_distance(pts[f, 0], pts[e]), _point_polyline_distance(pts[f, -1], pts[e]),
                         _point_polyline_distance(pts[e, 0], pts[f]), _point_polyline_distance(pts[e, -1], pts[f])))
    return moves.min(dim=0).values.sum() / num_graphs + _zero(pos, offsets)


def count_curve_crossings(layout, offsets, graph, k=K):
    """Number of crossing pairs of CURVES (evaluation): two curves cross if any of their sub-segments do."""
    pos = layout.positions.detach()
    u, v = graph.edge_index.to(pos.device)
    pts = sample(pos, (u, v), offsets.detach(), k)
    _, num_graphs = node_graph(graph, pos.device)
    return len(_crossing_curve_pairs(pts, graph, pos.device)) / num_graphs


def curve_bend(layout, offsets, graph):
    pos = layout.positions
    _, num_graphs = node_graph(graph, pos.device)
    return bend(pos, graph.edge_index.to(pos.device), offsets).sum() / num_graphs + _zero(pos, offsets)


CURVE_TERMS = {"curve_node_overlaps": curve_node_overlaps, "curve_curve_overlaps": curve_curve_overlaps,
               "curve_crossings": curve_crossings, "curve_bend": curve_bend}
