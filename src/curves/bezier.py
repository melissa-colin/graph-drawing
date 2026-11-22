"""Curved edges: one quadratic Bezier curve per edge, drawn as a thick stroke with round joins.

Edge (u, v) goes from p_u to p_v with control point c = (p_u + p_v) / 2 + o, where o (2 values per edge) is
optimized; o = 0 gives the straight edge. The curve is sampled into a polyline of K segments, differentiable
in (p_u, p_v, o). The signed distance to the stroke is the minimum over the segments of the capsule distance
(distance to the segment - w / 2): the exact SDF of a polyline drawn with round joins and caps. The analytic
SDF of a Bezier curve (Quilez) needs a cubic solve whose gradient is unstable near the cusps; the polyline is
exact for what is drawn and converges to the curve as K grows (tested).
"""
import torch

K = 8  # segments per curve


def control_points(pos, edge_index, offsets):
    """(E, 2) control point of each edge."""
    u, v = edge_index
    return (pos[u] + pos[v]) / 2 + offsets


def sample(pos, edge_index, offsets, k=K):
    """(E, k + 1, 2) points of each curve at t = 0, 1/k, ..., 1 (first p_u, last p_v)."""
    u, v = edge_index
    a, b, c = pos[u], pos[v], control_points(pos, edge_index, offsets)
    t = torch.linspace(0, 1, k + 1, dtype=pos.dtype, device=pos.device)[None, :, None]
    return (1 - t) ** 2 * a[:, None] + 2 * (1 - t) * t * c[:, None] + t ** 2 * b[:, None]


def segment_distance(px, py, a, b):
    """Distance from the points (px, py) to the segments [a, b] (broadcast; a, b given as (x, y) pairs).
    The +1e-12 keeps a gradient where the distance is 0."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = (((px - a[0]) * abx + (py - a[1]) * aby) / (abx * abx + aby * aby).clamp_min(1e-12)).clamp(0, 1)
    dx, dy = px - (a[0] + t * abx), py - (a[1] + t * aby)
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def capsule_sdf(px, py, a, b, w):
    """Signed distance to a segment of width w with round caps. Negative inside."""
    return segment_distance(px, py, a, b) - w / 2


def curve_sdf(px, py, points, w):
    """Signed distance to the stroke of a polyline. points: (..., k + 1, 2) with the leading dims broadcastable
    against px[..., None]; returns the shape of px."""
    a = (points[..., :-1, 0], points[..., :-1, 1])
    b = (points[..., 1:, 0], points[..., 1:, 1])
    return capsule_sdf(px[..., None], py[..., None], a, b, w[..., None] if torch.is_tensor(w) else w).amin(dim=-1)


def curve_length(points):
    """(E,) length of each polyline."""
    return (points[:, 1:] - points[:, :-1]).norm(dim=-1).sum(dim=-1)


def bend(pos, edge_index, offsets):
    """(E,) how curved each edge is: |o|^2 / |p_u - p_v|^2 (0 straight; scale-free)."""
    u, v = edge_index
    return offsets.square().sum(-1) / ((pos[u] - pos[v]).square().sum(-1) + 1e-6)
