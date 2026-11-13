"""Versions vectorisées de node_overlaps et edge_overlaps (NON TESTÉES : à comparer à l'ancienne boucle).

Principe : au lieu d'une boucle Python avec une grille par paire (4 .item() = 4 synchronisations
GPU->CPU par paire, et des milliers de petits noeuds dans le graphe autograd), on calcule la boîte de
TOUTES les paires d'un coup, puis on évalue les couvertures sur un tenseur (k, Sy, Sx) par paquet de paires.

Hypothèse à vérifier dans data/raster/sdf.py : node_coverage_torch et edge_coverage_torch ne font
que des opérations élément par élément, et n'utilisent des centres/extrémités que via c[0], c[1],
a[0], a[1], b[0], b[1]. Ici, on leur passe des tuples (cx, cy) de tenseurs de forme (k, 1, 1).
Si elles font autre chose (stack, indexation de pos...), il faut adapter les deux lambdas `cov`.

Architecture : batched_overlap, _boxes et _segment_distance peuvent aller dans metrics_helpers.py.
"""
import torch

from data.raster.sdf import node_coverage_torch, edge_coverage_torch, coverage_margin


def _col(t):
    return t[:, None, None]


def _boxes(lo, hi, W, H):
    """Boîte en pixels de chaque paire, tronquée à l'image : x0 <= X < x1, y0 <= Y < y1.
    Même convention que pixel_grid. Valeurs entières stockées en float."""
    x0 = lo[:, 0].floor().clamp(min=0)
    y0 = lo[:, 1].floor().clamp(min=0)
    x1 = (hi[:, 0].ceil() + 1).clamp(max=W)
    y1 = (hi[:, 1].ceil() + 1).clamp(max=H)
    keep = (x0 < x1) & (y0 < y1)
    return x0, y0, x1, y1, keep


def batched_overlap(x0, y0, x1, y1, coverage_i, coverage_j, like, max_cells=2_000_000):
    """Somme sur toutes les paires et tous les pixels de cov_i * cov_j.

    x0, y0, x1, y1 : (K,) boîte de chaque paire. coverage_i/j(X, Y, idx) : couvertures des paires idx,
    avec X, Y de forme (k, Sy, Sx). Les paires sont triées par taille de boîte et groupées en paquets
    dont le tenseur ne dépasse pas max_cells cases, pour limiter le remplissage inutile et la mémoire.
    """
    K = x0.numel()
    total = like.new_zeros(())
    if K == 0:
        return total

    sx, sy = x1 - x0, y1 - y0
    order = torch.argsort(sx * sy)
    sxs, sys_ = sx[order].long().tolist(), sy[order].long().tolist()  # une seule synchronisation

    start = 0
    while start < K:
        end, mx, my = start + 1, sxs[start], sys_[start]
        while end < K:
            nx, ny = max(mx, sxs[end]), max(my, sys_[end])
            if (end - start + 1) * nx * ny > max_cells:
                break
            mx, my, end = nx, ny, end + 1
        idx = order[start:end]

        xs = torch.arange(mx, dtype=like.dtype, device=like.device)
        ys = torch.arange(my, dtype=like.dtype, device=like.device)
        X = (_col(x0[idx]) + xs[None, None, :]).expand(-1, my, -1)
        Y = (_col(y0[idx]) + ys[None, :, None]).expand(-1, -1, mx)
        inside = (X < _col(x1[idx])) & (Y < _col(y1[idx]))  # masque le remplissage hors de la vraie boîte

        total = total + (coverage_i(X, Y, idx) * coverage_j(X, Y, idx) * inside).sum()
        start = end
    return total


# --------------------------------------------------------------------------------------------
# Nœuds
# --------------------------------------------------------------------------------------------
def node_overlaps(layout, graph, softness=0.5):
    pos = layout.positions  # (N, 2)
    r = graph.node_size.to(device=pos.device, dtype=pos.dtype)  # (N,)
    H, W = layout.height, layout.width
    margin = coverage_margin(softness)

    with torch.no_grad():  # sélection des paires et des boîtes : sans gradient, comme avant
        close = torch.triu(torch.cdist(pos, pos) < r[:, None] + r[None, :] + margin, diagonal=1)
        i, j = close.nonzero(as_tuple=True)
        lo = torch.maximum(pos[i] - r[i, None], pos[j] - r[j, None]) - margin
        hi = torch.minimum(pos[i] + r[i, None], pos[j] + r[j, None]) + margin
        x0, y0, x1, y1, keep = _boxes(lo, hi, W, H)
    i, j, x0, y0, x1, y1 = (t[keep] for t in (i, j, x0, y0, x1, y1))

    pi, pj, ri, rj = pos[i], pos[j], r[i], r[j]  # avec gradient

    def cov(p, rad):
        return lambda X, Y, idx: node_coverage_torch(
            X, Y, (_col(p[idx, 0]), _col(p[idx, 1])), _col(rad[idx]), softness)

    return batched_overlap(x0, y0, x1, y1, cov(pi, ri), cov(pj, rj), pos)


# --------------------------------------------------------------------------------------------
# Arêtes
# --------------------------------------------------------------------------------------------
def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_segment_distance(p, a, b):
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (p - (a + t[:, None] * ab)).norm(dim=-1)


def _segment_distance(a, b, c, d):
    """Distance entre les segments [a,b] et [c,d] : 0 s'ils se croisent, sinon le plus petit des
    4 écarts extrémité-segment."""
    d1, d2 = _cross2(b - a, c - a), _cross2(b - a, d - a)
    d3, d4 = _cross2(d - c, a - c), _cross2(d - c, b - c)
    cross = (d1 * d2 < 0) & (d3 * d4 < 0)
    dist = torch.minimum(
        torch.minimum(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d)),
        torch.minimum(_point_segment_distance(c, a, b), _point_segment_distance(d, a, b)),
    )
    return torch.where(cross, torch.zeros_like(dist), dist)


def edge_overlaps(layout, graph, softness=0.5):
    pos = layout.positions  # (N, 2)
    u, v = graph.edge_index.to(pos.device)  # (E,), (E,)
    w = float(graph.edge_width)
    H, W = layout.height, layout.width
    margin = coverage_margin(softness)

    pu, pv = pos[u], pos[v]  # avec gradient (extrémités de chaque arête)

    with torch.no_grad():
        box_min = torch.minimum(pu, pv) - w / 2
        box_max = torch.maximum(pu, pv) + w / 2

        # filtre grossier : boîtes qui se touchent, sans sommet commun (comme avant)
        touch = ((box_min[:, None] - margin < box_max[None, :])
                 & (box_min[None, :] - margin < box_max[:, None])).all(dim=-1)
        share = ((u[:, None] == u[None, :]) | (u[:, None] == v[None, :])
                 | (v[:, None] == u[None, :]) | (v[:, None] == v[None, :]))
        i, j = torch.triu(touch & ~share, diagonal=1).nonzero(as_tuple=True)

        # filtre exact : la couverture est nulle au-delà de w/2 + margin de chaque segment,
        # donc le produit est nul si les segments sont à plus de w + 2*margin l'un de l'autre.
        keep = _segment_distance(pu[i], pv[i], pu[j], pv[j]) < w + 2 * margin
        i, j = i[keep], j[keep]

        lo = torch.maximum(box_min[i], box_min[j]) - margin
        hi = torch.minimum(box_max[i], box_max[j]) + margin
        x0, y0, x1, y1, keep = _boxes(lo, hi, W, H)
    i, j, x0, y0, x1, y1 = (t[keep] for t in (i, j, x0, y0, x1, y1))

    def cov(a, b):
        return lambda X, Y, idx: edge_coverage_torch(
            X, Y, (_col(a[idx, 0]), _col(a[idx, 1])), (_col(b[idx, 0]), _col(b[idx, 1])), w, softness)

    return batched_overlap(x0, y0, x1, y1, cov(pu[i], pv[i]), cov(pu[j], pv[j]), pos)