"""Versions vectorisées de node_overlaps et edge_overlaps.

Testées : mêmes valeurs qu'une somme pixel par pixel sur toute l'image (à 0,1 % près pour les nœuds, la
différence venant des queues de sigmoïde coupées à 1 %), voir les tests dans le message de commit.

Principe : au lieu d'une boucle Python avec une grille par paire (4 .item() = 4 synchronisations
GPU->CPU par paire, et des milliers de petits noeuds dans le graphe autograd), on calcule la boîte de
TOUTES les paires d'un coup, puis on évalue les couvertures sur un tenseur (k, Sy, Sx) par paquet de paires.

Hypothèse à vérifier dans data/raster/sdf.py : node_coverage_torch et edge_coverage_torch ne font
que des opérations élément par élément, et n'utilisent des centres/extrémités que via c[0], c[1],
a[0], a[1], b[0], b[1]. Ici, on leur passe des tuples (cx, cy) de tenseurs de forme (k, 1, 1).
Si elles font autre chose (stack, indexation de pos...), il faut adapter les deux lambdas `cov`.

graph peut être un batch PyG : seules les paires d'un même graphe comptent, et le résultat est la moyenne
sur les graphes de la somme par graphe (voir metrics/batching.py).

Architecture : batched_overlap, _boxes et _clip_to_strip peuvent aller dans metrics_helpers.py.
"""
import torch

from data.raster.sdf import node_coverage_torch, edge_coverage_torch, coverage_margin
from metrics.batching import node_graph, edge_width


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
    total = like.sum() * 0.0  # zéro relié aux positions : backward() marche même sans aucune paire
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
    node_graph_index, num_graphs = node_graph(graph, pos.device)

    with torch.no_grad():  # sélection des paires et des boîtes : sans gradient, comme avant
        same_graph = node_graph_index[:, None] == node_graph_index[None, :]
        # chaque couverture déborde de `margin` au-delà du rayon : le produit peut être non nul jusqu'à
        # r_i + r_j + 2 margin (l'ancien filtre, à 1 margin, ratait ces queues)
        close = torch.triu((torch.cdist(pos, pos) < r[:, None] + r[None, :] + 2 * margin) & same_graph, diagonal=1)
        i, j = close.nonzero(as_tuple=True)
        lo = torch.maximum(pos[i] - r[i, None], pos[j] - r[j, None]) - margin
        hi = torch.minimum(pos[i] + r[i, None], pos[j] + r[j, None]) + margin
        x0, y0, x1, y1, keep = _boxes(lo, hi, W, H)
    i, j, x0, y0, x1, y1 = (t[keep] for t in (i, j, x0, y0, x1, y1))

    pi, pj, ri, rj = pos[i], pos[j], r[i], r[j]  # avec gradient

    def cov(p, rad):
        return lambda X, Y, idx: node_coverage_torch(
            X, Y, (_col(p[idx, 0]), _col(p[idx, 1])), _col(rad[idx]), softness)

    return batched_overlap(x0, y0, x1, y1, cov(pi, ri), cov(pj, rj), pos) / num_graphs


# --------------------------------------------------------------------------------------------
# Arêtes
# --------------------------------------------------------------------------------------------
def _clip_to_strip(p0, p1, a, b, half_width):
    """Partie du segment [p0, p1] à distance <= half_width de la droite (a, b).
    Retourne les extrémités (q0, q1) de cette partie et un masque `nonempty`. Tous de forme (K, 2) / (K,)."""
    ab = b - a
    normal = torch.stack((-ab[:, 1], ab[:, 0]), dim=1) / ab.norm(dim=1, keepdim=True).clamp_min(1e-12)
    s0 = ((p0 - a) * normal).sum(1)  # distance signée de p0 à la droite
    ds = ((p1 - p0) * normal).sum(1)  # variation de cette distance le long du segment
    parallel = ds.abs() < 1e-9
    safe = torch.where(parallel, torch.ones_like(ds), ds)
    t1, t2 = (-half_width - s0) / safe, (half_width - s0) / safe
    lo = torch.where(parallel, torch.zeros_like(ds), torch.minimum(t1, t2)).clamp(min=0)
    hi = torch.where(parallel, torch.ones_like(ds), torch.maximum(t1, t2)).clamp(max=1)
    nonempty = (lo <= hi) & (~parallel | (s0.abs() <= half_width))
    seg = p1 - p0
    return p0 + lo[:, None] * seg, p0 + hi[:, None] * seg, nonempty


def edge_overlaps(layout, graph, softness=0.5):
    pos = layout.positions  # (N, 2)
    u, v = graph.edge_index.to(pos.device)  # (E,), (E,)
    H, W = layout.height, layout.width
    margin = coverage_margin(softness)
    node_graph_index, num_graphs = node_graph(graph, pos.device)
    edge_graph = node_graph_index[u]
    w = edge_width(graph, edge_graph, pos)  # (E,) : un float par graphe, répété sur ses arêtes

    pu, pv = pos[u], pos[v]  # avec gradient (extrémités de chaque arête)

    with torch.no_grad():
        box_min = torch.minimum(pu, pv) - w[:, None] / 2
        box_max = torch.maximum(pu, pv) + w[:, None] / 2

        # filtre grossier : boîtes qui se touchent, sans sommet commun. Chaque couverture déborde de
        # `margin`, donc deux boîtes peuvent être séparées de 2 margin et se recouvrir quand même.
        touch = ((box_min[:, None] - 2 * margin < box_max[None, :])
                 & (box_min[None, :] - 2 * margin < box_max[:, None])).all(dim=-1)
        share = ((u[:, None] == u[None, :]) | (u[:, None] == v[None, :])
                 | (v[:, None] == u[None, :]) | (v[:, None] == v[None, :]))
        same_graph = edge_graph[:, None] == edge_graph[None, :]
        i, j = torch.triu(touch & ~share & same_graph, diagonal=1).nonzero(as_tuple=True)

        # Boîte serrée (optimisation) : la couverture d'une arête est nulle au-delà de rho = w/2 + margin
        # du segment. Un pixel couvert par les deux arêtes est donc à moins de rho_i d'un point de i qui est
        # lui-même à moins de rho_i + rho_j de la droite de j : on coupe i à la bande de cette demi-largeur
        # autour de j (et inversement), et la boîte est l'intersection des deux morceaux agrandis de rho. Pour deux
        # arêtes qui se croisent, elle se réduit au voisinage du croisement (quelques dizaines de pixels),
        # au lieu de l'intersection des boîtes englobantes (souvent des dizaines de milliers de pixels).
        rho_i, rho_j = w[i] / 2 + margin, w[j] / 2 + margin  # portée de chaque couverture
        ai, bi, ok_i = _clip_to_strip(pu[i], pv[i], pu[j], pv[j], rho_i + rho_j)
        aj, bj, ok_j = _clip_to_strip(pu[j], pv[j], pu[i], pv[i], rho_i + rho_j)
        lo = torch.maximum(torch.minimum(ai, bi) - rho_i[:, None], torch.minimum(aj, bj) - rho_j[:, None])
        hi = torch.minimum(torch.maximum(ai, bi) + rho_i[:, None], torch.maximum(aj, bj) + rho_j[:, None])
        x0, y0, x1, y1, keep = _boxes(lo, hi, W, H)
        keep = keep & ok_i & ok_j
    i, j, x0, y0, x1, y1 = (t[keep] for t in (i, j, x0, y0, x1, y1))

    def cov(a, b, width):
        return lambda X, Y, idx: edge_coverage_torch(
            X, Y, (_col(a[idx, 0]), _col(a[idx, 1])), (_col(b[idx, 0]), _col(b[idx, 1])), _col(width[idx]), softness)

    return batched_overlap(x0, y0, x1, y1, cov(pu[i], pv[i], w[i]), cov(pu[j], pv[j], w[j]), pos) / num_graphs

# --------------------------------------------------------------------------------------------
# Arête sur un nœud
# --------------------------------------------------------------------------------------------
def edge_node_overlaps(layout, graph, softness=0.5):
    """Surface (en pixels) où une arête recouvre un nœud qui n'est pas une de ses extrémités : une arête
    qui passe sur un nœud fait croire qu'elle y est reliée. Même principe que node_overlaps et
    edge_overlaps (somme des produits des couvertures SDF, pixel par pixel)."""
    pos = layout.positions  # (N, 2)
    r = graph.node_size.to(device=pos.device, dtype=pos.dtype)  # (N,)
    u, v = graph.edge_index.to(pos.device)
    H, W = layout.height, layout.width
    margin = coverage_margin(softness)
    node_graph_index, num_graphs = node_graph(graph, pos.device)
    edge_graph = node_graph_index[u]
    w = edge_width(graph, edge_graph, pos)

    with torch.no_grad():
        # paires (arête e, nœud k) du même graphe, k n'étant pas une extrémité de e
        k_all = torch.arange(pos.shape[0], device=pos.device)
        candidates = (edge_graph[:, None] == node_graph_index[None, :]) \
            & (u[:, None] != k_all[None, :]) & (v[:, None] != k_all[None, :])
        e, k = candidates.nonzero(as_tuple=True)
        # filtre exact : couvertures nulles au-delà de r + margin du centre et de w/2 + margin du segment
        reach = r[k] + w[e] / 2 + 2 * margin
        keep = _point_segment_distance(pos[k], pos[u[e]], pos[v[e]]) < reach
        e, k = e[keep], k[keep]
        node_lo, node_hi = pos[k] - (r[k] + margin)[:, None], pos[k] + (r[k] + margin)[:, None]
        edge_lo = torch.minimum(pos[u[e]], pos[v[e]]) - (w[e] / 2 + margin)[:, None]
        edge_hi = torch.maximum(pos[u[e]], pos[v[e]]) + (w[e] / 2 + margin)[:, None]
        x0, y0, x1, y1, keep = _boxes(torch.maximum(node_lo, edge_lo), torch.minimum(node_hi, edge_hi), W, H)
    e, k, x0, y0, x1, y1 = (t[keep] for t in (e, k, x0, y0, x1, y1))

    pa, pb, pk = pos[u[e]], pos[v[e]], pos[k]  # avec gradient
    we, rk = w[e], r[k]

    def cov_edge(X, Y, idx):
        return edge_coverage_torch(X, Y, (_col(pa[idx, 0]), _col(pa[idx, 1])), (_col(pb[idx, 0]), _col(pb[idx, 1])),
                                   _col(we[idx]), softness)

    def cov_node(X, Y, idx):
        return node_coverage_torch(X, Y, (_col(pk[idx, 0]), _col(pk[idx, 1])), _col(rk[idx]), softness)

    return batched_overlap(x0, y0, x1, y1, cov_edge, cov_node, pos) / num_graphs


def _point_segment_distance(p, a, b):
    """Distance de chaque point p au segment [a, b] correspondant. Tous de forme (K, 2)."""
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (p - (a + t[:, None] * ab)).norm(dim=-1)
