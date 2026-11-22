"""Geometric counterparts of the three screen-space terms of the loss: the CONTROL of the resolution sweep.

The refinement loss (train.REFINE_WEIGHTS) mixes geometric terms (stress, crossings, angles) with three terms
computed on pixel coverages through SDFs (metrics/overlaps_vectorised.py): overlapping nodes, overlapping
edges, edges over nodes. To know whether computing these three in screen space is what helps, the sweep runs
the same refinement with these geometric versions instead, which see nodes as disks and edges as thick
segments, but no pixel: same budget, same other terms, same weight selection.

    node_overlaps       sum over node pairs of relu(r_i + r_j - d_ij)^2             (as FORBID)
    edge_node_overlaps  sum over (edge, node not one of its ends) of relu(r_k + w/2 - dist(k, edge))^2
    edge_overlaps       sum over pairs of edges without a common node of
                          w_i w_j / max(|sin(angle)|, 0.05)   if they cross (area of the parallelogram
                                                              where the two thick edges overlap)
                          relu((w_i + w_j) / 2 - dist(edge_i, edge_j))^2   otherwise

All in px^2, like the screen versions. graph can be a PyG batch: sum per graph, mean over the graphs (as
every metric of metrics/). Static pairs are cached while refining one graph (metrics/batching.py).
"""
import torch

from metrics.batching import cached, edge_width, node_graph
from metrics.edge_crossings import _candidate_pairs, _side
from metrics.overlaps_vectorised import _point_segment_distance

MIN_SIN = 0.05  # the overlap area of two crossing edges grows as 1 / sin(angle): capped for almost parallel ones


def node_overlaps(layout, graph):
    pos = layout.positions
    r = graph.node_size.to(device=pos.device, dtype=pos.dtype)
    node_graph_index, num_graphs = node_graph(graph, pos.device)

    def static():
        same = torch.triu(node_graph_index[:, None] == node_graph_index[None, :], diagonal=1)
        return same.nonzero(as_tuple=True)
    i, j = cached(graph, ("geo_node_pairs", str(pos.device)), static)
    d = _distance(pos[i] - pos[j])
    return torch.relu(r[i] + r[j] - d).square().sum() / num_graphs + pos.sum() * 0.0


def edge_node_overlaps(layout, graph):
    pos = layout.positions
    r = graph.node_size.to(device=pos.device, dtype=pos.dtype)
    node_graph_index, num_graphs = node_graph(graph, pos.device)

    def static():
        u, v = graph.edge_index.to(pos.device)
        edge_graph = node_graph_index[u]
        k_all = torch.arange(pos.shape[0], device=pos.device)
        candidates = (edge_graph[:, None] == node_graph_index[None, :]) \
            & (u[:, None] != k_all[None, :]) & (v[:, None] != k_all[None, :])
        e, k = candidates.nonzero(as_tuple=True)
        return u, v, edge_width(graph, edge_graph, pos), e, k
    u, v, w, e, k = cached(graph, ("geo_edge_node_pairs", str(pos.device)), static)
    d = _point_segment_distance(pos[k], pos[u[e]], pos[v[e]])
    return torch.relu(r[k] + w[e] / 2 - d).square().sum() / num_graphs + pos.sum() * 0.0


def edge_overlaps(layout, graph):
    pos = layout.positions
    node_graph_index, num_graphs = node_graph(graph, pos.device)
    u, v, i, j = _candidate_pairs(graph, pos.device)  # pairs of edges of a same graph, no common node
    w = cached(graph, ("geo_edge_width", str(pos.device)), lambda: edge_width(graph, node_graph_index[u], pos))

    with torch.no_grad():  # which pairs cross, which are close: no gradient through the selection
        a, b, c, d = pos[u[i]], pos[v[i]], pos[u[j]], pos[v[j]]
        crossing = (_side(a, b, c) * _side(a, b, d) < 0) & (_side(c, d, a) * _side(c, d, b) < 0)
        reach = (w[i] + w[j]) / 2
        near = ((torch.minimum(a, b) - reach[:, None] < torch.maximum(c, d))
                & (torch.minimum(c, d) - reach[:, None] < torch.maximum(a, b))).all(dim=1) & ~crossing

    total = pos.sum() * 0.0  # + 0 linked to pos: backward always works
    ci, cj = i[crossing], j[crossing]
    if len(ci):
        e1, e2 = pos[v[ci]] - pos[u[ci]], pos[v[cj]] - pos[u[cj]]
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        sin = cross.abs() / (_distance(e1) * _distance(e2))
        total = total + (w[ci] * w[cj] / sin.clamp_min(MIN_SIN)).sum()
    ni, nj = i[near], j[near]
    if len(ni):
        a, b, c, d = pos[u[ni]], pos[v[ni]], pos[u[nj]], pos[v[nj]]
        dist = torch.stack((_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
                            _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))).min(dim=0).values
        total = total + torch.relu((w[ni] + w[nj]) / 2 - dist).square().sum()
    return total / num_graphs


def _distance(vectors):
    """Norm of each row, with a gradient defined at 0 (two nodes at the same place)."""
    return (vectors.square().sum(-1) + 1e-12).sqrt()


GEOMETRIC = {"node_overlaps": node_overlaps, "edge_overlaps": edge_overlaps, "edge_node_overlaps": edge_node_overlaps}
