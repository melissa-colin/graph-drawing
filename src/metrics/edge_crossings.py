import torch

from data.input import GraphInput
from data.output import Layout
from metrics.batching import node_graph


def _side(a, b, p):
    """Signed distance (pixels) from p to the line (a, b): > 0 on one side, < 0 on the other."""
    ab = b - a
    cross = ab[:, 0] * (p[:, 1] - a[:, 1]) - ab[:, 1] * (p[:, 0] - a[:, 0])
    return cross / ab.norm(dim=1).clamp_min(1e-9)


def edge_crossings(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Edge crossings, as a loss: sum over the crossing pairs of the smallest move that uncrosses them.

    Two segments [a, b] and [c, d] cross when c and d are on both sides of the line (a, b) AND a and b
    on both sides of the line (c, d). Moving one endpoint across the other line is enough to uncross them
    (both endpoints then lie on the same side). The cost of a crossing pair is the smallest of these moves:
        min(|d(c)|, |d(d)|, |d(a)|, |d(b)|)   (perpendicular distances, in pixels)
    0 when no edges cross, and its gradient pushes the closest endpoint to the other side: unlike
    edge_overlaps (overlap area of the thick edges), it does uncross the edges.
    Pairs sharing a node are ignored. graph can be a PyG batch: sum per graph, mean over the graphs.
    """
    pos = layout.positions
    u, v = graph.edge_index.to(pos.device)
    node_graph_index, num_graphs = node_graph(graph, pos.device)

    with torch.no_grad():  # which pairs cross: geometry only, no gradient
        edge_graph = node_graph_index[u]
        candidates = (edge_graph[:, None] == edge_graph[None, :]) & ~(
            (u[:, None] == u[None, :]) | (u[:, None] == v[None, :])
            | (v[:, None] == u[None, :]) | (v[:, None] == v[None, :]))
        i, j = torch.triu(candidates, diagonal=1).nonzero(as_tuple=True)
        a, b, c, d = pos[u[i]], pos[v[i]], pos[u[j]], pos[v[j]]
        crossing = (_side(a, b, c) * _side(a, b, d) < 0) & (_side(c, d, a) * _side(c, d, b) < 0)
        i, j = i[crossing], j[crossing]

    a, b, c, d = pos[u[i]], pos[v[i]], pos[u[j]], pos[v[j]]  # with gradient
    moves = torch.stack((_side(a, b, c).abs(), _side(a, b, d).abs(), _side(c, d, a).abs(), _side(c, d, b).abs()))
    return moves.min(dim=0).values.sum() / num_graphs + pos.sum() * 0.0  # + 0 linked to pos: backward always works


def count_crossings(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Number of crossing pairs of edges (not differentiable: for evaluation). Batch: mean over the graphs."""
    pos = layout.positions.detach()
    u, v = graph.edge_index.to(pos.device)
    node_graph_index, num_graphs = node_graph(graph, pos.device)
    edge_graph = node_graph_index[u]
    candidates = (edge_graph[:, None] == edge_graph[None, :]) & ~(
        (u[:, None] == u[None, :]) | (u[:, None] == v[None, :])
        | (v[:, None] == u[None, :]) | (v[:, None] == v[None, :]))
    i, j = torch.triu(candidates, diagonal=1).nonzero(as_tuple=True)
    a, b, c, d = pos[u[i]], pos[v[i]], pos[u[j]], pos[v[j]]
    crossing = (_side(a, b, c) * _side(a, b, d) < 0) & (_side(c, d, a) * _side(c, d, b) < 0)
    return crossing.sum().to(pos.dtype) / num_graphs
