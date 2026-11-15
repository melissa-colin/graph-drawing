import math
import torch

from data.input import GraphInput
from data.output import Layout
from metrics.batching import cached, node_graph, per_graph


def edge_angles(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Angular resolution, as a loss: mean over the nodes of degree >= 2 of 1 - theta_min(v) / theta_ideal(v).

    theta_min(v): smallest angle between two consecutive edges around v.
    theta_ideal(v) = 2 pi / deg(v): the edges evenly spread around v.
    In [0, 1]: 0 when every node has evenly spread edges. 1 - this value is the minimum angle metric
    of Purchase (2002), as used in the graph drawing literature.
    graph can be a PyG batch: mean over the nodes of each graph, then mean over the graphs.

    Correction : l'ancienne version prenait le plus petit angle de TOUT le graphe, comparé à 2 pi / degré max.
    Un seul couple d'arêtes recevait un gradient, et un nœud de degré 2 était jugé avec l'angle idéal du
    nœud de plus haut degré. Ici, chaque nœud a son propre angle idéal et contribue à la moyenne.
    Calcul : on trie les arêtes de chaque nœud par direction, les angles entre arêtes voisines sont les
    écarts entre directions consécutives (O(E log E), au lieu de la matrice E x E de toutes les paires).
    """
    pos = layout.positions
    n = pos.shape[0]
    two_pi = 2 * math.pi
    node_graph_index, num_graphs = node_graph(graph, pos.device)

    def static():  # depends on the graph only: cached while refining one graph
        src, dst = graph.edge_index_bidirectional.to(pos.device)  # each edge seen from both ends
        degree = torch.bincount(src, minlength=n)
        has_angle = degree >= 2
        count = per_graph(has_angle.to(pos.dtype), node_graph_index, num_graphs, "sum")
        return src, dst, degree, has_angle, count
    src, dst, degree, has_angle, count = cached(graph, ("angles", str(pos.device)), static)

    # direction of each edge leaving its node; a zero-length edge has no direction: no gradient through it
    vec = pos[dst] - pos[src]
    valid = vec.detach().norm(dim=1) > 1e-6
    vec = torch.where(valid[:, None], vec, vec.new_tensor([1.0, 0.0]))
    angle = torch.atan2(vec[:, 1], vec[:, 0])  # in [-pi, pi]

    # sort by node, then by direction: two stable sorts (exact, unlike a combined float key)
    order = torch.argsort(angle.detach())
    order = order[torch.argsort(src[order], stable=True)]
    node, angle = src[order], angle[order]

    # next edge around the same node; the last one wraps to the first one (+ 2 pi)
    first = torch.cumsum(degree, 0) - degree  # index of the first edge of each node in the sorted list
    rank = torch.arange(node.numel(), device=pos.device)
    is_last = rank == first[node] + degree[node] - 1
    following = torch.where(is_last, first[node], rank + 1)
    gap = angle[following] - angle + is_last.to(pos.dtype) * two_pi

    theta_min = pos.new_full((n,), two_pi).scatter_reduce(0, node, gap, reduce="amin", include_self=False)
    theta_ideal = two_pi / degree.clamp(min=1).to(pos.dtype)
    per_node = torch.where(has_angle, 1.0 - theta_min / theta_ideal, torch.zeros_like(theta_min))

    # mean over the nodes of degree >= 2 of each graph (0 for a graph without any), then over the graphs
    total = per_graph(per_node, node_graph_index, num_graphs, "sum")
    return (total / count.clamp_min(1)).mean() + pos.sum() * 0.0  # + 0 linked to pos: backward always works
