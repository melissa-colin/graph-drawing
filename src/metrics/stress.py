import torch

from data.input import GraphInput
from data.output import Layout
from metrics.batching import cached, node_graph, per_graph


def stress(layout: Layout, graph: GraphInput, gamma: float = 2.0) -> torch.Tensor:
    """Kamada-Kawai stress: sum over pairs of (||p_u - p_v|| - L d_G(u,v))² / (L d_G(u,v))^gamma.

    The pairs come from graph.dist_index (2, P) and graph.dist_value (P,): each pair i < j once,
    only those with 0 < d_G < inf (see GraphInput). L converts graph distances to pixels (see below).
    graph can be a PyG batch: the sum is per graph, and the result is the mean over the graphs.
    """
    pos = layout.positions

    def static():  # pairs, graph distances and their pixel version: fixed, cached while refining one graph
        i, j = graph.dist_index.to(pos.device)
        d = graph.dist_value.to(device=pos.device, dtype=pos.dtype)
        return i, j, d, _pixel_distances(layout, graph, d, i)
    i, j, d, d_px = cached(graph, ("stress", str(pos.device), layout.width, layout.height), static)
    e = (pos[i] - pos[j]).norm(dim=1)  # euclidean distance in the drawing
    _, num_graphs = node_graph(graph, pos.device)
    return ((e - d_px) ** 2 / d_px ** gamma).sum() / num_graphs


def _pixel_distances(layout, graph, d, i):
    """Graph distances converted to pixels, L d_G (see stress)."""
    pos = layout.positions
    # Échelle : d_G compte des arêtes, e est en pixels. Sans conversion, le dessin idéal met 1 pixel par
    # arête, et le stress écrase tout le graphe sur quelques pixels. L = nombre de pixels par arête, choisi
    # pour que les deux nœuds les plus éloignés (distance max d_G) aillent d'un bord à l'autre de l'image,
    # moins le rayon des nœuds de chaque côté : le dessin idéal remplit le cadre. Avec gamma = 2, le
    # résultat n'a pas d'unité : même échelle quelle que soit la taille de l'image.
    # Dans un batch, L est calculé pour chaque graphe (son propre diamètre et ses propres nœuds).
    node_graph_index, num_graphs = node_graph(graph, pos.device)
    pair_graph = node_graph_index[i]
    margin = per_graph(graph.node_size.to(device=pos.device, dtype=pos.dtype), node_graph_index, num_graphs, "amax")
    L = (min(layout.width, layout.height) - 2 * margin) / per_graph(d, pair_graph, num_graphs, "amax").clamp_min(1)
    return L[pair_graph] * d  # distances du graphe converties en pixels


def scaled_stress(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Stress at the best scale (scale-normalized stress, Ahmed et al. 2026; CoRe-GD trains on it):
    sum over pairs of (s ||p_u - p_v|| - d_G(u,v))² / d_G(u,v)², where s is the scale that minimizes it
    (closed form, per graph). Only the shape counts, not its size: it is exactly what our benchmark
    measures (up to the division by the number of pairs), unlike `stress`, which also imposes a size
    (L pixels per edge). graph can be a PyG batch: sum per graph, mean over the graphs."""
    pos = layout.positions
    node_graph_index, num_graphs = node_graph(graph, pos.device)

    def static():  # fixed, cached while refining one graph
        i, j = graph.dist_index.to(pos.device)
        d = graph.dist_value.to(device=pos.device, dtype=pos.dtype)
        return i, j, d, node_graph_index[i], d ** -2
    i, j, d, pair_graph, w = cached(graph, ("scaled_stress", str(pos.device)), static)
    e = (pos[i] - pos[j]).norm(dim=1)
    s = per_graph(w * e * d, pair_graph, num_graphs, "sum") / per_graph(w * e * e, pair_graph, num_graphs, "sum").clamp_min(1e-12)
    return (w * (s[pair_graph] * e - d) ** 2).sum() / num_graphs
