import torch

from data.input import GraphInput
from data.output import Layout
from metrics.batching import node_graph, per_graph


def stress(layout: Layout, graph: GraphInput, gamma: float = 2.0) -> torch.Tensor:
    """Kamada-Kawai stress: sum over pairs of (||p_u - p_v|| - L d_G(u,v))² / (L d_G(u,v))^gamma.

    The pairs come from graph.dist_index (2, P) and graph.dist_value (P,): each pair i < j once,
    only those with 0 < d_G < inf (see GraphInput). L converts graph distances to pixels (see below).
    graph can be a PyG batch: the sum is per graph, and the result is the mean over the graphs.
    """
    pos = layout.positions
    i, j = graph.dist_index.to(pos.device)
    d = graph.dist_value.to(device=pos.device, dtype=pos.dtype)
    e = (pos[i] - pos[j]).norm(dim=1)  # euclidean distance in the drawing
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
    d_px = L[pair_graph] * d  # distances du graphe converties en pixels
    return ((e - d_px) ** 2 / d_px ** gamma).sum() / num_graphs
