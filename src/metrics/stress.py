import torch

from data.input import GraphInput
from data.output import Layout

def stress(layout: Layout, graph: GraphInput, gamma: float = 2.0) -> torch.Tensor:
    """Kamada-Kawai stress: sum over pairs of (||p_u - p_v|| - L d_G(u,v))² / (L d_G(u,v))^gamma.

    The pairs come from graph.dist_index (2, P) and graph.dist_value (P,): each pair i < j once,
    only those with 0 < d_G < inf (see GraphInput). L converts graph distances to pixels (see below).
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
    margin = graph.node_size.max().to(device=pos.device, dtype=pos.dtype)
    L = (min(layout.width, layout.height) - 2 * margin) / d.max()
    d_px = L * d  # distances du graphe converties en pixels
    return ((e - d_px) ** 2 / d_px ** gamma).sum()