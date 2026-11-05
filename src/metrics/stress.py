import torch

from data.input import GraphInput
from data.output import Layout


def stress(layout: Layout, graph: GraphInput, gamma: float = 2.0) -> torch.Tensor:
    """Kamada-Kawai stress: sum over pairs of (||p_u - p_v|| - d_G(u,v))² / d_G(u,v)^gamma.

    The pairs come from graph.dist_index (2, P) and graph.dist_value (P,): each pair i < j once,
    only those with 0 < d_G < inf (see GraphInput).
    """
    pos = layout.positions
    i, j = graph.dist_index.to(pos.device)
    d = graph.dist_value.to(device=pos.device, dtype=pos.dtype)
    e = (pos[i] - pos[j]).norm(dim=1)  # euclidean distance in the drawing
    return ((e - d) ** 2 / d ** gamma).sum()
