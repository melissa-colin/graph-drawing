import torch

from data.input import GraphInput
from data.output import Layout


def stress(layout: Layout, graph: GraphInput, gamma: float = 2.0) -> torch.Tensor:
    """Kamada-Kawai-inspired stress loss comparing geometric distances to ideal distances L * d_G(u, v).
    """
    pos = layout.positions
    i, j = graph.dist_index.to(pos.device)
    d = graph.dist_value.to(device=pos.device, dtype=pos.dtype)

    L = min(layout.width, layout.height) / d.max()
    e = (pos[i] - pos[j]).norm(dim=1)

    return (((e - L * d) ** 2) / (L * d) ** gamma).sum()