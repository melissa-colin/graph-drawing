
import torch

from data.input import GraphInput
from data.output import Layout


def stress(layout: Layout, graph: GraphInput, gamma: float = 2.0,) -> torch.Tensor:
    """Kamada-Kawai stress between node centers."""

    positions = layout.positions

    if positions.shape[0] < 2:
        return positions.new_zeros(())

    # Graph-theoretic distances between nodes in the original graph
    d_G = graph.dist.to(device=positions.device, dtype=positions.dtype)

    # Euclidean distances in pixels between every pair of node centers in the layout
    delta_uv = torch.cdist(positions, positions)

    # Create a mask that selects each unordered node pair (u, v) only once,
    # excluding pairs where u == v (the diagonal)
    pairs = torch.triu(torch.ones_like(d_G, dtype=torch.bool), diagonal=1,)

    # Exclude unconnected pairs and zero distances
    pairs = pairs & torch.isfinite(d_G) & (d_G > 0)

    # E_KK = somme de (delta_uv - d_G(u,v))² / d_G(u,v)²
    differences_squared = (delta_uv - d_G) ** 2
    weighted_stress = differences_squared / (d_G ** gamma)

    return weighted_stress[pairs].sum()
