
import torch

from data.input import GraphInput
from data.output import Layout


def stress(layout, graph, gamma=2.0):
    """Kamada-Kawai stress: sum over pairs of (||p_u - p_v|| - d_G(u,v))² / d_G(u,v)^gamma."""
    pos = layout.positions
    d_G = graph.dist.to(device=pos.device, dtype=pos.dtype) # good format
    delta = torch.cdist(pos, pos) # euclidian distance

    pairs = torch.triu(torch.isfinite(d_G), diagonal=1)  # each pair once (keep all element above diagonal), no inf
    d, e = d_G[pairs], delta[pairs]
    return ((e - d) ** 2 / d ** gamma).sum()