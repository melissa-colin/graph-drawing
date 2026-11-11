import torch

from data.input import GraphInput
from data.output import Layout


def screen_fill(layout: Layout, graph: GraphInput) -> torch.Tensor:
    positions = layout.positions

    x = positions[:, 0]
    y = positions[:, 1]
    
    return (2.0 - (x.max() - x.min()) / layout.width - (y.max() - y.min()) / layout.height)