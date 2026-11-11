import torch

from data.input import GraphInput
from data.output import Layout


def screen_inside(layout: Layout, graph: GraphInput) -> torch.Tensor:
    x = layout.positions[:, 0]
    y = layout.positions[:, 1]

    out_x = torch.relu(-x) + torch.relu(x - layout.width)
    out_y = torch.relu(-y) + torch.relu(y - layout.height)

    return (out_x ** 2 + out_y ** 2).sum()