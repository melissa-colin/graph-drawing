"""Training loss: weighted sum of metrics (see metrics/)."""

from typing import Callable

import torch
import torch.nn as nn

from data.input import GraphInput
from data.output import Layout
from metrics.screen import screen
from metrics.stress import stress

LossFn = Callable[
    [Layout, GraphInput],
    tuple[torch.Tensor, dict[str, float]],
]


class Loss(nn.Module):
    def __init__(self, weights: dict[str, float]):
        """Weights control the contribution of each metric."""
        super().__init__()
        self.weights = weights

    def forward(
        self, layout: Layout, graph: GraphInput
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return total loss and individual metric values."""

        values = {
            "screen": screen(layout, graph),
            "stress": stress(layout, graph),
        }

        total = layout.positions.new_zeros(())

        for name, value in values.items():
            weight = self.weights.get(name, 0.0)
            total = total + weight * value

        metrics_values = {
            name: value.detach().item()
            for name, value in values.items()
        }

        return total, metrics_values