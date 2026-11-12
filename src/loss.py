"""Training loss: weighted sum of metrics (see metrics/)."""

from typing import Callable

import torch
import torch.nn as nn

from data.input import GraphInput
from data.output import Layout

from metrics.screen_fill import screen_fill
from metrics.stress import stress
from metrics.node_overlaps import node_overlaps
from metrics.edge_overlaps import edge_overlaps
from metrics.edge_angles import edge_angles

LossFn = Callable[[Layout, GraphInput], tuple[torch.Tensor, dict[str, float]]]


class Loss(nn.Module):
    def __init__(self, weights: dict[str, float]):
        """Weights control the contribution of each metric."""
        super().__init__()
        self.weights = weights

    def forward(
        self, layout: Layout, graph: GraphInput
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return total loss and individual active metric values."""

        total = layout.positions.new_zeros(())
        metrics_values = {}

        metrics = {
            "screen_fill": screen_fill,
            "stress": stress,
            "node_overlaps": node_overlaps,
            "edge_overlaps": edge_overlaps,
            "edge_anlges": edge_angles,
        }

        for name, metric_fn in metrics.items():
            weight = self.weights.get(name, 0.0)

            # Skip metrics with zero weight
            if weight == 0.0:
                continue

            value = metric_fn(layout, graph)
            total = total + weight * value
            metrics_values[name] = value.detach().item()

        return total, metrics_values
