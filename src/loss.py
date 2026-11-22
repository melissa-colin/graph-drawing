"""Training loss: weighted sum of metrics (see metrics/)."""

from typing import Callable

import torch
import torch.nn as nn

from data.input import GraphInput
from data.output import Layout

from metrics.stress import scaled_stress, stress
from metrics.overlaps_vectorised import node_overlaps, edge_overlaps, edge_node_overlaps
from metrics.edge_crossings import edge_crossings
from metrics.edge_angles import edge_angles

LossFn = Callable[[Layout, GraphInput], tuple[torch.Tensor, dict[str, torch.Tensor]]]


class Loss(nn.Module):
    def __init__(self, weights: dict[str, float]):
        """Weights control the contribution of each metric."""
        super().__init__()
        self.weights = weights

    def forward(
        self, layout: Layout, graph: GraphInput
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return total loss and individual active metric values."""

        total = layout.positions.new_zeros(())
        metrics_values = {}

        metrics = {
            "stress": stress,
            "scaled_stress": scaled_stress,
            "node_overlaps": node_overlaps,
            "edge_overlaps": edge_overlaps,
            "edge_anlges": edge_angles,
            "edge_crossings": edge_crossings,
            "edge_node_overlaps": edge_node_overlaps,
        }

        for name, metric_fn in metrics.items():
            weight = self.weights.get(name, 0.0)

            # Skip metrics with zero weight
            if weight == 0.0:
                continue

            value = metric_fn(layout, graph)
            total = total + weight * value
            # Tenseur détaché, pas .item() : .item() attend le GPU (une synchronisation par métrique et
            # par graphe, à chaque pas d'entraînement) alors que l'entraînement n'utilise pas ces valeurs.
            # float(v) pour les afficher.
            metrics_values[name] = value.detach()

        return total, metrics_values
