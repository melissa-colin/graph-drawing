"""Loss d'entraînement : somme pondérée de métriques (voir metrics/)."""

import torch
import torch.nn as nn

from data.input import GraphInput
from data.output import Layout


class Loss(nn.Module):
    def __init__(self, weights: dict[str, float]):
        """weights : poids de chaque terme, ex. {"stress": 1.0, "overlap_screen": 0.5}."""
        super().__init__()
        self.weights = weights

    def forward(self, layout: Layout, graph: GraphInput) -> tuple[torch.Tensor, dict[str, float]]:
        """Renvoie la loss totale (pour backward) et la valeur de chaque terme (pour le suivi)."""
        # TODO(rôle C)
        raise NotImplementedError
