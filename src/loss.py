"""Training loss: weighted sum of metrics (see metrics/)."""

from typing import Callable
import torch
import torch.nn as nn

from data.input import GraphInput
from data.output import Layout

LossFn = Callable[[Layout, GraphInput], torch.Tensor] # contract

class Loss(nn.Module):
    def __init__(self, weights: dict[str, float]):
        """weights: weight of each term, e.g. {"stress": 1.0, "overlap_screen": 0.5}."""
        super().__init__()
        self.weights = weights

    def forward(self, layout: Layout, graph: GraphInput) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns the total loss (for backward) and the value of each term (for monitoring)."""
        # TODO(role C)
        raise NotImplementedError
