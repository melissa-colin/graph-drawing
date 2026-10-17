"""Model: a graph as input, screen positions as output.

Version 0: an MLP applied to each node (it ignores the edges), followed by a soft-argmax head.
The MLP will be replaced by a GNN; the soft-argmax head will stay.

Soft-argmax: each node gets a probability map over the grid, and its position is the expectation
of the cell coordinates. Continuous and differentiable, unlike an argmax.
References: DSNT (Nibali et al., 2018), kornia.geometry.subpix.spatial_expectation2d,
mmpose IntegralRegressionHead.
"""

import torch
import torch.nn as nn

from data.input import GraphInput
from data.output import Layout


class NodeMLP(nn.Module):
    """Temporary encoder: node features (F,) -> embedding (D,), node by node."""

    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, F) -> (N, hidden)"""
        return self.net(x)


class SoftArgmaxHead(nn.Module):
    """Embeddings (N, D) -> one probability map (grid_h, grid_w) per node -> positions (N, 2) in pixels."""

    def __init__(self, dim: int, width: int, height: int, margin: float,
                 grid_w: int | None = None, grid_h: int | None = None, temperature: float = 1.0):
        """margin: margin in pixels (the node radius) so that the disks stay inside the image.
        grid_w, grid_h: grid size; defaults to the image size (full resolution).
        temperature: > 1 spreads the distribution, < 1 makes it sharper. The temperature divides
            the logits (logits / T), the usual softmax convention.
        """
        super().__init__()
        self.grid_w = grid_w or width
        self.grid_h = grid_h or height
        self.temperature = temperature
        self.margin = margin
        self.scores = nn.Linear(dim, self.grid_w * self.grid_h)
        centers_x = torch.linspace(margin, width - margin, steps=self.grid_w, dtype=torch.float32)
        centers_y = torch.linspace(margin, height - margin, steps=self.grid_h, dtype=torch.float32)
        self.register_buffer("centers_x", centers_x)
        self.register_buffer("centers_y", centers_y)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (N, D) -> positions (N, 2) in pixels, maps (N, grid_h, grid_w)

        Conventions: row = y (pointing down), column = x.
        """
        logits = self.scores(h) / self.temperature
        probs = logits.softmax(dim=-1).view(-1, self.grid_h, self.grid_w)

        col_probs = probs.sum(dim=1)  # (N, grid_w)
        row_probs = probs.sum(dim=2)  # (N, grid_h)

        x = (col_probs * self.centers_x).sum(dim=-1)
        y = (row_probs * self.centers_y).sum(dim=-1)
        positions = torch.stack((x, y), dim=-1)
        return positions, probs


class LayoutModel(nn.Module):
    """Graph -> position of each node in a width x height image."""

    def __init__(self, in_features: int, hidden: int, width: int, height: int,
                 margin: float = 4.0, temperature: float = 1.0):
        super().__init__()
        self.width = width
        self.height = height
        self.encoder = NodeMLP(in_features, hidden)
        self.head = SoftArgmaxHead(hidden, width, height, margin, temperature=temperature)

    def forward(self, graph: GraphInput) -> Layout:
        """Continuous positions computed by soft-argmax over the output grid.

        Since the coordinates are an expectation over cell centers bounded in [margin, width - margin]
        and [margin, height - margin], the positions automatically stay inside the visible area.
        """
        embeddings = self.encoder(graph.x)
        positions, _ = self.head(embeddings)
        return Layout(positions=positions, width=self.width, height=self.height)
