"""Model: a graph as input, screen positions as output.

Architecture adapted from CoRe-GD (Grötschla, Mathys, Veres, Wattenhofer, "CoRe-GD: A Hierarchical
Framework for Scalable Graph Visualization with GNNs", ICLR 2024), MIT license,
https://github.com/floriangroetschla/CoRe-GD (get_mlp, GRUEdgeConv, the rewiring loop).

Changes from the original:
- their sigmoid decoder is replaced by our soft-argmax head (positions in pixels, inside the frame);
- torch_cluster.knn is replaced by a masked torch.cdist + topk (no extra dependency);
- no beacons, no replay buffer, no coarsening (beacons were tried and did not help, see docs/idees/melissa-archi.md).
As in CoRe-GD, train.py draws the number of iterations at random for each batch.

Soft-argmax: each node gets a probability map over the grid, and its position is the expectation
of the cell coordinates. Continuous and differentiable, unlike an argmax.
References: DSNT (Nibali et al., 2018), kornia.geometry.subpix.spatial_expectation2d,
mmpose IntegralRegressionHead.
"""

import copy

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected

from data.input import GraphInput
from data.output import Layout


def get_mlp(in_dim: int, hidden: int, depth: int, out_dim: int, dropout: float = 0.0,
            last_relu: bool = True) -> nn.Sequential:
    """Linear -> LayerNorm -> ReLU -> Dropout blocks, then a final Linear (+ LayerNorm, ReLU)."""
    layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(depth):
        layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden, out_dim))
    if last_relu:
        layers += [nn.LayerNorm(out_dim), nn.ReLU()]
    return nn.Sequential(*layers)


class GRUEdgeConv(MessagePassing):
    """Message = MLP(x_j, x_i), summed over neighbors; the node state is updated by a GRU cell."""

    def __init__(self, dim: int, mlp_edge: nn.Module, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.rnn = nn.GRUCell(dim, dim)
        self.mlp_edge = mlp_edge

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.rnn(self.propagate(edge_index, x=x), x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return self.mlp_edge(torch.cat((x_j, x_i), dim=1))


def knn_edges(positions: torch.Tensor, batch: torch.Tensor, k: int) -> torch.Tensor:
    """k nearest neighbors of each node in drawing space, within its own graph.

    positions: (N, 2), batch: (N,) graph id of each node.
    Returns edge_index (2, E) with edges neighbor -> node (source in row 0, PyG convention).
    Dense (N, N) distances: fine for Rome-sized batches, to revisit for large graphs.
    """
    n = positions.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=positions.device)
    dist = torch.cdist(positions, positions)
    other_graph = batch.unsqueeze(0) != batch.unsqueeze(1)
    dist = dist.masked_fill(other_graph, float("inf"))
    dist.fill_diagonal_(float("inf"))
    values, neighbors = dist.topk(k, dim=1, largest=False)
    targets = torch.arange(n, device=positions.device).unsqueeze(1).expand(-1, k)
    valid = torch.isfinite(values)  # small graphs have fewer than k neighbors
    return torch.stack((neighbors[valid], targets[valid]))


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
    """Graph -> position of each node in a width x height image (CoRe-GD-style recurrent GNN).

    encoder MLP -> `iterations` rounds of [convs on the graph edges, then one conv on the kNN graph
    of the current drawing] -> last convs -> soft-argmax head.
    """

    def __init__(self, in_features: int, hidden: int, width: int, height: int,
                 margin: float = 4.0, temperature: float = 1.0, iterations: int = 5, knn_k: int = 8,
                 convs_per_iteration: int = 2, mlp_depth: int = 2,
                 hidden_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        # Constructor arguments, saved in checkpoints to rebuild the model (infer.load_model).
        self.config = dict(in_features=in_features, hidden=hidden, width=width, height=height,
                           margin=margin, temperature=temperature, iterations=iterations, knn_k=knn_k,
                           convs_per_iteration=convs_per_iteration, mlp_depth=mlp_depth,
                           hidden_factor=hidden_factor, dropout=dropout)
        self.width = width
        self.height = height
        self.iterations = iterations
        self.knn_k = knn_k

        wide = hidden_factor * hidden
        self.encoder = get_mlp(in_features, wide, mlp_depth, hidden, dropout, last_relu=True)
        mlp_edge = get_mlp(2 * hidden, wide, mlp_depth, hidden, dropout, last_relu=True)
        main_conv = GRUEdgeConv(hidden, mlp_edge)
        # Same conv weights reused at every iteration (recurrence); conv_alt runs on the kNN edges.
        self.convs = nn.ModuleList([copy.deepcopy(main_conv) for _ in range(convs_per_iteration)])
        self.conv_alt = copy.deepcopy(main_conv)
        self.head = SoftArgmaxHead(hidden, width, height, margin, temperature=temperature)

    def forward(self, graph: GraphInput, iterations: int | None = None) -> Layout:
        """Continuous positions computed by soft-argmax over the output grid.

        Since the coordinates are an expectation over cell centers bounded in [margin, width - margin]
        and [margin, height - margin], the positions automatically stay inside the visible area.
        Works on a single GraphInput or on a PyG Batch (graph.batch gives the graph of each node).
        iterations: number of rounds, self.iterations by default (train.py draws it at random).
        """
        iterations = iterations or self.iterations
        x = graph.x
        edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        batch = graph.batch if getattr(graph, "batch", None) is not None \
            else torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device)

        x = self.encoder(x)
        for _ in range(iterations - 1):
            for conv in self.convs:
                x = conv(x, edge_index)
            # Intermediate drawing only used to pick neighbors: no gradient through it.
            with torch.no_grad():
                positions, _ = self.head(x)
                rewired = knn_edges(positions, batch, self.knn_k)
            x = self.conv_alt(x, rewired)
        for conv in self.convs:
            x = conv(x, edge_index)

        positions, _ = self.head(x)
        return Layout(positions=positions, width=self.width, height=self.height)
