import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch


@dataclass
class GraphInput:
    name: str
    node_ids: list[str]
    edge_index: torch.Tensor  # canonical edges of the graph, (u, v) direction, one occurrence per link
    x: torch.Tensor           # node features for the model
    node_size: torch.Tensor   # radius of each node, in pixels
    dist: torch.Tensor  # (N, N) float: geodesic distances; dist[i, i] = 0, dist[i, j] = finite shortest path
                       # for unconnected pairs, the value is inf (as in floyd_warshall_numpy).
                       # IMPORTANT: the stress must only use pairs with 0 < dist < inf;
                       # the diagonal and unconnected pairs must be excluded before dividing by d².
    edge_width: float = 1.0  # stroke width for visual rendering

    @property
    def num_nodes(self) -> int:
        return self.node_size.shape[0]

    @property
    def edge_index_bidirectional(self) -> torch.Tensor:
        """Edges in both directions: (u, v) and (v, u), for directed message-passing usage."""
        return torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)


def load_graphml(path: Path, node_size: float = 4.0, edge_width: float = 1.0) -> GraphInput:
    graph = nx.read_graphml(path)
    node_ids = [str(node) for node in graph.nodes()]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = [(node_index[u], node_index[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
    features = torch.column_stack((
        torch.tensor([graph.degree(node) for node in node_ids], dtype=torch.float32),
        torch.ones(len(node_ids), dtype=torch.float32),
    ))
    distances_matrix = nx.floyd_warshall_numpy(graph, nodelist=node_ids)
    distances = torch.from_numpy(distances_matrix.astype("float32"))
    return GraphInput(
        name=path.stem,
        node_ids=node_ids,
        edge_index=edge_index,
        x=features,
        node_size=torch.full((len(node_ids),), float(node_size), dtype=torch.float32),
        dist=distances,
        edge_width=float(edge_width),
    )


def split_dataset(data_dir: Path, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 0) -> tuple[list[Path], list[Path], list[Path]]:
    """Split into disjoint train/val/test sets with a reproducible shuffle.

    The ratios are applied on the whole dataset, with train = remaining files after
    reserving validation and test sets. The three sets are disjoint and deterministic.
    """
    files = sorted(data_dir.glob("*.graphml"))
    if not files:
        return [], [], []

    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_ratio))
    n_test = int(round(len(shuffled) * test_ratio))
    n_val = max(0, min(len(shuffled), n_val))
    n_test = max(0, min(len(shuffled) - n_val, n_test))

    val_start = 0
    val_end = n_val
    test_start = n_val
    test_end = n_val + n_test
    train_end = len(shuffled)

    train_files = shuffled[test_end:train_end]
    val_files = shuffled[val_start:val_end]
    test_files = shuffled[test_start:test_end]
    return train_files, val_files, test_files
