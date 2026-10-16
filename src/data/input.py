import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch


@dataclass
class GraphInput:
    name: str
    node_ids: list[str]
    edge_index: torch.Tensor  # arêtes canonique du graphe, au sens (u, v), 1 occurrence par liaison
    x: torch.Tensor           # attributs des nœuds pour le modèle
    node_size: torch.Tensor   # rayon de chaque nœud, en pixels
    dist: torch.Tensor  # (N, N) float : distances géodésiques ; dist[i, i] = 0, dist[i, j] = plus court chemin fini
                       # pour des paires non reliées, la valeur est inf (comme dans floyd_warshall_numpy).
                       # IMPORTANT : le stress ne doit utiliser que les paires avec 0 < dist < inf ;
                       # la diagonale et les paires non reliées doivent être exclues avant division par d².
    edge_width: float = 1.0  # largeur de trait pour le rendu visuel

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


def split_dataset(data_dir: Path, test_ratio: float = 0.2, seed: int = 0) -> tuple[list[Path], list[Path]]:
    files = sorted(data_dir.glob("*.graphml"))
    if not files:
        return [], []
    shuffled_files = files[:]
    random.Random(seed).shuffle(shuffled_files)
    n_test = int(round(len(shuffled_files) * test_ratio))
    n_test = max(0, min(len(shuffled_files), n_test))
    return shuffled_files[n_test:], shuffled_files[:n_test]
