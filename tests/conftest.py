"""Shared test fixtures.

Every test runs on the CPU, without the Rome data, on small graphs built by hand: `make_graph` writes them
as .graphml and loads them with data.input.load_graphml, so the tests go through the real loading path.
Tests that need the Rome graphs use the `rome_dir` fixture and are skipped when data/rome is missing.
"""
import sys
from pathlib import Path

import networkx as nx
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.input import load_graphml  # noqa: E402  (needs src/ on the path)


@pytest.fixture
def make_graph(tmp_path):
    """make_graph(edges, n=None, node_size=4.0, edge_width=1.0, pe_dim=0, name="g") -> GraphInput.

    edges: list of (u, v) with integer nodes 0..n-1; n defaults to 1 + the largest node of `edges`
    (isolated nodes need an explicit n). Nodes keep their order: row i of the GraphInput is node i.
    node_size: a radius for every node (float) or one per node (sequence of length n).
    """
    def build(edges, n=None, node_size=4.0, edge_width=1.0, pe_dim=0, name="g"):
        n = n if n is not None else 1 + max(max(e) for e in edges)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        path = tmp_path / f"{name}.graphml"
        nx.write_graphml(G, path)
        graph = load_graphml(path, node_size=4.0, edge_width=edge_width, pe_dim=pe_dim)
        if isinstance(node_size, (int, float)):
            graph.node_size = torch.full((n,), float(node_size))
        else:
            graph.node_size = torch.as_tensor(node_size, dtype=torch.float32).clone()
            assert graph.node_size.shape == (n,)
        return graph
    return build


@pytest.fixture(scope="session")
def rome_dir():
    path = ROOT / "data" / "rome"
    if not (path / "Graph.log").exists():
        pytest.skip("Rome graphs not found in data/rome")
    return path
