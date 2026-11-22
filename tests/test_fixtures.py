"""The fixtures build what the rest of the code expects."""
import torch

from data.input import load_graphml


def test_make_graph_path(make_graph):
    g = make_graph([(0, 1), (1, 2), (2, 3)])
    assert g.num_nodes == 4
    assert g.node_ids == ["0", "1", "2", "3"]  # row i is node i
    assert sorted(map(tuple, g.edge_index.t().tolist())) == [(0, 1), (1, 2), (2, 3)]
    # graph distances of a path: |i - j| for every pair i < j
    d = {(int(i), int(j)): float(v) for (i, j), v in zip(g.dist_index.t(), g.dist_value)}
    assert d == {(i, j): float(j - i) for i in range(4) for j in range(i + 1, 4)}


def test_make_graph_sizes(make_graph):
    g = make_graph([(0, 1)], n=3, node_size=[2.0, 3.0, 5.0], edge_width=2.5)
    assert g.num_nodes == 3  # node 2 isolated
    assert torch.equal(g.node_size, torch.tensor([2.0, 3.0, 5.0]))
    assert g.edge_width == 2.5
    # the isolated node has no finite distance: no pair with it
    assert not (g.dist_index == 2).any()


def test_rome_loads(rome_dir):
    g = load_graphml(next(rome_dir.glob("*.graphml")))
    assert g.num_nodes >= 10 and g.x.shape == (g.num_nodes, 3 + 8)
