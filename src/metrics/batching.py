"""Metrics on a whole PyG batch at once.

Every metric accepts a single GraphInput or a PyG Batch of them (with the Layout of all its nodes), and
returns the MEAN over the graphs of the per-graph value. Same result as calling it graph by graph and
averaging, but with one call per batch: train.batch_loss no longer loops over the graphs in Python, which
cost one series of small GPU kernels and synchronizations per graph and per metric.
"""
import torch


def node_graph(graph, device):
    """Graph index of each node (N,) and number of graphs B. A single graph: all zeros, B = 1."""
    batch = getattr(graph, "batch", None)
    if batch is None:
        return torch.zeros(graph.num_nodes, dtype=torch.long, device=device), 1
    return batch.to(device), int(graph.num_graphs)


def per_graph(values, graph_index, num_graphs, reduce):
    """Reduce per-item values to one value per graph: reduce = "sum", "amin" or "amax"."""
    out = values.new_zeros(num_graphs)
    return out.scatter_reduce(0, graph_index, values, reduce=reduce, include_self=False)


def edge_width(graph, edge_graph, like):
    """Width of each edge (E,): graph.edge_width is a float for one graph, a (B,) tensor in a batch."""
    width = graph.edge_width
    if torch.is_tensor(width) and width.dim() > 0:
        return width.to(device=like.device, dtype=like.dtype)[edge_graph]
    return like.new_full(edge_graph.shape, float(width))
