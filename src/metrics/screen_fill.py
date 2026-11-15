import torch

from data.input import GraphInput
from data.output import Layout
from metrics.batching import node_graph, per_graph


def screen_fill(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """2 - width / W - height / H of the drawing: 0 when it spans the whole image.
    graph can be a PyG batch: the extent is per graph, and the result is the mean over the graphs."""
    positions = layout.positions
    node_graph_index, num_graphs = node_graph(graph, positions.device)

    x = positions[:, 0]
    y = positions[:, 1]
    span_x = per_graph(x, node_graph_index, num_graphs, "amax") - per_graph(x, node_graph_index, num_graphs, "amin")
    span_y = per_graph(y, node_graph_index, num_graphs, "amax") - per_graph(y, node_graph_index, num_graphs, "amin")

    return (2.0 - span_x / layout.width - span_y / layout.height).mean()
