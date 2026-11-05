import torch

from data.input import GraphInput
from data.output import Layout


def stress(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Gap between drawn distances and graph distances (graph.dist_index, graph.dist_value). Differentiable.

    sum_{i<j} (||x_i - x_j|| - d_ij)^2 / d_ij^2, after rescaling the drawing to the best scale.
    """
    # TODO(role C)
    raise NotImplementedError
