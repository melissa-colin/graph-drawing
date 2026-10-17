import torch

from data.input import GraphInput
from data.output import Layout


def node_overlap_geometric(layout: Layout, graph: GraphInput) -> torch.Tensor:
    """Node overlap computed geometrically: penalizes ||c_i - c_j|| < r_i + r_j. Differentiable."""
    # TODO(role C)
    raise NotImplementedError


def node_overlap_screen(image: torch.Tensor) -> torch.Tensor:
    """Node overlap computed on the rendered image. Differentiable.

    Idea: use the sum map S = sum_i I_i rather than pair by pair.
    """
    # TODO(role C)
    raise NotImplementedError
