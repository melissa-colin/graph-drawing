"""Drawing a graph = the model, then a short direct optimization of its positions.

The model gives a good drawing in one pass; a few Adam steps directly on the positions, with the
training loss, then remove the remaining local defects (overlapping nodes, edges over nodes, small
angles) without degrading the stress. It is the second axis of the subject (direct optimization, graph
by graph), started from the model's drawing instead of a random one.

Chosen on the validation split (runs/eval_post.py): 12 model rounds then 25 steps are better than the
model alone on all five criteria (stress, crossings, overlapping nodes, edges over nodes, angles).
"""
import torch
from torch_geometric.data import Batch

from data.output import Layout
from loss import Loss

ROUNDS = 12   # model rounds at inference (trained around 5: more rounds keep improving the drawing)
STEPS = 25    # Adam steps on the positions
LR = 1.0      # pixels per step, roughly


def refine(layout: Layout, graph, loss_fn: Loss, steps: int = STEPS, lr: float = LR, margin: float = 4.0) -> Layout:
    """`steps` Adam steps on the positions of `layout`, with `loss_fn`; positions kept inside the frame.
    graph can be a PyG batch: each graph keeps its own positions (the loss is a mean over the graphs)."""
    positions = layout.positions.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            loss_fn(Layout(positions, layout.width, layout.height), graph)[0].backward()
            optimizer.step()
            with torch.no_grad():
                positions[:, 0].clamp_(margin, layout.width - margin)
                positions[:, 1].clamp_(margin, layout.height - margin)
    return Layout(positions.detach(), layout.width, layout.height)


def draw(model, graph, loss_fn: Loss, rounds: int = ROUNDS, steps: int = STEPS) -> Layout:
    """Full drawing: the model with `rounds` rounds, then `steps` refinement steps (0: model only)."""
    with torch.no_grad():
        layout = model(graph, rounds)
    return refine(layout, graph, loss_fn, steps) if steps > 0 else layout
