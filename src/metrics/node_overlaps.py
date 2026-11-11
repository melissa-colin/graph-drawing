import math
import torch
from data.raster.sdf import node_coverage_torch, coverage_margin
from .metrics_helpers import overlap_in_box

def node_overlaps(layout, graph, softness=0.5):
    pos = layout.positions            # (N, 2)
    r = graph.node_size               # (N,)
    H, W = layout.height, layout.width
    margin = coverage_margin(softness)

    # check only nearby nodes: DEPENDS ON THE SHAPE
    dist = torch.cdist(pos.detach(), pos.detach())
    # dist < (r1 + r2 + margin)
    close = torch.triu(dist < r[:, None] + r[None, :] + margin, diagonal=1)

    loss = pos.new_zeros(())
    for i, j in close.nonzero().tolist():
        # define the area within which the overlap will be sought
        corner_up = torch.maximum(pos[i] - r[i], pos[j] - r[j]).detach() - margin
        corner_down = torch.minimum(pos[i] + r[i], pos[j] + r[j]).detach() + margin

        # check the coverage
        loss = loss + overlap_in_box(
            corner_up, corner_down, W, H, pos,
            lambda X, Y: node_coverage_torch(X, Y, pos[i], r[i], softness),
            lambda X, Y: node_coverage_torch(X, Y, pos[j], r[j], softness),
        )

    return loss