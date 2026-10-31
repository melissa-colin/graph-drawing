import math
import torch
from data.raster.sdf import node_sdf_torch

def node_overlaps(layout, graph, softness=0.5):
    pos = layout.positions            # (N, 2)
    r = graph.node_size               # (N,)
    H, W = layout.height, layout.width
    m = 4 * softness

    # check only nearby nodes
    dist = torch.cdist(pos.detach(), pos.detach())
    close = torch.triu(dist < r[:, None] + r[None, :] + m, diagonal=1)

    loss = pos.new_zeros(())
    for i, j in close.nonzero().tolist():
        # which pixels to examine ? avoid to examine all the image
        # rectangle with the 2 nodes
        corner_up = torch.maximum(pos[i] - r[i], pos[j] - r[j]).detach() - m
        corner_down = torch.minimum(pos[i] + r[i], pos[j] + r[j]).detach() + m
        # prevents exceeding the boundaries
        x0, y0 = max(math.floor(corner_up[0].item()), 0), max(math.floor(corner_up[1].item()), 0)
        x1, y1 = min(math.ceil(corner_down[0].item()) + 1, W), min(math.ceil(corner_down[1].item()) + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue

        # create pixel grid
        ys = torch.arange(y0, y1, dtype=pos.dtype, device=pos.device)
        xs = torch.arange(x0, x1, dtype=pos.dtype, device=pos.device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")

        # test each pixel
        occ_i = torch.sigmoid(-node_sdf_torch(X, Y, pos[i], r[i]) / softness)
        occ_j = torch.sigmoid(-node_sdf_torch(X, Y, pos[j], r[j]) / softness)
        loss = loss + (occ_i * occ_j).sum()

    return loss