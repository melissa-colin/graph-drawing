import math
import torch

def pixel_grid(corner_up, corner_down, W, H, like):
    """Grid of pixel centers (X, Y) covering the box [corner_up, corner_down] in the canva
    """
    # clip to the canvas
    x0, y0 = max(math.floor(corner_up[0].item()), 0), max(math.floor(corner_up[1].item()), 0)
    x1, y1 = min(math.ceil(corner_down[0].item()) + 1, W), min(math.ceil(corner_down[1].item()) + 1, H)
    if x0 >= x1 or y0 >= y1:  # cannot happen while the model keeps the centers inside the canvas
        return None

    ys = torch.arange(y0, y1, dtype=like.dtype, device=like.device)
    xs = torch.arange(x0, x1, dtype=like.dtype, device=like.device)

    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X, Y

def overlap_in_box(corner_up, corner_down, W, H, like, coverage_i, coverage_j):
    """Sum over the pixels of the box of cov_i * cov_j.

    cov_i, cov_j: functions (X, Y) -> coverage in [0, 1] (one per shape).
    Returns 0 if the clipped box is empty.
    """
    grid = pixel_grid(corner_up, corner_down, W, H, like)
    if grid is None:
        return like.new_zeros(())
    X, Y = grid
    return (coverage_i(X, Y) * coverage_j(X, Y)).sum()