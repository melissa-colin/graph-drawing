"""Tools of the resolution sweep."""
import pytest
import torch

from data.output import Layout
from loss import Loss
from resolution import (TERMS, balanced_weights, defect_score, evaluate, make_loss, refine_all,
                        rescale_to_frame, scale_normalized_stress, term_means, with_style)
from train import REFINE_WEIGHTS

GRID = [(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)] + \
       [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)]


@pytest.mark.parametrize("size,margin", [(100, 2), (500, 4), (150, 8)])
def test_rescale_fills_the_frame(size, margin):
    P = torch.tensor([[3.0, -1.0], [10.0, 4.0], [-7.0, 2.0]])
    Q = rescale_to_frame(P, size, margin)
    assert Q.min() >= margin - 1e-4 and Q.max() <= size - margin + 1e-4
    span = (Q.max(0).values - Q.min(0).values)
    assert float(span.max()) == pytest.approx(size - 2 * margin, rel=1e-5)  # the largest side fills the frame
    # shape kept: ratios of distances unchanged
    assert torch.allclose(torch.pdist(Q) / torch.pdist(Q).max(), torch.pdist(P) / torch.pdist(P).max(), atol=1e-5)
    assert torch.allclose(rescale_to_frame(Q, size, margin), Q, atol=1e-4)  # idempotent


def test_rescale_degenerate():
    assert torch.equal(rescale_to_frame(torch.zeros(3, 2), 100, 4), torch.full((3, 2), 50.0))


def test_with_style_does_not_touch_the_original(make_graph):
    g = make_graph([(0, 1)], node_size=4.0, edge_width=1.0)
    s = with_style(g, 7.0, 3.0)
    assert torch.equal(s.node_size, torch.full((2,), 7.0)) and s.edge_width == 3.0
    assert torch.equal(g.node_size, torch.full((2,), 4.0)) and g.edge_width == 1.0


def test_screen_loss_matches_the_team_refinement_loss(make_graph):
    """make_loss("screen", REFINE_WEIGHTS renamed) is exactly loss.Loss(REFINE_WEIGHTS)."""
    torch.manual_seed(0)
    g = make_graph(GRID, node_size=4.0)
    layout = Layout(torch.rand(16, 2) * 60 + 4, 100, 100)
    rename = {"edge_anlges": "angles", "edge_crossings": "crossings"}
    weights = {rename.get(k, k): v for k, v in REFINE_WEIGHTS.items() if k != "screen_fill"}
    assert float(make_loss("screen", weights)(layout, g)[0]) == pytest.approx(float(Loss(REFINE_WEIGHTS)(layout, g)[0]), rel=1e-6)


def test_unknown_term_is_an_error():
    with pytest.raises(ValueError):
        make_loss("geometric", {"screen_fill": 1.0})


def test_balanced_weights():
    w = balanced_weights({"stress": 3.0, "angles": 0.5, "node_overlaps": 0.0, "crossings": 20.0}, 0.1)
    assert w["stress"] == 1.0
    assert w["angles"] == pytest.approx(0.2) and w["crossings"] == pytest.approx(0.005)
    assert w["node_overlaps"] == pytest.approx(0.005)  # absent at the start: smallest sibling weight


@pytest.mark.parametrize("kind", ["screen", "geometric"])
def test_refinement_removes_the_defects_it_targets(make_graph, kind):
    """A crowded drawing at a small size: both refinements must lower their own defects and the image ones."""
    torch.manual_seed(0)
    size, radius = 100, 4.0
    graphs = [with_style(make_graph(GRID, name=f"g{i}"), radius, 1.0) for i in range(2)]
    starts = [rescale_to_frame(torch.rand(16, 2), 40, 0) + 30 for _ in graphs]  # in a 40 px square: overlaps
    means = term_means(list(zip(starts, graphs)), kind, size)
    assert set(means) == set(TERMS[kind])
    loss_fn = make_loss(kind, balanced_weights(means, 0.3))
    refined = refine_all(starts, graphs, size, loss_fn, steps=80, lr=1.0, margin=radius, batch_size=1)
    before = [evaluate(p, g, size) for p, g in zip(starts, graphs)]
    after = [evaluate(p, g, size) for p, g in zip(refined, graphs)]
    assert sum(map(defect_score, after)) < sum(map(defect_score, before))
    for p in refined:
        assert torch.isfinite(p).all() and p.min() >= radius - 1e-4 and p.max() <= size - radius + 1e-4


def test_refine_all_batches_like_one_by_one(make_graph):
    torch.manual_seed(0)
    graphs = [with_style(make_graph(GRID, name=f"g{i}"), 3.0, 1.0) for i in range(3)]
    starts = [torch.rand(16, 2) * 80 + 10 for _ in graphs]
    loss_fn = make_loss("geometric", balanced_weights(term_means(list(zip(starts, graphs)), "geometric", 100), 0.1))
    # metrics are means over the graphs of a batch: one batch of 3 = 3 graphs with a learning signal / 3,
    # Adam is invariant to that scale, so the results match up to floating-point noise
    together = refine_all(starts, graphs, 100, loss_fn, steps=10, lr=1.0, margin=3.0, batch_size=3)
    alone = refine_all(starts, graphs, 100, loss_fn, steps=10, lr=1.0, margin=3.0, batch_size=1)
    for a, b in zip(together, alone):
        assert torch.allclose(a, b, atol=1e-2)


def test_scale_normalized_stress_is_scale_free(make_graph):
    g = make_graph(GRID)
    P = torch.rand(16, 2)
    assert scale_normalized_stress(P, g) == pytest.approx(scale_normalized_stress(37 * P + 5, g), rel=1e-6)


def test_evaluate_has_every_measure(make_graph):
    g = with_style(make_graph(GRID), 2.0, 1.0)
    m = evaluate(rescale_to_frame(torch.rand(16, 2), 100, 2), g, 100)
    assert {"stress", "crossings", "hidden_nodes", "merged_nodes", "ambiguous_edges"} <= set(m)
