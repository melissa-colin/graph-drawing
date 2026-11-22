"""Tools of the resolution sweep."""
import pytest
import torch

from data.output import Layout
from loss import Loss
from resolution import (TERMS, anchor, anchored_stress, balanced_weights, defect_score, evaluate, make_loss,
                        refine_all, rescale_to_frame, scale_normalized_stress, term_means, with_style)
from train import REFINE_WEIGHTS

from pathlib import Path
ROOT_SRC = Path(__file__).resolve().parent.parent / "src"

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
    """With the team's scale L, make_loss("screen", REFINE_WEIGHTS renamed) is exactly loss.Loss(REFINE_WEIGHTS):
    same terms, and the anchored stress is the team's stress formula with another L."""
    torch.manual_seed(0)
    g = make_graph(GRID, node_size=4.0)
    layout = Layout(torch.rand(16, 2) * 60 + 4, 100, 100)
    team_L = (100 - 2 * 4.0) / float(g.dist_value.max())  # metrics/stress._pixel_distances
    g.stress_scale = torch.tensor([team_L])
    rename = {"edge_anlges": "angles", "edge_crossings": "crossings"}
    weights = {rename.get(k, k): v for k, v in REFINE_WEIGHTS.items() if k != "screen_fill"}
    assert float(make_loss("screen", weights)(layout, g)[0]) == pytest.approx(float(Loss(REFINE_WEIGHTS)(layout, g)[0]), rel=1e-6)


def test_anchor_is_the_best_scale(make_graph):
    g = make_graph(GRID)
    P = torch.rand(16, 2, dtype=torch.float64) * 50
    best = float(anchor(g, P).stress_scale)
    layout = Layout(P, 100, 100)
    def at(L):
        g.stress_scale = torch.tensor([L])
        return float(anchored_stress(layout, g))
    assert at(best) < at(best * 1.01) and at(best) < at(best * 0.99)
    with pytest.raises(ValueError):
        anchored_stress(layout, make_graph(GRID, name="plain"))


def test_anchored_stress_does_not_shrink_a_framed_drawing(make_graph):
    """The team's pixel-scaled stress shrinks a grid put in the frame; the anchored one keeps its size."""
    from metrics.stress import stress
    from resolution import TermLoss
    g = with_style(make_graph(GRID), 4.0, 1.0)
    grid = torch.tensor([[c * 10.0, r * 10.0] for r in range(4) for c in range(4)])
    P = rescale_to_frame(grid + torch.randn(16, 2) * 0.5, 120, 4.0)
    span = lambda Q: float((Q.max(0).values - Q.min(0).values).max())
    team = refine_all([P], [g], 120, TermLoss({"stress": stress}, {"stress": 1.0}), 50, 1.0, 4.0)[0]
    ours = refine_all([P], [anchor(g, P)], 120, make_loss("screen", {"stress": 1.0}), 50, 1.0, 4.0)[0]
    assert span(team) < 0.9 * span(P)  # shrunk by more than 10 %
    assert span(ours) == pytest.approx(span(P), rel=0.03)


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
    starts = [rescale_to_frame(torch.rand(16, 2), 40, 0) + 30 for _ in range(2)]  # in a 40 px square: overlaps
    graphs = [anchor(with_style(make_graph(GRID, name=f"g{i}"), radius, 1.0), P) for i, P in enumerate(starts)]
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
    starts = [torch.rand(16, 2) * 80 + 10 for _ in range(3)]
    graphs = [anchor(with_style(make_graph(GRID, name=f"g{i}"), 3.0, 1.0), P) for i, P in enumerate(starts)]
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


def test_choose_respects_the_stress_budget():
    cands = [({"a": 1}, {"stress": 1.00, "hidden_nodes": .5, "merged_nodes": 0, "ambiguous_edges": 0}),
             ({"a": 2}, {"stress": 1.04, "hidden_nodes": .1, "merged_nodes": 0, "ambiguous_edges": 0}),
             ({"a": 3}, {"stress": 1.30, "hidden_nodes": .0, "merged_nodes": 0, "ambiguous_edges": 0})]
    from resolution import choose
    assert choose(cands, 1.0)[0] == {"a": 2}  # the best within +5 % of stress
    assert choose(cands, 0.5)[0] == {"a": 1}  # nothing within budget: the lowest stress


def test_sweep_setting_end_to_end(make_graph):
    """A tiny sweep: 2 validation and 2 test graphs, 2 starting methods, 1 x 2 candidates."""
    from resolution import sweep_setting
    torch.manual_seed(0)
    g = [make_graph(GRID, name=f"g{i}") for i in range(4)]
    crowded = lambda: torch.rand(16, 2) * 10  # rescaled to the frame anyway
    val = {"kk": [(crowded(), g[0]), (crowded(), g[1])], "ours": [(crowded(), g[0]), (crowded(), g[1])]}
    test = {"kk": [(crowded(), g[2]), (crowded(), g[3])], "ours": [(crowded(), g[2]), (crowded(), g[3])],
            "random": [(crowded(), g[2]), (crowded(), g[3])]}
    rows, selection = sweep_setting(80, 4.0, 1.0, val, test, ["kk", "ours"], fractions=(0.3,),
                                    learning_rates=(0.5, 1.0), steps=5, batch_size=2)
    methods = {r["method"] for r in rows}
    assert methods == {"kk", "ours", "random", "kk+screen", "kk+geometric", "ours+screen", "ours+geometric"}
    assert len(rows) == 7 * 2
    for kind in ("screen", "geometric"):
        assert selection[kind]["lr"] in (0.5, 1.0) and len(selection[kind]["candidates"]) == 2
        assert selection[kind]["weights"]["stress"] == 1.0


def test_evaluate_all_in_a_process_pool_matches_serial(make_graph):
    import multiprocessing
    from resolution import evaluate_all
    torch.manual_seed(0)
    jobs = [(rescale_to_frame(torch.rand(16, 2), 100, 3), with_style(make_graph(GRID, name=f"g{i}"), 3.0, 1.0))
            for i in range(3)]
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        parallel = evaluate_all(jobs, 100, pool)
    assert parallel == evaluate_all(jobs, 100)



def test_pool_survives_many_graphs_under_a_low_descriptor_limit(tmp_path):
    """A sweep sends thousands of graphs to the workers: with torch's default sharing (one file descriptor
    per tensor) the pool hung on "Too many open files" at 1024 descriptors. Run 400 evaluations in a child
    process whose limit is 128 descriptors (the former pool fails there with "Too many open files": checked by
    hand, see the commit)."""
    import subprocess
    import sys as _sys
    script = tmp_path / "many.py"
    script.write_text(f"""
import resource, sys
sys.path[:0] = [{str(ROOT_SRC)!r}]
import torch
from data.input import GraphInput
from resolution import evaluate_all, make_pool, rescale_to_frame, with_style

if __name__ == "__main__":  # spawn re-imports this file in every worker
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    dist = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.float32)
    g = with_style(GraphInput(name="p", node_ids=["0", "1", "2"], edge_index=torch.tensor([[0, 1], [1, 2]]),
                              x=torch.ones(3, 3), node_size=torch.full((3,), 3.0), dist=dist), 3.0, 1.0)
    jobs = [(rescale_to_frame(torch.rand(3, 2), 60, 3), g) for _ in range(400)]
    with make_pool(2) as pool:
        print(len(evaluate_all(jobs, 60, pool)))
""")
    out = subprocess.run([_sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("400")


def test_sweep_setting_extra_controls_only(make_graph):
    from resolution import sweep_setting
    torch.manual_seed(0)
    g = [make_graph(GRID, name=f"m{i}") for i in range(2)]
    d = lambda i: [(torch.rand(16, 2) * 10, g[i])]
    rows, selection = sweep_setting(80, 4.0, 1.0, {"kk": d(0)}, {"kk": d(1)}, ["kk"], fractions=(0.3,),
                                    learning_rates=(1.0,), steps=3, batch_size=1, kinds=("geometric_m1",),
                                    measure_starts=False)
    assert {r["method"] for r in rows} == {"kk+geometric_m1"} and set(selection) == {"geometric_m1"}
