"""Demo logic (demo/core.py), tested without launching the interface."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

import core  # noqa: E402
from data.output import Layout  # noqa: E402
from refine import refine  # noqa: E402
from resolution import anchor, balanced_weights, make_loss, rescale_to_frame, term_means, with_style  # noqa: E402


@pytest.mark.parametrize("kind", list(core.GENERATORS))
def test_generators_give_connected_graphs_with_model_features(kind):
    g = core.generate(kind, 20, seed=1)
    assert g.num_nodes >= 4 and g.x.shape == (g.num_nodes, 3 + 8)
    assert len(g.dist_value) == g.num_nodes * (g.num_nodes - 1) // 2  # connected: every pair at finite distance


def test_draw_the_whole_chain():
    g = core.generate("grille", 16)
    d = core.draw(g, "kk", 120, 4.0, 1.0, refine=True, curves=True)
    assert set(core.MEASURE_LABELS) <= set(d.measures)
    assert d.positions.min() >= 4 - 1e-4 and d.positions.max() <= 120 - 4 + 1e-4
    assert len(d.frames) == core.REFINE_STEPS + 1
    assert set(d.seconds) == {"départ", "affinage", "courbes"}
    table = core.measures_table([d, d])
    assert len(table) == len(core.MEASURE_LABELS) and len(table[0]) == 3


def test_refinement_helps_in_a_small_image():
    g = core.generate("géométrique aléatoire", 60, seed=0)
    raw = core.draw(g, "kk", 120, 4.0, 1.0, refine=False)
    refined = core.draw(g, "kk", 120, 4.0, 1.0, refine=True)
    score = lambda d: d.measures["hidden_nodes"] + d.measures["merged_nodes"] + d.measures["ambiguous_edges"]
    assert score(refined) < score(raw)


def test_snapshots_do_not_change_the_refinement():
    g0 = core.generate("arbre", 25, seed=3)
    P = rescale_to_frame(core.kamada_kawai(g0), 150, 4.0)
    g = anchor(with_style(g0, 4.0, 1.0), P)
    loss_fn = make_loss("screen", balanced_weights(term_means([(P, g)], "screen", 150), 0.1))
    ours, frames = core.refine_with_snapshots(P, g, 150, loss_fn, 10, 1.0, 4.0)
    team = refine(Layout(P, 150, 150), g, loss_fn, steps=10, lr=1.0, margin=4.0).positions
    assert torch.allclose(ours, team, atol=1e-5)
    assert len(frames) == 11 and torch.equal(frames[0], P) and torch.equal(frames[-1], ours)


def test_our_model_needs_a_checkpoint_or_a_model():
    g = core.generate("cycle", 10)
    with pytest.raises(ValueError):
        core.draw(g, "ours", 100, 3.0, 1.0)
    from model import LayoutModel
    model = LayoutModel(in_features=g.x.shape[1], hidden=16, width=500, height=500).eval()  # untrained
    d = core.draw(g, "ours", 100, 3.0, 1.0, refine=False, model=model)
    assert d.positions.shape == (g.num_nodes, 2)


def test_render_and_animation(tmp_path):
    g = core.generate("grille", 9)
    d = core.draw(g, "kk", 80, 3.0, 1.0, refine=True, curves=True, steps=4)
    image = core.render(d, scale=3)
    assert image.size == (240, 240)
    path = core.animation(d, tmp_path / "a.gif", scale=1)
    from PIL import Image
    with Image.open(path) as gif:
        assert gif.n_frames >= 5  # start, 4 steps (and the curves if any edge bent)


def test_defects_are_painted_in_red():
    g = core.generate("grille", 16)
    d = core.draw(g, "kk", 60, 6.0, 1.0, refine=False)  # far too small: defects everywhere
    assert d.measures["merged_nodes"] > 0
    plain, painted = core.render(d, 1, defects=False), core.render(d, 1, defects=True)
    import numpy as np
    a, b = np.asarray(plain, dtype=int), np.asarray(painted, dtype=int)
    redder = (b[..., 0] - b[..., 2]) > (a[..., 0] - a[..., 2]) + 20
    assert redder.sum() > 20


def test_app_builds_and_draws(monkeypatch, tmp_path):
    """The interface builds (without launching a server) and its two callbacks run on a generated graph."""
    import app
    monkeypatch.setattr(app, "rome_test_graphs", lambda: [])
    blocks = app.build_app(model=None)
    assert blocks is not None
    g = app.get_graph("Générateur", None, "grille", 9, 0, None)
    assert g.num_nodes == 9
    graph = ["Générateur", None, "grille", 9, 0, None]
    before, after, gif, table, info = blocks.callbacks["draw"](*graph, 80, 3.0, 1.0, "Kamada-Kawai", True, 5, True, True)
    assert before.size == after.size == (160, 160) and Path(gif).exists()
    assert len(table) == len(core.MEASURE_LABELS) and "9 nœuds" in info
    *images, table2 = blocks.callbacks["compare"](*graph, 80, 3.0, 1.0)
    assert len(images) == 3 and len(table2[0]) == 4


def test_rome_test_graphs_are_in_the_test_split(rome_dir):
    """The listed graphs exist and belong to the test split (CoRe-GD's seed): the model never saw them."""
    import json
    names = json.loads((ROOT / "demo" / "rome_test_graphs.json").read_text())
    assert len(names) == 40 and len(set(names)) == 40
    from data.input import split_dataset
    train, val, test = split_dataset(rome_dir)
    test_names = {p.stem for p in test}
    assert set(names) <= test_names
