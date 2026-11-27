"""Le pipeline tourne de bout en bout sur un petit graphe (CPU, sans les données Rome) :
le modèle dessine dans le cadre, l'affinage baisse la loss sans sortir du cadre."""
import sys
from pathlib import Path

import networkx as nx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.input import load_graphml  # noqa: E402
from data.output import Layout  # noqa: E402
from loss import Loss  # noqa: E402
from model import LayoutModel  # noqa: E402
from refine import refine  # noqa: E402
from train import REFINE_WEIGHTS  # noqa: E402


def grid(tmp_path):
    """Grille 4 x 4, chargée par le vrai chemin de chargement (.graphml)."""
    path = tmp_path / "grille.graphml"
    nx.write_graphml(nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4)), path)
    return load_graphml(path)


def test_le_modele_dessine_dans_le_cadre(tmp_path):
    g = grid(tmp_path)
    torch.manual_seed(0)
    model = LayoutModel(g.x.shape[1], hidden=16, width=500, height=500).eval()
    with torch.no_grad():
        p = model(g, 3).positions
    assert p.shape == (16, 2) and torch.isfinite(p).all()
    assert (p >= 0).all() and (p <= 500).all()


def test_l_affinage_baisse_la_loss_et_reste_dans_le_cadre(tmp_path):
    g = grid(tmp_path)
    torch.manual_seed(0)
    start = Layout(4 + torch.rand(16, 2) * 125, 500, 500)  # tassé dans un coin : chevauchements, croisements
    loss_fn = Loss(REFINE_WEIGHTS)
    after = refine(start, g, loss_fn, steps=60)
    assert float(loss_fn(after, g)[0]) < float(loss_fn(start, g)[0])
    assert (after.positions >= 4 - 1e-4).all() and (after.positions <= 496 + 1e-4).all()
