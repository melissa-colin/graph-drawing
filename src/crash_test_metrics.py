from pathlib import Path

import torch

from data.input import load_graphml
from data.output import Layout
from model import LayoutModel
from loss import Loss


def check_finite(name, value):
    if not torch.isfinite(torch.as_tensor(value)).all():
        raise ValueError(f"{name}: NaN ou infini detecte !")


def load_one_graph():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "rome"
    files = sorted(data_dir.glob("*.graphml"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .graphml dans {data_dir}")
    graph = load_graphml(files[0])
    print(f"Graphe teste : {graph.name}")
    print(f"Noeuds : {graph.num_nodes}, aretes : {graph.edge_index.shape[1]}")
    return graph


def test_model():
    """Test du couple modele + loss (ton main actuel)."""
    graph = load_one_graph()
    model = LayoutModel(graph.x.shape[1], hidden=64, width=800, height=600)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = Loss(weights={
        "screen_fill": 1, "screen_inside": 1, "stress": 1,
        "node_overlaps": 1, "edge_overlaps": 1, "edge_anlges": 1,
    })

    model.train()
    for step in range(1, 21):
        optimizer.zero_grad()
        layout = model(graph)
        loss, values = loss_fn(layout, graph)
        check_finite("loss", loss)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                check_finite(f"gradient {name}", param.grad)
        optimizer.step()
        print(f"Etape {step:02d} | loss = {loss.item():.6f} | metriques = {values}")


def test_free_positions():
    """Test de la loss seule : on optimise directement les positions, sans modele."""
    graph = load_one_graph()
    model = LayoutModel(graph.x.shape[1], hidden=64, width=800, height=600)

    with torch.no_grad():
        init = model(graph)  # juste pour obtenir des positions de depart

    pos = init.positions.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=1.0)  # ~1 pixel par pas
    loss_fn = Loss(weights={
        "screen_fill": 0, "screen_inside": 0, "stress": 0,
        "node_overlaps": 0, "edge_overlaps": 1, "edge_anlges": 0,
    })

    for step in range(300):
        opt.zero_grad()
        layout = Layout(pos, init.width, init.height)  # adapte au constructeur de Layout
        loss, values = loss_fn(layout, graph)
        loss.backward()
        if step % 20 == 0:
            print(f"{step:03d} | loss = {loss.item():.4f} | grad norm = {pos.grad.norm().item():.4f}")
        opt.step()


if __name__ == "__main__":
    # Decommente celui que tu veux lancer
    # test_model()
    test_free_positions()