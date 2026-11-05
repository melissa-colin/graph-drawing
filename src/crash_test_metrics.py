from pathlib import Path

import torch

from data.input import load_graphml
from model import LayoutModel
from loss import Loss


def check_finite(name, value):
    if not torch.isfinite(torch.as_tensor(value)).all():
        raise ValueError(f"{name}: NaN ou infini detecte !")


def main():
    # Charger un seul graphe pour un test rapide
    data_dir = Path(__file__).resolve().parent.parent / "data" / "rome"
    files = sorted(data_dir.glob("*.graphml"))

    if not files:
        raise FileNotFoundError(f"Aucun fichier .graphml dans {data_dir}")

    graph = load_graphml(files[0])
    print(f"Graphe teste : {graph.name}")
    print(f"Noeuds : {graph.num_nodes}, aretes : {graph.edge_index.shape[1]}")

    model = LayoutModel(
        graph.x.shape[1],
        hidden=64,
        width=800,
        height=600,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Modifier les coefficients pour choisir les metriques a optimiser
    loss_fn = Loss(weights={
        "screen": 0,
        "stress": 1,
        "node_overlaps": 0,
        "edge_overlaps": 0,
        "edge_anlges": 0,
    })

    model.train()
    print("\n--- Test de quelques iterations ---")

    for step in range(1, 11):
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

    # Verifier une prediction finale
    model.eval()
    with torch.no_grad():
        layout = model(graph)
        check_finite("positions", layout.positions)

    print("\nPrediction OK")
    print("Forme des positions :", tuple(layout.positions.shape))
    print("Tous les tests numeriques sont passes.")


if __name__ == "__main__":
    main()