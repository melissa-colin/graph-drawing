from pathlib import Path

from data.input import load_graphml, split_dataset
from loss import Loss
from model import LayoutModel


def train(data_dir: Path, out_path: Path, epochs: int, lr: float) -> None:
    """Entraîne LayoutModel sur la base et enregistre ses poids dans out_path.

    Étapes : découper la base, charger les graphes, boucle (modèle -> layout -> loss -> backward),
    suivre chaque terme de la loss, sauvegarder le modèle.
    """
    # TODO(rôle B)
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(rôle B) : lire les paramètres (argparse ou fichier de config)
    train(Path("data/rome"), Path("model.pt"), epochs=10, lr=1e-3)
